"""Controlled seq2seq models: common shells, cross-attention, and real Mamba-1."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


def reverse_valid(x, mask):
    """Reverse valid prefixes, leaving trailing padding at the trailing positions."""
    lengths = mask.long().sum(dim=1)
    positions = torch.arange(x.size(1), device=x.device)[None, :].expand(x.size(0), -1)
    indices = torch.where(positions < lengths[:, None], lengths[:, None] - 1 - positions, positions)
    return x.gather(1, indices.unsqueeze(-1).expand_as(x))


def new_mamba(config):
    try:
        from mamba_ssm import Mamba
    except (ImportError, OSError) as exc:
        raise RuntimeError("Mamba requires the Linux CUDA installation from the experiment notebook. "
                           "No reference implementation is silently substituted.") from exc
    return Mamba(d_model=config.d_model, d_state=config.d_state,
                 d_conv=config.d_conv, expand=config.expand, use_fast_path=True)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.head_dim = config.heads, config.d_model // config.heads
        self.q = nn.Linear(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def split(self, value):
        return value.reshape(value.size(0), value.size(1), self.heads, self.head_dim).transpose(1, 2)

    def project_memory(self, memory):
        return self.split(self.k(memory)), self.split(self.v(memory))

    def forward(self, x, mask=None, causal=False, cache=None, projected=None, use_cache=False):
        query = self.split(self.q(x))
        key, value = projected if projected is not None else self.project_memory(x)
        if cache is not None:
            key, value = torch.cat((cache[0], key), 2), torch.cat((cache[1], value), 2)
        allowed = mask[:, None, None, :].bool() if mask is not None else None
        # During incremental generation a single query may attend to all cached keys.
        if causal and cache is not None and x.size(1) != 1:
            offset = key.size(2) - query.size(2)
            causal_mask = torch.arange(key.size(2), device=x.device)[None, :] <= (
                torch.arange(query.size(2), device=x.device)[:, None] + offset)
            allowed = causal_mask if allowed is None else allowed & causal_mask
        use_causal = causal and cache is None and allowed is None
        if causal and cache is None and allowed is not None:
            allowed = allowed & torch.ones(query.size(2), key.size(2), dtype=torch.bool,
                                           device=x.device).tril()
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=allowed,
                                                dropout_p=self.dropout if self.training else 0.0,
                                                is_causal=use_causal)
        result = result.transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), -1)
        return self.out(result), (key, value) if use_cache else None


class BidirectionalMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.forward_branch, self.backward_branch = new_mamba(config), new_mamba(config)
        self.merge = nn.Linear(config.d_model * 2, config.d_model)

    def forward(self, x, mask):
        x = x * mask.unsqueeze(-1)
        forward = self.forward_branch(x)
        backward = reverse_valid(self.backward_branch(reverse_valid(x, mask)), mask)
        # Trailing padded states are computed, but cannot influence valid causal prefixes.
        return self.merge(torch.cat((forward, backward), -1)) * mask.unsqueeze(-1)


def feedforward(config):
    return nn.Sequential(nn.Linear(config.d_model, config.feedforward_dim), nn.GELU(),
                         nn.Dropout(config.dropout), nn.Linear(config.feedforward_dim, config.d_model))


class EncoderLayer(nn.Module):
    def __init__(self, config, mamba):
        super().__init__()
        self.is_mamba = mamba
        self.mixer = BidirectionalMamba(config) if mamba else Attention(config)
        self.norm1, self.norm2 = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        self.ff, self.dropout = feedforward(config), nn.Dropout(config.dropout)

    def forward(self, x, mask):
        mixed = self.mixer(self.norm1(x), mask) if self.is_mamba else self.mixer(self.norm1(x), mask)[0]
        x = x + self.dropout(mixed)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x * mask.unsqueeze(-1)


class DecoderLayer(nn.Module):
    def __init__(self, config, mamba):
        super().__init__()
        self.is_mamba = mamba
        self.mixer = new_mamba(config) if mamba else Attention(config)
        self.cross = Attention(config)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(config.d_model) for _ in range(3))
        self.ff, self.dropout = feedforward(config), nn.Dropout(config.dropout)

    def forward(self, x, memory_mask, projected_memory, cache=None, use_cache=False):
        normalized = self.norm1(x)
        if self.is_mamba:
            if use_cache:
                if normalized.is_cuda and torch.is_autocast_enabled():
                    normalized = normalized.to(torch.get_autocast_dtype("cuda"))
                if cache is None:
                    cache = self.mixer.allocate_inference_cache(x.size(0), 1, dtype=normalized.dtype)
                pieces = []
                for token in normalized.split(1, dim=1):
                    output, _, _ = self.mixer.step(token, *cache)
                    pieces.append(output)
                mixed = torch.cat(pieces, dim=1)
            else:
                mixed = self.mixer(normalized)
        else:
            mixed, cache = self.mixer(normalized, causal=True, cache=cache, use_cache=use_cache)
        x = x + self.dropout(mixed)
        x = x + self.dropout(self.cross(self.norm2(x), mask=memory_mask,
                                        projected=projected_memory)[0])
        return x + self.dropout(self.ff(self.norm3(x))), cache


@dataclass
class EncodedMemory:
    hidden: torch.Tensor
    mask: torch.Tensor
    projected: list | None = None


class ScratchQA(nn.Module):
    def __init__(self, config, vocab_size, pad_id, eos_id, input_cap, output_cap):
        super().__init__()
        self.config, self.pad_id, self.eos_id = config, pad_id, eos_id
        self.start_id = pad_id  # T5's decoder start-token convention
        self.input_cap, self.output_cap = input_cap, output_cap
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        self.encoder_positions = nn.Embedding(input_cap, config.d_model)
        self.decoder_positions = nn.Embedding(output_cap + 1, config.d_model)
        encoder_mamba, decoder_mamba = config.variant.split("_")
        self.encoder = nn.ModuleList(EncoderLayer(config, encoder_mamba == "mamba")
                                     for _ in range(config.encoder_layers))
        self.decoder = nn.ModuleList(DecoderLayer(config, decoder_mamba == "mamba")
                                     for _ in range(config.decoder_layers))
        self.encoder_norm, self.decoder_norm = nn.LayerNorm(config.d_model), nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.d_model, vocab_size, bias=False)
        if config.tie_embeddings:
            self.output_projection.weight = self.embedding.weight
        # Initialize only our embeddings; never blanket-reinitialize Mamba's dt/A parameters.
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.encoder_positions.weight, std=0.02)
        nn.init.normal_(self.decoder_positions.weight, std=0.02)

    def encode(self, input_ids, attention_mask):
        if input_ids.size(1) > self.input_cap:
            raise ValueError("Input exceeds the configured positional capacity.")
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.dropout(self.embedding(input_ids) * math.sqrt(self.config.d_model)
                         + self.encoder_positions(positions))
        for layer in self.encoder:
            x = layer(x, attention_mask)
        return EncodedMemory(self.encoder_norm(x), attention_mask)

    def decode_tokens(self, tokens, memory, cache=None, use_cache=False):
        offset = cache["length"] if cache is not None else 0
        if offset + tokens.size(1) > self.output_cap + 1:
            raise ValueError("Decoder exceeds its positional capacity.")
        positions = torch.arange(offset, offset + tokens.size(1), device=tokens.device)
        x = self.dropout(self.embedding(tokens) * math.sqrt(self.config.d_model)
                         + self.decoder_positions(positions))
        if memory.projected is None:
            memory.projected = [layer.cross.project_memory(memory.hidden) for layer in self.decoder]
        next_states = []
        for index, layer in enumerate(self.decoder):
            x, state = layer(x, memory.mask, memory.projected[index],
                             cache["layers"][index] if cache is not None else None, use_cache)
            next_states.append(state)
        logits = self.output_projection(self.decoder_norm(x)) / math.sqrt(self.config.d_model)
        return logits, {"length": offset + tokens.size(1), "layers": next_states} if use_cache else None

    def forward(self, input_ids, attention_mask, labels):
        decoder = labels.new_full(labels.shape, self.pad_id)
        decoder[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, self.pad_id)
        return self.decode_tokens(decoder, self.encode(input_ids, attention_mask))[0]


class T5QA(nn.Module):
    def __init__(self, hf_model, config):
        super().__init__()
        self.hf_model, self.config = hf_model, config
        self.pad_id = hf_model.config.pad_token_id
        self.eos_id = hf_model.config.eos_token_id
        self.start_id = hf_model.config.decoder_start_token_id

    def encode(self, input_ids, attention_mask):
        output = self.hf_model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask,
                                            return_dict=True)
        return EncodedMemory(output.last_hidden_state, attention_mask)

    def decode_tokens(self, tokens, memory, cache=None, use_cache=False):
        from transformers.modeling_outputs import BaseModelOutput
        output = self.hf_model(encoder_outputs=BaseModelOutput(last_hidden_state=memory.hidden),
                               attention_mask=memory.mask, decoder_input_ids=tokens,
                               past_key_values=cache, use_cache=use_cache, return_dict=True)
        return output.logits, output.past_key_values if use_cache else None

    def forward(self, input_ids, attention_mask, labels):
        return self.hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                             use_cache=False, return_dict=True).logits


def build_model(config, tokenizer, data_config, pretrained=True):
    if config.variant == "t5_small":
        from transformers import T5ForConditionalGeneration, T5Config
        if pretrained:
            hf_model = T5ForConditionalGeneration.from_pretrained(config.t5_name, revision=config.t5_revision)
        else:
            hf_model = T5ForConditionalGeneration(T5Config.from_pretrained(config.t5_name,
                                                                          revision=config.t5_revision))
        if hf_model.config.vocab_size < len(tokenizer):
            raise ValueError("T5 model vocabulary does not cover the frozen tokenizer.")
        return T5QA(hf_model, config)
    return ScratchQA(config, len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id,
                     data_config.max_input_length, data_config.max_output_length)


@torch.inference_mode()
def generate_from_memory(model, memory, max_new_tokens, fixed_length=False, use_cache=True):
    batch, device = memory.hidden.size(0), memory.hidden.device
    tokens = torch.full((batch, 1), model.start_id, device=device, dtype=torch.long)
    generated, finished, cache = [], torch.zeros(batch, dtype=torch.bool, device=device), None
    for _ in range(max_new_tokens):
        logits, cache = model.decode_tokens(tokens[:, -1:] if use_cache else tokens, memory,
                                            cache=cache, use_cache=use_cache)
        next_token = logits[:, -1].argmax(dim=-1)
        if not fixed_length:
            next_token = torch.where(finished, model.pad_id, next_token)
            finished |= next_token.eq(model.eos_id)
        generated.append(next_token)
        tokens = torch.cat((tokens, next_token[:, None]), dim=1)
        if not fixed_length and finished.all():
            break
    return torch.stack(generated, dim=1)


@torch.inference_mode()
def generate(model, input_ids, attention_mask, max_new_tokens, fixed_length=False, use_cache=True):
    return generate_from_memory(model, model.encode(input_ids, attention_mask),
                                max_new_tokens, fixed_length, use_cache)
