"""Library-backed seq2seq models with a common teacher-forcing/generation interface."""

import math

import torch
from torch import nn

from qa_assignment.models import EncodedMemory, ScratchQA, generate as scratch_generate


def shifted(labels, start_id, pad_id):
    tokens = labels.new_full(labels.shape, pad_id)
    tokens[:, 0] = start_id
    tokens[:, 1:] = labels[:, :-1].masked_fill(labels[:, :-1] == -100, pad_id)
    return tokens


class RecurrentTranslator(nn.Module):
    """Plain tanh RNN or LSTM with Luong-style dot attention (no input feeding)."""

    def __init__(self, config, tokenizer, data_config):
        super().__init__()
        self.config = config
        self.pad_id, self.eos_id, self.start_id = tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id
        self.input_cap, self.output_cap = data_config.max_input_length, data_config.max_output_length
        self.attentive = config.stage == "lstm_attention"
        recurrent = nn.LSTM if self.attentive else nn.RNN
        self.embedding = nn.Embedding(len(tokenizer), config.d_model, padding_idx=self.pad_id)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = recurrent(config.d_model, config.d_model, config.encoder_layers,
                                 batch_first=True, dropout=config.dropout if config.encoder_layers > 1 else 0)
        self.decoder = recurrent(config.d_model, config.d_model, config.decoder_layers,
                                 batch_first=True, dropout=config.dropout if config.decoder_layers > 1 else 0)
        self.hidden_bridge = nn.Linear(config.encoder_layers * config.d_model, config.decoder_layers * config.d_model)
        if self.attentive:
            self.cell_bridge = nn.Linear(config.encoder_layers * config.d_model, config.decoder_layers * config.d_model)
            self.query = nn.Linear(config.d_model, config.d_model, bias=False)
            self.combine = nn.Linear(2 * config.d_model, config.d_model)
        self.output_projection = nn.Linear(config.d_model, len(tokenizer))

    def bridge(self, state, layer):
        return layer(state.transpose(0, 1).reshape(state.size(1), -1)).tanh().reshape(
            state.size(1), self.config.decoder_layers, self.config.d_model).transpose(0, 1).contiguous()

    def encode(self, input_ids, attention_mask):
        lengths = attention_mask.sum(1).cpu()
        if input_ids.size(1) > self.input_cap or (lengths < 1).any():
            raise ValueError("Invalid source length.")
        packed = nn.utils.rnn.pack_padded_sequence(self.dropout(self.embedding(input_ids)),
                                                   lengths, batch_first=True, enforce_sorted=False)
        output, state = self.encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=input_ids.size(1))
        state = (self.bridge(state[0], self.hidden_bridge), self.bridge(state[1], self.cell_bridge)) if self.attentive else (
            self.bridge(state, self.hidden_bridge))
        return EncodedMemory(output, attention_mask, recurrent_state=state)

    def decode_tokens(self, tokens, memory, cache=None, use_cache=False):
        state = memory.recurrent_state if cache is None else cache["state"]
        offset = 0 if cache is None else cache["length"]
        if tokens.size(1) + offset > self.output_cap + 1:
            raise ValueError("Target exceeds decoder capacity.")
        output, state = self.decoder(self.dropout(self.embedding(tokens)), state)
        if self.attentive:
            scores = torch.bmm(self.query(output), memory.hidden.transpose(1, 2)) / math.sqrt(self.config.d_model)
            scores = scores.masked_fill(~memory.mask[:, None, :].bool(), float("-inf"))
            context = torch.bmm(scores.softmax(-1), memory.hidden)
            output = self.combine(torch.cat((output, context), dim=-1)).tanh()
        logits = self.output_projection(self.dropout(output))
        return logits, {"state": state, "length": offset + tokens.size(1)} if use_cache else None

    def forward(self, input_ids, attention_mask, labels):
        return self.decode_tokens(shifted(labels, self.start_id, self.pad_id), self.encode(input_ids, attention_mask))[0]


class TransformerTranslator(ScratchQA):
    """Reuse the tested cached PyTorch SDPA implementation with translation BOS/output."""

    def __init__(self, config, tokenizer, data_config):
        super().__init__(config, len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id,
                         data_config.max_input_length, data_config.max_output_length)
        self.start_id = tokenizer.bos_token_id

    def decode_tokens(self, *args, **kwargs):
        logits, cache = super().decode_tokens(*args, **kwargs)
        # Untied vocabulary projection uses standard unscaled output logits.
        return logits * math.sqrt(self.config.d_model), cache

    def forward(self, input_ids, attention_mask, labels):
        return self.decode_tokens(shifted(labels, self.start_id, self.pad_id), self.encode(input_ids, attention_mask))[0]


class MarianTranslator(nn.Module):
    def __init__(self, config, hf_model):
        super().__init__()
        self.config, self.hf_model = config, hf_model

    def forward(self, input_ids, attention_mask, labels):
        return self.hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                             use_cache=False, return_dict=True).logits


def build_model(config, bundle):
    if config.stage == "marian_en_vi":
        if not bundle.pretrained:
            raise ValueError("Marian requires its own pretrained tokenizer and token IDs.")
        from packaging.version import Version
        if Version(torch.__version__.split("+")[0]) < Version("2.6"):
            raise RuntimeError("This upstream Marian revision supplies .bin weights. Transformers requires "
                               "PyTorch >=2.6 to load them. Use a Kaggle runtime with a newer CUDA PyTorch; "
                               "setup deliberately preserves the preinstalled GPU build.")
        from transformers import MarianMTModel
        hf_model = MarianMTModel.from_pretrained(bundle.config.pretrained_name,
                                                 revision=bundle.config.pretrained_revision)
        if hf_model.config.vocab_size != len(bundle.tokenizer):
            raise ValueError("Pretrained vocabulary and model sizes differ.")
        return MarianTranslator(config, hf_model)
    if bundle.pretrained:
        raise ValueError("Scratch models require the train-only scratch tokenizer.")
    if config.stage == "transformer":
        return TransformerTranslator(config, bundle.tokenizer, bundle.config)
    return RecurrentTranslator(config, bundle.tokenizer, bundle.config)


@torch.inference_mode()
def generate(model, batch, max_new_tokens):
    if isinstance(model, MarianTranslator):
        generated = model.hf_model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                           max_new_tokens=max_new_tokens, max_length=None,
                                           num_beams=1, do_sample=False, use_cache=True)
        return generated[:, 1:]  # HF includes the initial decoder token
    return scratch_generate(model, batch["input_ids"], batch["attention_mask"], max_new_tokens)
