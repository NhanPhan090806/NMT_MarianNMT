import pytest
import torch

from qa_assignment.config import ModelConfig
from qa_assignment.models import (BidirectionalMamba, ScratchQA, T5QA, generate, reverse_valid)
from qa_assignment.runtime import probe_model
from qa_assignment.data import QACollator


def test_reverse_valid_keeps_padding_out_of_backward_prefix():
    x = torch.arange(12).reshape(2, 6, 1).float()
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
    reversed_x = reverse_valid(x, mask)
    assert reversed_x[0, :, 0].tolist() == [2, 1, 0, 3, 4, 5]
    assert torch.equal(reverse_valid(reversed_x, mask), x)


def test_encoder_padding_cannot_change_real_token_representations(tiny_model):
    tiny_model.eval()
    ids = torch.tensor([[2, 3, 1, 0, 0, 0, 0, 0]])
    mask = ids.ne(0).long()
    changed = ids.clone()
    changed[:, 3:] = 25
    original = tiny_model.encode(ids, mask).hidden
    replaced = tiny_model.encode(changed, mask).hidden
    torch.testing.assert_close(original[:, :3], replaced[:, :3])


def test_decoder_causality_and_cached_logits(tiny_model):
    tiny_model.eval()
    memory = tiny_model.encode(torch.tensor([[2, 3, 1, 0]]), torch.tensor([[1, 1, 1, 0]]))
    prefix = torch.tensor([[0, 4, 5, 6]])
    original, _ = tiny_model.decode_tokens(prefix, memory)
    changed = prefix.clone()
    changed[:, 2:] = 20
    altered, _ = tiny_model.decode_tokens(changed, memory)
    torch.testing.assert_close(original[:, :2], altered[:, :2])
    cache, pieces = None, []
    for index in range(prefix.size(1)):
        logits, cache = tiny_model.decode_tokens(prefix[:, index:index + 1], memory, cache, True)
        pieces.append(logits)
    torch.testing.assert_close(torch.cat(pieces, 1), original, atol=1e-5, rtol=1e-5)


def test_generation_fixed_length_and_early_eos(tiny_model):
    tiny_model.eval()
    ids, mask = torch.tensor([[2, 3, 1]]), torch.ones(1, 3, dtype=torch.long)
    # A deterministic decoder isolates EOS control from randomly initialized weights.
    def eos_decoder(tokens, memory, cache=None, use_cache=False):
        logits = torch.zeros(tokens.size(0), tokens.size(1), 32)
        logits[..., 1] = 10
        return logits, None
    tiny_model.decode_tokens = eos_decoder
    assert generate(tiny_model, ids, mask, 6).shape == (1, 1)
    assert generate(tiny_model, ids, mask, 6, fixed_length=True).shape == (1, 6)


def test_t5_wrapper_forward_backward_and_cached_generation(tiny_bundle):
    from transformers import T5Config, T5ForConditionalGeneration

    config = T5Config(vocab_size=32, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_decoder_layers=1,
                      num_heads=2, dropout_rate=0, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
    model = T5QA(T5ForConditionalGeneration(config), ModelConfig(variant="t5_small"))
    batch = QACollator(tiny_bundle.tokenizer, tiny_bundle.config)(tiny_bundle.train.rows[:2])
    assert probe_model(model, batch, "cpu", output_cap=6)["finite_backward"]


def test_mamba_shells_have_bidirectional_padding_and_incremental_interfaces(monkeypatch, tiny_bundle):
    """Test the integration shell, not Mamba mathematics, with a simple causal mixer."""
    from qa_assignment import models

    class CausalTestMixer(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.projection = torch.nn.Linear(config.d_model, config.d_model)
        def forward(self, x):
            return self.projection(x).cumsum(dim=1)
        def allocate_inference_cache(self, batch_size, length, dtype=None):
            return (torch.zeros(batch_size, 1, self.projection.out_features, dtype=dtype), None)
        def step(self, token, state, unused):
            state.add_(self.projection(token))
            return state.clone(), state, unused

    monkeypatch.setattr(models, "new_mamba", CausalTestMixer)
    batch = QACollator(tiny_bundle.tokenizer, tiny_bundle.config)(tiny_bundle.train.rows[:2])
    for variant in ("mamba_attn", "attn_mamba", "mamba_mamba"):
        config = ModelConfig(variant=variant, d_model=16, heads=2, encoder_layers=1,
                             decoder_layers=1, feedforward_dim=32, dropout=0)
        model = ScratchQA(config, 32, 0, 1, 8, 6)
        assert probe_model(model, batch, "cpu", output_cap=6)["finite_backward"]
        model.eval()
        changed = batch["input_ids"].clone()
        changed[:, 3:] = 20
        torch.testing.assert_close(model.encode(changed, batch["attention_mask"]).hidden[:, :3],
                                   model.encode(batch["input_ids"], batch["attention_mask"]).hidden[:, :3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_attention_and_t5_actual_cuda_probes(tiny_bundle, tiny_model):
    from transformers import T5Config, T5ForConditionalGeneration

    batch = QACollator(tiny_bundle.tokenizer, tiny_bundle.config)(tiny_bundle.train.rows[:2])
    model = tiny_model.cuda()
    for precision in ("fp32", "fp16"):
        assert probe_model(model, batch, "cuda", precision, output_cap=6)["finite_backward"]
    config = T5Config(vocab_size=32, d_model=16, d_ff=32, d_kv=8, num_layers=1, num_decoder_layers=1,
                      num_heads=2, dropout_rate=0, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
    t5 = T5QA(T5ForConditionalGeneration(config), ModelConfig(variant="t5_small")).cuda()
    assert probe_model(t5, batch, "cuda", output_cap=6)["finite_backward"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is required")
def test_real_mamba_variants_when_extensions_are_available(tiny_bundle):
    import platform
    if platform.system() != "Linux":
        pytest.skip("Real Mamba extensions require the Linux Kaggle environment")
    from qa_assignment.runtime import require_mamba_kernels
    require_mamba_kernels()
    batch = QACollator(tiny_bundle.tokenizer, tiny_bundle.config)(tiny_bundle.train.rows[:2])
    for variant in ("mamba_attn", "attn_mamba", "mamba_mamba"):
        config = ModelConfig(variant=variant, d_model=16, heads=2, encoder_layers=1,
                             decoder_layers=1, feedforward_dim=32, dropout=0)
        model = ScratchQA(config, 32, 0, 1, 8, 6).cuda()
        assert probe_model(model, batch, "cuda", output_cap=6)["finite_backward"]
