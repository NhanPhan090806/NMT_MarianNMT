import json
import math
from dataclasses import replace

import pytest
import torch

from qa_assignment.config import DiagnosticConfig, TrainConfig
from qa_assignment.data import QACollator
from qa_assignment.diagnostics import (diagnostic_allows_training, diverse_subset,
                                       generalization_diagnostic, overfit_diagnostic)
from qa_assignment.evaluation import evaluate
from qa_assignment.training import Trainer, load_checkpoint


def test_diagnostic_selects_diverse_articles_reproducibly(tiny_bundle):
    subset = diverse_subset(tiny_bundle.train, 4, 42)
    assert len(subset) == len({r["article_id"] for r in subset.rows}) == 4
    assert subset.rows == diverse_subset(tiny_bundle.train, 4, 42).rows
    assert len(diverse_subset(tiny_bundle.train, 100)) == len(tiny_bundle.train)
    with pytest.raises(ValueError):
        diverse_subset(tiny_bundle.train, 0)


def test_diagnostic_persists_answers_curves_and_finite_gradient_paths(tiny_bundle, tiny_model, tmp_path):
    config = DiagnosticConfig(examples=4, steps=3, eval_every_steps=2, micro_batch_size=2)
    report = overfit_diagnostic(tiny_bundle, tiny_model.config,
        QACollator(tiny_bundle.tokenizer, tiny_bundle.config), "cpu", config=config, output_dir=tmp_path)
    assert report["steps"] == 3
    assert [item["step"] for item in report["history"]] == [0, 2, 3]
    assert len(report["training_losses"]) == 3
    assert len(report["metrics"]["prediction_samples"]) == 4
    norms = report["metrics"]["gradient_norms_before_clipping"]
    assert all(math.isfinite(value) and value > 0 for value in norms.values())
    predictions = json.loads((tmp_path / "predictions.json").read_text())
    assert set(predictions["predictions"]) == set(report["example_ids"])
    assert (tmp_path / "report.json").exists() and (tmp_path / "learning.png").exists()
    assert not list(tmp_path.glob("*.pt"))  # Diagnostic weights never become main-run checkpoints.


def test_failed_diagnostic_skips_scratch_but_allows_t5_and_explicit_override():
    failed = {"attn_attn": {"overfit_demonstrated": False, "metrics": {"f1": 0}}}
    assert not diagnostic_allows_training("attn_attn", failed)
    assert diagnostic_allows_training("t5_small", {})
    assert diagnostic_allows_training("attn_attn", failed, require_pass=False)
    with pytest.raises(RuntimeError, match="Run the diagnostic"):
        diagnostic_allows_training("lstm_lstm", failed)


def test_evaluation_records_teacher_forcing_and_eos_without_changing_qa_scores(tiny_bundle, tiny_model):
    collator = QACollator(tiny_bundle.tokenizer, tiny_bundle.config)
    plain = evaluate(tiny_model, tiny_bundle.development, collator, tiny_bundle.tokenizer, "cpu", 6)
    measured = evaluate(tiny_model, tiny_bundle.development, collator, tiny_bundle.tokenizer, "cpu", 6,
                        include_teacher_forced=True, sample_limit=2)
    assert plain["em"] == measured["em"] and plain["f1"] == measured["f1"]
    assert measured["teacher_forced_loss"] > 0
    assert 0 <= measured["teacher_forced_token_accuracy"] <= 100
    assert 0 <= measured["first_token_eos_percent"] <= 100
    assert len(measured["prediction_samples"]) == 2


def test_equal_f1_selects_lower_dev_loss_without_resetting_stopping(tiny_bundle, tiny_model, tmp_path, monkeypatch):
    from qa_assignment import training

    losses = iter([9.0, 7.0, 8.0])
    monkeypatch.setattr(training, "evaluate", lambda *args, **kwargs: {
        "em": 0, "f1": 0, "teacher_forced_loss": next(losses),
        "teacher_forced_token_accuracy": 0, "first_token_eos_percent": 100, "prediction_samples": []})
    trainer = Trainer(tiny_model, tiny_bundle, QACollator(tiny_bundle.tokenizer, tiny_bundle.config),
                      TrainConfig(early_stopping_patience=2), tmp_path, "cpu")
    for step in (1, 2, 3):
        trainer.state["global_step"] = step
        trainer.development_evaluation()
    best = load_checkpoint(tmp_path / "best.pt")
    assert best["state"]["global_step"] == 2
    assert best["state"]["best_development_loss"] == 7.0
    assert trainer.state["early_stopping"]["stopped"]  # Loss tiebreak does not alter F1 patience.


def test_small_generalization_pilot_evaluates_only_internal_development(tiny_bundle, tiny_model, tmp_path, monkeypatch):
    from qa_assignment import training

    original_evaluate = training.evaluate
    seen = []
    def internal_only(model, dataset, *args, **kwargs):
        seen.append([row["id"] for row in dataset.rows])
        assert set(seen[-1]) <= {row["id"] for row in tiny_bundle.development.rows}
        return original_evaluate(model, dataset, *args, **kwargs)
    monkeypatch.setattr(training, "evaluate", internal_only)
    report = generalization_diagnostic(tiny_bundle, tiny_model.config, TrainConfig(),
        QACollator(tiny_bundle.tokenizer, tiny_bundle.config), tmp_path, "cpu",
        examples=4, development_examples=2, epochs=1)
    assert len(seen) == 1
    assert report["train_examples"] == 4 and report["state"]["global_step"] == 1


@pytest.fixture
def local_t5_export(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    tokenizer = Tokenizer(WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "Denver": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    model = T5ForConditionalGeneration(T5Config(vocab_size=4, d_model=16, d_ff=32, d_kv=8,
        num_layers=1, num_decoder_layers=1, num_heads=2,
        pad_token_id=0, eos_token_id=1, decoder_start_token_id=0))
    directory = tmp_path / "checkpoints" / "t5_small" / "seed_42" / "hf_export"
    model.save_pretrained(directory, safe_serialization=True)
    fast.save_pretrained(directory)
    (directory.parent / "data_manifest.json").write_text(json.dumps({
        "contract": {"config": {"max_input_length": 32, "max_output_length": 6}}}))
    return directory


def test_t5_export_discovery_loading_prompt_limits_and_reuse(local_t5_export, monkeypatch):
    from qa_assignment.inference import T5Answerer, find_t5_exports

    assert find_t5_exports([local_t5_export.parents[3]]) == [local_t5_export.resolve()]
    qa = T5Answerer.from_export(local_t5_export, "cpu")
    assert not qa.model.training and qa.max_input_length == 32 and qa.max_output_length == 6
    calls = []
    def answer_tokens(**kwargs):
        calls.append(kwargs)
        return torch.tensor([[0, 3, 1]])
    monkeypatch.setattr(qa.model, "generate", answer_tokens)
    assert qa.answer("Who won?", "Denver won.") == "Denver"
    assert len(qa.answer_many(["Who?", "Where?"], "Denver won.")) == 2
    assert len(qa.history) == len(calls) == 3
    assert all(call["max_new_tokens"] == 6 and call["do_sample"] is False for call in calls)
    expected = qa.tokenizer("question: Who won? context: Denver won.", return_tensors="pt")["input_ids"]
    torch.testing.assert_close(calls[0]["input_ids"], expected)
    with pytest.raises(ValueError, match="no text was truncated"):
        qa.answer("Who?", "word " * 40)
    with pytest.raises(ValueError, match="nonempty question"):
        qa.answer("", "Denver won.")
    with pytest.raises(ValueError, match="passage"):
        qa.answer("Who?", "")
    with pytest.raises(ValueError, match="max_new_tokens"):
        qa.answer("Who?", "Denver won.", max_new_tokens=7)
    assert len(calls) == 3  # Invalid inputs never reach generation.
