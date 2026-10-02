from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qa_assignment.config import BASELINE_VARIANTS, BenchmarkConfig, TrainConfig
from qa_assignment.data import QACollator
from qa_assignment.evaluation import benchmark
from qa_assignment.models import ScratchQA, build_model
from qa_assignment.training import Trainer, load_checkpoint
from qa_assignment.utils import seed_everything
from qa_assignment.workflow import archive_outputs, collect_results, run_experiment


@pytest.mark.parametrize("variant", ["attn_attn", "rnn_rnn", "lstm_lstm"])
def test_mid_epoch_resume_matches_uninterrupted_training(tiny_bundle, tiny_model, tmp_path, variant):
    config = TrainConfig(epochs=2, micro_batch_size=2, accumulation_steps=3, eval_batch_size=2,
                         save_every_steps=1, keep_step_checkpoints=10, log_every_steps=1)
    collator = QACollator(tiny_bundle.tokenizer, tiny_bundle.config)
    model_config = replace(tiny_model.config, variant=variant)
    tiny_model = build_model(model_config, tiny_bundle.tokenizer, tiny_bundle.config)
    seed_everything(11)
    initial = {key: value.clone() for key, value in tiny_model.state_dict().items()}
    reference = Trainer(tiny_model, tiny_bundle, collator, config, tmp_path / "reference", "cpu")
    reference.fit()
    # 7 examples / batch 2 = 4 batches; first optimizer step saves after batch 3, before partial tail.
    checkpoint = tmp_path / "reference" / "step_00000001.pt"
    assert load_checkpoint(checkpoint)["state"]["next_batch"] == 3
    resumed_model = build_model(model_config, tiny_bundle.tokenizer, tiny_bundle.config)
    resumed_model.load_state_dict(initial)
    resumed = Trainer(resumed_model, tiny_bundle, collator, config, tmp_path / "resumed", "cpu")
    resumed.resume(checkpoint)
    resumed.fit()
    for name, tensor in tiny_model.state_dict().items():
        torch.testing.assert_close(tensor, resumed_model.state_dict()[name], rtol=0, atol=0)
    assert reference.state["global_step"] == resumed.state["global_step"] == 4
    assert reference.state["examples_seen"] == resumed.state["examples_seen"] == 14
    incompatible = Trainer(resumed_model, tiny_bundle, collator, replace(config, learning_rate=1e-3),
                           tmp_path / "bad", "cpu")
    with pytest.raises(ValueError, match="differ"):
        incompatible.resume(checkpoint)


@pytest.mark.parametrize("variant", ["attn_attn", "rnn_rnn", "lstm_lstm"])
def test_end_to_end_workflow_writes_isolated_checkpoints_results_and_archive(tiny_bundle, tiny_model, tmp_path, variant):
    config = TrainConfig(epochs=1, micro_batch_size=2, accumulation_steps=2, eval_batch_size=2,
                         save_every_steps=1, log_every_steps=1)
    timing = BenchmarkConfig(examples=2, repeats=1, warmup_runs=1, throughput_batch_size=2, fixed_output_tokens=3)
    model_config = replace(tiny_model.config, variant=variant)
    summary = run_experiment(tiny_bundle, model_config, config, tmp_path / "outputs",
                             device="cpu", benchmark_config=timing)
    directory = tmp_path / "outputs" / "checkpoints" / variant / "seed_42"
    assert (directory / "last.pt").is_file() and (directory / "best.pt").is_file()
    assert summary["validation"]["examples"] == 2
    assert summary["benchmark"]["fixed_workload"]["measurements"][0]["mean_generated_tokens"] == 3
    assert collect_results(tmp_path / "outputs").shape[0] == 1
    assert collect_results(tmp_path / "outputs", variants=["t5_small"]).empty
    if variant in ("rnn_rnn", "lstm_lstm"):
        assert summary["benchmark"]["attention_backend"] is None
    assert archive_outputs(tmp_path / "outputs").is_file()
    resumed = run_experiment(tiny_bundle, model_config, config, tmp_path / "outputs",
                             device="cpu", benchmark_config=None)
    assert resumed["optimizer_steps"] == summary["optimizer_steps"]


@pytest.mark.parametrize("variant", BASELINE_VARIANTS)
def test_all_baselines_stop_on_dev_f1_select_actual_best_and_stay_stopped(
        tiny_bundle, tiny_model, tmp_path, monkeypatch, variant):
    from qa_assignment import training
    if variant == "t5_small":
        from transformers import T5Config, T5ForConditionalGeneration
        hf_config = T5Config(vocab_size=32, d_model=16, d_ff=32, d_kv=8, num_layers=1,
                             num_decoder_layers=1, num_heads=2, dropout_rate=0,
                             pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
        monkeypatch.setattr(T5ForConditionalGeneration, "from_pretrained",
                            lambda *args, **kwargs: T5ForConditionalGeneration(hf_config))
        monkeypatch.setattr(tiny_bundle.tokenizer, "save_pretrained", lambda *args: None, raising=False)
    scores, seen = iter([10.0, 10.05, 10.08]), []
    def development_score(model, dataset, *args, **kwargs):
        assert dataset is tiny_bundle.development  # Never decide stopping from final validation.
        score = next(scores)
        seen.append(score)
        return {"em": score, "f1": score, "examples": len(dataset)}
    monkeypatch.setattr(training, "evaluate", development_score)
    config = TrainConfig(epochs=8, micro_batch_size=2, accumulation_steps=2, eval_batch_size=2,
                         eval_every_steps=1, save_every_steps=1, log_every_steps=1,
                         early_stopping_patience=2, early_stopping_min_delta=0.1)
    model_config = replace(tiny_model.config, variant=variant)
    output_root = tmp_path / "outputs"
    summary = run_experiment(tiny_bundle, model_config, config, output_root,
                             device="cpu", benchmark_config=None)
    assert summary["optimizer_steps"] == 3  # early-stops inside the second epoch
    assert summary["early_stopping"]["stopped"] and summary["early_stopping"]["stop_step"] == 3
    assert summary["selected_step"] == 3 and summary["best_development_f1"] == 10.08
    directory = output_root / "checkpoints" / variant / "seed_42"
    saved = load_checkpoint(directory / "last.pt")
    assert saved["state"]["next_batch"] == 2  # at an optimizer/accumulation boundary
    assert saved["state"]["early_stopping"]["checks_without_improvement"] == 2
    assert saved["state"]["history"][-1]["early_stopping"]["stopped"]
    # A step snapshot precedes its scheduled evaluation; resumption must complete
    # that check and stop before performing another optimizer update.
    monkeypatch.setattr(training, "evaluate", lambda *args, **kwargs: {"em": 10.08, "f1": 10.08})
    interrupted = Trainer(build_model(model_config, tiny_bundle.tokenizer, tiny_bundle.config),
                          tiny_bundle, QACollator(tiny_bundle.tokenizer, tiny_bundle.config),
                          config, tmp_path / "interrupted", "cpu")
    interrupted.resume(directory / "step_00000003.pt")
    recovered = interrupted.fit()
    assert recovered["global_step"] == 3 and recovered["early_stopping"]["stopped"]
    resumed = run_experiment(tiny_bundle, model_config, config, output_root,
                             device="cpu", benchmark_config=None)
    assert resumed["optimizer_steps"] == 3 and len(seen) == 3
    assert collect_results(output_root).stopped_early.tolist() == [True]


def test_early_stopping_patience_resumes_before_stop(tiny_bundle, tiny_model, tmp_path, monkeypatch):
    from qa_assignment import training
    monkeypatch.setattr(training, "evaluate", lambda *args, **kwargs: {"em": 10.0, "f1": 10.0})
    config = TrainConfig(epochs=5, micro_batch_size=2, accumulation_steps=2, eval_every_steps=1,
                         early_stopping_patience=2)
    collator = QACollator(tiny_bundle.tokenizer, tiny_bundle.config)
    trainer = Trainer(tiny_model, tiny_bundle, collator, config, tmp_path / "original", "cpu")
    trainer.state["global_step"] = 1
    trainer.development_evaluation()
    trainer.state["global_step"] = 2
    trainer.development_evaluation()
    resumed = Trainer(build_model(tiny_model.config, tiny_bundle.tokenizer, tiny_bundle.config),
                      tiny_bundle, collator, config, tmp_path / "resumed", "cpu")
    resumed.resume(tmp_path / "original" / "last.pt")
    assert resumed.state["early_stopping"]["checks_without_improvement"] == 1
    resumed.state["global_step"] = 3
    resumed.development_evaluation()
    assert resumed.state["early_stopping"]["stopped"]


def test_stopping_waits_for_minimum_steps_resets_after_improvement_and_can_be_disabled(
        tiny_bundle, tiny_model, tmp_path):
    config = TrainConfig(early_stopping_patience=2, early_stopping_min_delta=0.1,
                         early_stopping_min_steps=5)
    trainer = Trainer(tiny_model, tiny_bundle, QACollator(tiny_bundle.tokenizer, tiny_bundle.config),
                      config, tmp_path, "cpu")
    for step, f1, expected in [(1, 10, 0), (2, 10, 0), (4, 10, 0), (5, 10, 1),
                               (6, 10.2, 0), (7, 10.2, 1), (8, 10.2, 2)]:
        trainer.state["global_step"] = step
        trainer.update_early_stopping(f1)
        assert trainer.state["early_stopping"]["checks_without_improvement"] == expected
    assert trainer.state["early_stopping"]["stopped"]
    disabled = Trainer(tiny_model, tiny_bundle, trainer.collator, TrainConfig(), tmp_path / "disabled", "cpu")
    for _ in range(10):
        disabled.update_early_stopping(0.0)
    assert not disabled.state["early_stopping"]["stopped"]


@pytest.mark.parametrize("setting", [{"early_stopping_patience": -1},
                                    {"early_stopping_min_steps": -1},
                                    {"early_stopping_min_delta": -0.1},
                                    {"early_stopping_min_delta": float("nan")}])
def test_invalid_stopping_settings_are_rejected(setting):
    with pytest.raises(ValueError, match="stopping"):
        TrainConfig(**setting)
