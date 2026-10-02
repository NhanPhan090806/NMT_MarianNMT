from dataclasses import replace
from pathlib import Path

import pytest
import torch

from qa_assignment.config import BenchmarkConfig, TrainConfig
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
