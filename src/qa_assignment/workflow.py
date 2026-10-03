"""High-level notebook entry points; long implementation stays in Python files."""

import gc
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

from .config import BenchmarkConfig
from .data import QACollator
from .evaluation import benchmark, evaluate
from .models import build_model, model_backends
from .runtime import probe_model, require_mamba_kernels
from .training import Trainer, load_checkpoint
from .utils import environment_info, parameter_counts, seed_everything, write_json


def run_experiment(bundle, model_config, train_config, output_root, device="cuda", resume="auto",
                   benchmark_config=BenchmarkConfig(), repo_root=None):
    output_root = Path(output_root)
    run_name = Path(model_config.variant) / f"seed_{train_config.seed}"
    checkpoint_dir, result_dir = output_root / "checkpoints" / run_name, output_root / "results" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Resolve an existing run before touching its metadata or loading pretrained weights.
    resume_path = checkpoint_dir / "last.pt" if resume == "auto" else Path(resume) if resume else None
    if resume_path is not None and not resume_path.exists():
        if resume != "auto":
            raise FileNotFoundError(resume_path)
        resume_path = None
    if resume_path is None and (checkpoint_dir / "last.pt").exists():
        raise FileExistsError("Run already exists. Use resume='auto' or a different output root/seed.")
    kernels = require_mamba_kernels() if "mamba" in model_config.variant else model_backends(model_config)
    seed_everything(train_config.seed)
    environment = environment_info(repo_root)
    model = build_model(model_config, bundle.tokenizer, bundle.config).to(device)
    collator = QACollator(bundle.tokenizer, bundle.config)
    trainer = Trainer(model, bundle, collator, train_config, checkpoint_dir, device,
                      source_revision=environment["git_commit"])
    if resume_path is not None:
        trainer.resume(resume_path)
        # Imported last.pt may have an adjacent best.pt; preserve checkpoint selection history.
        if not (checkpoint_dir / "best.pt").exists():
            source_best = resume_path.parent / "best.pt"
            if source_best.exists():
                import shutil
                if load_checkpoint(source_best)["contract"] != trainer.contract:
                    raise ValueError("Adjacent best.pt belongs to a different run.")
                shutil.copy2(source_best, checkpoint_dir / "best.pt")
            elif trainer.state["best_f1"] >= 0:
                raise FileNotFoundError("Resume needs best.pt as well as last.pt to preserve selection. "
                                        "Restore the complete per-variant checkpoint folder.")
    probe = probe_model(model, collator(bundle.train.rows[:min(2, len(bundle.train))]), device,
                        train_config.precision, bundle.config.max_output_length)
    metadata = {"model_config": asdict(model_config), "train_config": asdict(train_config),
                "data_fingerprint": bundle.manifest["data_fingerprint"], "data_contract": bundle.manifest["contract"],
                "environment": environment, "kernel_paths": kernels, "probe": probe,
                **parameter_counts(model)}
    write_json(checkpoint_dir / "run_config.json", metadata)
    write_json(checkpoint_dir / "data_manifest.json", bundle.manifest)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (checkpoint_dir / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    state = trainer.fit()
    contract = trainer.contract
    # Release optimizer state before measuring inference memory.
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    selected = load_checkpoint(checkpoint_dir / "best.pt")
    if selected["contract"] != contract:
        raise ValueError("Best checkpoint belongs to a different run.")
    model.load_state_dict(selected["model"])
    selected_state = selected["state"]
    del selected
    gc.collect()
    validation = evaluate(model, bundle.validation, collator, bundle.tokenizer, device,
                          bundle.config.max_output_length, train_config.eval_batch_size,
                          train_config.precision, prediction_path=result_dir / "validation_predictions.json")
    timings = benchmark(model, bundle.validation, collator, device, bundle.config.max_output_length,
                        benchmark_config, train_config.precision) if benchmark_config is not None else None
    write_json(result_dir / "benchmark.json", timings)
    summary = {"variant": model_config.variant, "seed": train_config.seed,
               "group": ("transfer_learning" if model_config.variant == "t5_small" else
                         "recurrent_baseline" if model_config.variant in ("rnn_rnn", "lstm_lstm") else "controlled"),
               "pretrained": model_config.variant == "t5_small", "model_backends": model_backends(model_config),
               "train_config": asdict(train_config),
               "evaluation_scope": bundle.manifest["evaluation_scope"],
               "data_fingerprint": bundle.manifest["data_fingerprint"], "train_examples": len(bundle.train),
               "validation": validation, "training_seconds": state["training_seconds"],
               "selected_checkpoint_training_seconds": selected_state["training_seconds"],
               "optimizer_steps": state["global_step"], "examples_seen": state["examples_seen"],
               "early_stopping": state["early_stopping"],
               "training_peak_memory": state["training_peak_memory"], "selected_step": selected_state["global_step"],
               "best_development_f1": selected_state["best_f1"], "precision": train_config.precision,
               "selected_development_loss": selected_state.get("best_development_loss"),
               "checkpoint_selection": "development F1; lower teacher-forced loss breaks exact F1 ties",
               "benchmark": timings, **parameter_counts(model)}
    write_json(result_dir / "summary.json", summary)
    write_json(result_dir / "training_log.json", state["training_log"])
    if model_config.variant == "t5_small":
        model.hf_model.save_pretrained(checkpoint_dir / "hf_export", safe_serialization=True)
        bundle.tokenizer.save_pretrained(checkpoint_dir / "hf_export")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def collect_results(output_root, variants=None):
    import pandas as pd

    rows = []
    for path in sorted((Path(output_root) / "results").glob("*/seed_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if variants is not None and summary["variant"] not in variants:
            continue
        timings = summary["benchmark"]
        rows.append({"variant": summary["variant"], "seed": summary["seed"], "group": summary["group"],
                     "pretrained": summary.get("pretrained", summary["variant"] == "t5_small"),
                     "em": summary["validation"]["em"], "f1": summary["validation"]["f1"],
                     "training_seconds": summary["training_seconds"],
                     "qa_ms_per_sample": timings["qa_latency"]["mean_ms_per_sample"] if timings else None,
                     "throughput": timings["qa_throughput"]["samples_per_second"] if timings else None,
                     "training_peak_mib": summary["training_peak_memory"]["allocated_mib"],
                     "inference_peak_mib": max(r["peak_memory"]["allocated_mib"]
                                               for r in timings["qa_latency"]["measurements"]) if timings else None,
                     "parameters": summary["parameters"], "precision": summary["precision"],
                     "epochs": summary.get("train_config", {}).get("epochs"),
                     "learning_rate": summary.get("train_config", {}).get("learning_rate"),
                     "stopped_early": summary.get("early_stopping", {}).get("stopped", False),
                     "optimizer_steps": summary["optimizer_steps"],
                     "selected_step": summary["selected_step"],
                     "data_fingerprint": summary["data_fingerprint"]})
    table = pd.DataFrame(rows)
    if rows:
        if variants is not None:
            table = table.sort_values("variant", key=lambda values: values.map(
                {variant: index for index, variant in enumerate(variants)})).reset_index(drop=True)
        table.to_csv(Path(output_root) / "results" / "comparison.csv", index=False)
    return table


def plot_results(table, output_root, combine_groups=False):
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("No completed results to plot.")
    # Never silently combine incompatible datasets into a scientific comparison.
    if table.data_fingerprint.nunique() != 1:
        raise ValueError("Result data fingerprints differ; compare matching prepared data only.")
    figures = []
    groups = [("seq2seq_baselines", table)] if combine_groups else table.groupby("group")
    for group, group_table in groups:
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        labels = group_table.apply(lambda row: f"{row.variant}\ns{row.seed}" +
                                  (" (pretrained)" if row.variant == "t5_small" else ""), axis=1)
        for axis, column, title in zip(axes.flat, ("f1", "em", "training_seconds", "qa_ms_per_sample", "inference_peak_mib"),
                                       ("F1", "Exact match", "Training seconds", "QA latency (ms/sample)", "Inference peak memory (MiB)")):
            axis.bar(labels, group_table[column])
            axis.set_title(title)
            axis.tick_params(axis="x", labelrotation=25)
        axis = axes.flat[-1]
        valid = group_table.dropna(subset=["qa_ms_per_sample"])
        axis.scatter(valid.qa_ms_per_sample, valid.f1)
        for row in valid.itertuples():
            axis.annotate(f"{row.variant}:s{row.seed}", (row.qa_ms_per_sample, row.f1), fontsize=8)
        axis.set(xlabel="QA latency (ms/sample); lower is faster", ylabel="F1", title="Quality versus latency")
        path = Path(output_root) / "results" / f"{group}_comparison.png"
        figure.savefig(path, dpi=160)
        figures.append(figure)
    return figures


def archive_outputs(output_root, include_data=False):
    import zipfile

    output_root = Path(output_root)
    archive = output_root.parent / f"{output_root.name}_artifacts.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for directory in ("checkpoints", "results", "data") if include_data else ("checkpoints", "results"):
            for path in sorted((output_root / directory).rglob("*")):
                if path.is_file() and not path.name.endswith(".tmp"):
                    output.write(path, arcname=str(path.relative_to(output_root)))
    return archive
