"""Small learning diagnostics, deliberately separate from final experiments."""

import gc
import random
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import DiagnosticConfig, TrainConfig
from .data import QADataset, move_batch
from .evaluation import evaluate
from .models import build_model
from .runtime import require_mamba_kernels
from .training import Trainer, optimizer_for
from .utils import environment_info, fingerprint, seed_everything, write_json


def diverse_subset(dataset, examples, seed=42):
    """Deterministic subset preferring different training articles and inputs."""
    if examples < 1 or not len(dataset):
        raise ValueError("A diagnostic needs a positive count and nonempty training data.")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected, used, articles, inputs = [], set(), set(), set()
    for criterion in ("article", "input", "remaining"):
        for index in indices:
            if index in used:
                continue
            row = dataset.rows[index]
            signature = tuple(row["input_ids"])
            if criterion == "article" and row["article_id"] in articles:
                continue
            if criterion == "input" and signature in inputs:
                continue
            selected.append(row)
            used.add(index)
            articles.add(row["article_id"])
            inputs.add(signature)
            if len(selected) == min(examples, len(dataset)):
                return QADataset(selected)
    return QADataset(selected)


def _gradient_norms(model):
    groups = {name: 0.0 for name in ("embedding", "encoder", "decoder", "cross_attention", "output")}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        squared = float(parameter.grad.detach().float().square().sum())
        for group, match in (("embedding", "embedding" in name), ("encoder", "encoder" in name),
                             ("decoder", "decoder" in name), ("cross_attention", ".cross." in name),
                             ("output", "output_projection" in name or
                              (model.config.tie_embeddings and name == "embedding.weight"))):
            if match:
                groups[group] += squared
    return {name: value ** .5 for name, value in groups.items()}


def overfit_diagnostic(bundle, model_config, collator, device="cuda", steps=None, examples=None,
                       config=None, output_dir=None):
    """Fresh-model memorization check: constant LR, no warmup or early stopping."""
    config = config or DiagnosticConfig()
    if steps is not None:
        config = replace(config, steps=steps)
    if examples is not None:
        config = replace(config, examples=examples)
    if "mamba" in model_config.variant:
        require_mamba_kernels()
    seed_everything(config.seed)
    dataset = diverse_subset(bundle.train, config.examples, config.seed)
    diagnostic_model = replace(model_config, dropout=0.0) if config.disable_dropout else model_config
    model = build_model(diagnostic_model, bundle.tokenizer, bundle.config).to(device)
    optimizer = optimizer_for(model, TrainConfig(learning_rate=config.learning_rate, weight_decay=0.0))
    output_dir = Path(output_dir) if output_dir is not None else None
    history, losses, last_gradients = [], [], None
    total_tokens = sum(len(row["labels"]) for row in dataset.rows)
    batches = [dataset.rows[i:i + config.micro_batch_size]
               for i in range(0, len(dataset), config.micro_batch_size)]
    try:
        for step in range(config.steps + 1):
            if step % config.eval_every_steps == 0 or step == config.steps:
                metrics = evaluate(model, dataset, collator, bundle.tokenizer, device,
                    bundle.config.max_output_length, config.micro_batch_size,
                    include_teacher_forced=True, sample_limit=len(dataset),
                    prediction_path=output_dir / "predictions.json" if output_dir else None)
                metrics.update({"step": step, "gradient_norms_before_clipping": last_gradients})
                history.append(metrics)
                report = {"variant": model_config.variant, "config": asdict(config),
                    "model_config": asdict(diagnostic_model), "examples": len(dataset),
                    "article_count": len({row["article_id"] for row in dataset.rows}),
                    "example_ids": [row["id"] for row in dataset.rows], "steps": step,
                    "first_loss": losses[0] if losses else None, "last_loss": losses[-1] if losses else None,
                    "metrics": metrics, "history": history, "training_losses": losses,
                    "overfit_demonstrated": metrics["f1"] >= config.target_f1,
                    "note": "Memorization only: diagnostic LR, zero weight decay, no warmup/stopping; "
                            "diagnostic weights never initialize the main run."}
                if output_dir:
                    write_json(output_dir / "report.json", report)
                print(f"{model_config.variant} diagnostic step={step}: "
                      f"loss={metrics['teacher_forced_loss']:.4f}, F1={metrics['f1']:.2f}, "
                      f"first-token EOS={metrics['first_token_eos_percent']:.2f}%")
                for sample in metrics["prediction_samples"][:3]:
                    print(f"  predicted={sample['prediction']!r}; gold={sample['answers'][0]!r}")
                if report["overfit_demonstrated"] or step == config.steps:
                    break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.0
            for rows in batches:
                batch = move_batch(collator(rows), device)
                logits = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       batch["labels"].reshape(-1), ignore_index=-100, reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite diagnostic loss.")
                (loss / total_tokens).backward()
                loss_total += loss.detach().item()
                del logits, loss, batch
            if (step + 1) % config.eval_every_steps == 0 or step + 1 == config.steps:
                last_gradients = _gradient_norms(model)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite diagnostic gradients.")
            optimizer.step()
            losses.append(loss_total / total_tokens)
        if output_dir:
            plot_diagnostic(report, output_dir / "learning.png")
        return report
    finally:
        del optimizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def plot_diagnostic(report, path):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    steps = [entry["step"] for entry in report["history"]]
    for ax, key, title in zip(axes, ("teacher_forced_loss", "f1", "first_token_eos_percent"),
                             ("Teacher-forced loss", "Generated-answer F1", "Immediate EOS (%)")):
        ax.plot(steps, [entry[key] for entry in report["history"]], marker=".")
        ax.set(title=title, xlabel="Diagnostic optimizer step")
        ax.grid(alpha=.2)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_learning_diagnostics(bundle, model_configs, collator, output_root, device="cuda", config=None):
    reports = {}
    config = config or DiagnosticConfig()
    variants = sorted((v for v in model_configs if v != "t5_small"), key=lambda v: (v != "attn_attn", v))
    for variant in variants:
        directory = Path(output_root) / "results" / "diagnostics" / variant / f"seed_{config.seed}"
        reports[variant] = overfit_diagnostic(bundle, model_configs[variant], collator, device,
                                             config=config, output_dir=directory)
    return reports


def diagnostic_allows_training(variant, reports, require_pass=True):
    if variant == "t5_small" or not require_pass:
        return True
    report = reports.get(variant)
    if report is None:
        raise RuntimeError(f"No diagnostic for {variant}. Run the diagnostic cell before training, "
                           "or explicitly set REQUIRE_DIAGNOSTIC_PASS=False.")
    if not report["overfit_demonstrated"]:
        print(f"Skipping {variant}: diagnostic generated F1={report['metrics']['f1']:.2f}. "
              "Inspect results/diagnostics; adjust diagnostic settings before a full run.")
        return False
    return True


def generalization_diagnostic(bundle, model_config, train_config, collator, output_root,
                              device="cuda", examples=2000, development_examples=256, epochs=3,
                              repo_root=None):
    """Optional learning pilot: no official validation evaluation or benchmarking."""
    train = diverse_subset(bundle.train, examples, train_config.sample_order_seed)
    development = diverse_subset(bundle.development, development_examples, train_config.sample_order_seed)
    manifest = {**bundle.manifest, "data_fingerprint": fingerprint({
        "parent": bundle.manifest["data_fingerprint"], "train_ids": [r["id"] for r in train.rows],
        "development_ids": [r["id"] for r in development.rows]})}
    pilot_bundle = replace(bundle, train=train, development=development, manifest=manifest)
    config = replace(train_config, epochs=epochs, early_stopping_patience=0,
                     eval_every_steps=0, development_limit=None)
    directory = Path(output_root) / "results" / "generalization_diagnostic" / model_config.variant
    seed_everything(config.seed)
    if "mamba" in model_config.variant:
        require_mamba_kernels()
    model = build_model(model_config, bundle.tokenizer, bundle.config).to(device)
    environment = environment_info(repo_root)
    trainer = Trainer(model, pilot_bundle, collator, config, directory / "checkpoints", device,
                      source_revision=environment["git_commit"])
    try:
        if (directory / "checkpoints" / "last.pt").exists():
            trainer.resume(directory / "checkpoints" / "last.pt")
        state = trainer.fit()
        report = {"variant": model_config.variant, "train_config": asdict(config), "environment": environment,
                  "train_examples": len(train), "development_examples": len(development),
                  "state": state, "note": "Exploratory check; no official validation used."}
        write_json(directory / "report.json", report)
        return report
    finally:
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
