"""Pilots, model runs, portable exports, and historically ordered comparisons."""

import gc
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from qa_assignment.data import QADataset, move_batch
from qa_assignment.training import load_checkpoint, optimizer_for
from qa_assignment.utils import (autocast_context, environment_info, parameter_counts,
                                 reset_memory, seed_everything, synchronize, write_json)
from .config import MILESTONES, ModelConfig, STAGES, TrainConfig
from .data import TranslationCollator
from .evaluation import benchmark, evaluate
from .models import build_model, generate
from .training import TranslationTrainer


def effective_model_config(model):
    """Report the architecture that was built, including pretrained HF settings."""
    if not hasattr(model, "hf_model"):
        return asdict(model.config)
    config = model.hf_model.config
    return {"stage": model.config.stage, "model_type": config.model_type,
            "d_model": config.d_model, "encoder_layers": config.encoder_layers,
            "decoder_layers": config.decoder_layers,
            "encoder_attention_heads": config.encoder_attention_heads,
            "decoder_attention_heads": config.decoder_attention_heads,
            "encoder_ffn_dim": config.encoder_ffn_dim, "decoder_ffn_dim": config.decoder_ffn_dim,
            "dropout": config.dropout, "vocab_size": config.vocab_size,
            "tie_word_embeddings": config.tie_word_embeddings}


def diagnostic_rows(training_rows, examples=8):
    """Select meaningful training sentences across moderate sequence lengths."""
    if examples < 2:
        raise ValueError("Diagnostic requires at least two training examples.")
    def word_count(text):
        return sum(any(char.isalpha() for char in word) for word in text.split())
    eligible = [r for r in training_rows if word_count(r["source"]) >= 4 and
                word_count(r["target"]) >= 4 and max(len(r["input_ids"]), len(r["labels"])) <= 48]
    eligible.sort(key=lambda r: (len(r["input_ids"]) + len(r["labels"]), r["id"]))
    if len(eligible) < examples:
        raise ValueError(f"Diagnostic needs {examples} training pairs with at least four alphabetic words "
                         "on each side and at most 48 tokens; fewer are available.")
    # Sample evenly through the eligible length distribution, excluding trivial phrases.
    return [eligible[round(i * (len(eligible) - 1) / (examples - 1))] for i in range(examples)]


def model_parameter_counts(model):
    if not hasattr(model, "hf_model"):
        return parameter_counts(model)
    hf = model.hf_model
    embeddings = {id(p) for module in (hf.get_input_embeddings(), hf.get_output_embeddings())
                  if module is not None for p in module.parameters()}
    return {"parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "parameters_without_embeddings_output": sum(p.numel() for p in model.parameters()
                                                         if p.requires_grad and id(p) not in embeddings)}


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resource_pilot(bundle, model_config, train_config, device, steps=3):
    seed_everything(train_config.seed)
    model = build_model(model_config, bundle).to(device)
    collator = TranslationCollator(bundle.tokenizer)
    batch = move_batch(collator(bundle.train.rows[:train_config.micro_batch_size]), device)
    optimizer = optimizer_for(model, train_config)
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.precision == "fp16")
    durations, losses = [], []
    model.train()
    for i in range(steps + 1):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_config.precision):
            logits = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), batch["labels"].reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError("Pilot loss is nonfinite; use FP32 and inspect data.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("Pilot gradients are nonfinite.")
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        if i:
            durations.append(time.perf_counter() - start)
            losses.append(loss.item())
    model.eval()
    with autocast_context(device, train_config.precision):
        generated = generate(model, batch, min(16, bundle.config.max_output_length))
    samples = bundle.tokenizer.batch_decode(generated.cpu(), skip_special_tokens=True)[:3]
    result = {"stage": model_config.stage, "mean_microbatch_seconds": sum(durations) / len(durations),
              "approximate_epoch_training_seconds": sum(durations) / len(durations) * (
                  (len(bundle.train) + train_config.micro_batch_size - 1) // train_config.micro_batch_size),
              "estimate_note": "Short-batch estimate; excludes validation/checkpoints and length variation.",
              "losses": losses, "sample_translations": samples, **model_parameter_counts(model)}
    del optimizer, model, logits, loss, batch
    release_memory()
    return result


def learning_diagnostic(bundle, model_config, output_root, device="cpu", steps=1000, examples=8,
                        check_every=100, target_chrf=90.0):
    """Train-only memorization check; discard these weights before actual training."""
    if model_config.stage == "marian_en_vi":
        raise ValueError("The scratch memorization diagnostic is not a pretrained evaluation.")
    seed_everything(42)
    if steps < 1 or check_every < 1:
        raise ValueError("Diagnostic steps and check interval must be positive.")
    rows = diagnostic_rows(bundle.train.rows, examples)
    model = build_model(replace(model_config, dropout=0), bundle).to(device)
    collator = TranslationCollator(bundle.tokenizer)
    batch = move_batch(collator(rows), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    history, result = [], None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError("Diagnostic loss is nonfinite.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Diagnostic gradients are nonfinite.")
        optimizer.step()
        if step % check_every == 0 or step == steps:
            result = evaluate(model, QADataset(rows), collator, bundle.tokenizer, device,
                              bundle.config.max_output_length, batch_size=len(rows), include_predictions=True)
            history.append({"step": step, "chrf": result["chrf"], "bleu": result["bleu"],
                            "loss": result["teacher_forced_loss"], "gradient_norm": float(norm)})
            print(f"Diagnostic {model_config.stage}: step={step} loss={history[-1]['loss']:.4f} chrF={result['chrf']:.2f}")
            if result["chrf"] >= target_chrf:
                break
    report = {"stage": model_config.stage, "overfit_demonstrated": result["chrf"] >= target_chrf,
              "target_chrf": target_chrf, "examples": len(rows), "history": history, "final": result,
              "selection": "train-only length-spread sentences; >=4 alphabetic words per side, <=48 tokens",
              "seconds": time.perf_counter() - started, "settings": {"lr": 3e-3, "dropout": 0, "precision": "fp32"},
              "data_fingerprint": bundle.manifest["data_fingerprint"]}
    write_json(Path(output_root) / "results" / "diagnostics" / model_config.stage / "report.json", report)
    del model, optimizer, batch, logits, loss
    release_memory()
    return report


def export_model(model, bundle, directory):
    """Short self-contained model folder: no parent-run manifest is required for inference."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = {"task": "english_to_vietnamese_translation", "model": asdict(model.config),
              "effective_model_config": effective_model_config(model),
              "data": asdict(bundle.config), "data_fingerprint": bundle.manifest["data_fingerprint"]}
    if model.config.stage == "marian_en_vi":
        model.hf_model.save_pretrained(directory, safe_serialization=True)
        bundle.tokenizer.save_pretrained(directory)
    else:
        from safetensors.torch import save_file
        save_file({name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()},
                  str(directory / "model.safetensors"))
        for path in bundle.tokenizer.directory.glob("joint.*"):
            shutil.copy2(path, directory / path.name)
    write_json(directory / "translation_config.json", config)


def run_experiment(bundle, model_config, train_config, output_root, device="cuda:0", repo_root=None,
                   resume="auto", benchmark_examples=32):
    if train_config.precision != "fp32" and torch.device(device).type != "cuda":
        raise ValueError("Mixed precision requires CUDA.")
    expected = model_config.stage == "marian_en_vi"
    if bundle.pretrained != expected:
        raise ValueError("Choose bundle.for_pretrained() exclusively for Marian.")
    seed_everything(train_config.seed)
    output_root = Path(output_root)
    checkpoints = output_root / "checkpoints" / model_config.stage / f"seed_{train_config.seed}"
    results = output_root / "results" / model_config.stage / f"seed_{train_config.seed}"
    environment = environment_info(repo_root)
    model = build_model(model_config, bundle).to(device)
    collator = TranslationCollator(bundle.tokenizer)
    trainer = TranslationTrainer(model, bundle, collator, train_config, checkpoints, device,
                                 source_revision=environment["git_commit"])
    if resume == "auto" and (checkpoints / "last.pt").exists():
        trainer.resume(checkpoints / "last.pt")
    elif resume not in (None, "auto"):
        trainer.resume(resume)
        if not (checkpoints / "best.pt").exists():
            source_best = Path(resume).parent / "best.pt"
            if source_best.is_file() and load_checkpoint(source_best)["contract"] == trainer.contract:
                shutil.copy2(source_best, checkpoints / "best.pt")
            elif trainer.state["best_chrf"] >= 0:
                raise FileNotFoundError("Restore best.pt beside last.pt to preserve checkpoint selection.")
    elif (checkpoints / "last.pt").exists():
        raise FileExistsError("Choose a fresh output directory or resume the existing run.")
    write_json(checkpoints / "run_config.json", trainer.contract)
    write_json(checkpoints / "data_manifest.json", bundle.manifest)
    write_json(checkpoints / "environment.json", environment)
    state = trainer.fit()
    best = load_checkpoint(checkpoints / "best.pt")
    if best["contract"] != trainer.contract:
        raise ValueError("Best checkpoint belongs to a different run.")
    model.load_state_dict(best["model"])
    model.eval()
    # Release optimizer and full best payload before the inference-memory benchmark.
    selected_step = best["state"]["global_step"]
    del trainer, best
    release_memory()
    metrics = evaluate(model, bundle.test, collator, bundle.tokenizer, device,
                       bundle.config.max_output_length, train_config.eval_batch_size,
                       train_config.precision, include_predictions=True)
    predictions = metrics.pop("predictions")
    write_json(results / "test_predictions.json", predictions)
    measurements = benchmark(model, bundle, collator, device, train_config.precision,
                             examples=benchmark_examples) if benchmark_examples else None
    summary = {"stage": model_config.stage, "seed": train_config.seed, **MILESTONES[model_config.stage],
               "model_config": effective_model_config(model), "requested_model_config": asdict(model_config),
               "train_config": asdict(train_config),
               "data_fingerprint": bundle.manifest["data_fingerprint"], "counts": bundle.manifest["counts"],
               "evaluation_scope": bundle.manifest["evaluation_scope"], "test": metrics, "benchmark": measurements,
               "best_step": selected_step, "best_development_chrf": state["best_chrf"],
               "optimizer_steps": state["global_step"], "training_seconds": state["training_seconds"],
               "training_peak_memory": state["training_peak_memory"], "early_stopping": state["early_stopping"],
               **model_parameter_counts(model), "environment": environment,
               "comparison_note": "Marian has external pretraining, a different vocabulary and capacity; prior IWSLT exposure is not ruled out."}
    export_model(model, bundle, output_root / "models" / model_config.stage / f"seed_{train_config.seed}")
    write_json(results / "summary.json", summary)
    write_json(results / "training_log.json", state["training_log"])
    write_json(results / "history.json", state["history"])
    del model
    release_memory()
    return summary


def collect_results(output_root):
    import pandas as pd
    records = []
    for path in Path(output_root).glob("results/*/seed_*/summary.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["stage"] not in STAGES:
            continue
        timing = result.get("benchmark")
        records.append({"stage": result["stage"], "order": result["order"], "year": result["year"],
                        "pretrained": result["pretrained"], "seed": result["seed"],
                        "bleu": result["test"]["bleu"], "chrf": result["test"]["chrf"],
                        "test_loss": result["test"]["teacher_forced_loss"], "parameters": result["parameters"],
                        "training_seconds": result["training_seconds"],
                        "latency_ms": timing["latency"]["ms_per_sentence"] if timing else None,
                        "sentences_per_second": timing["throughput"]["sentences_per_second"] if timing else None,
                        "training_peak_mib": result["training_peak_memory"]["allocated_mib"],
                        "data_fingerprint": result["data_fingerprint"]})
    table = pd.DataFrame(records)
    if not table.empty:
        if table.data_fingerprint.nunique() != 1:
            raise ValueError("Different prepared corpora cannot be combined. Match data configurations in both notebooks.")
        table = table.sort_values(["order", "seed"]).reset_index(drop=True)
        table.to_csv(Path(output_root) / "results" / "comparison.csv", index=False)
    return table


def plot_comparison(table, output_root):
    import matplotlib.pyplot as plt
    if table.empty:
        return None
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    labels = [f"{r.stage}\n{r.year}" + (" pretrained" if r.pretrained else "") for r in table.itertuples()]
    for axis, key, title in zip(axes.flat, ("bleu", "chrf", "parameters", "training_seconds", "latency_ms", "training_peak_mib"),
                                ("BLEU ↑", "chrF ↑", "Parameters", "Training seconds", "Latency ms/sentence ↓", "Training peak MiB")):
        axis.bar(labels, table[key], color=["#6389a8" if not p else "#d19a46" for p in table.pretrained])
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=20)
    figure.suptitle("Historical progression; score and speed ordering are measured, not guaranteed")
    figure.savefig(Path(output_root) / "results" / "historical_comparison.png", dpi=150)
    return figure


def archive_translation_outputs(output_root):
    # Include portable models and prepared data, but omit downloaded HF caches.
    import zipfile
    output_root = Path(output_root)
    archive = output_root.parent / f"{output_root.name}_artifacts.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for folder in ("checkpoints", "results", "models", "data/prepared"):
            for path in sorted((output_root / folder).rglob("*")):
                if path.is_file() and not path.name.endswith(".tmp"):
                    output.write(path, arcname=path.relative_to(output_root))
    return archive


def restore_artifacts(source_root, output_root):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    if not source_root.is_dir() or source_root == output_root:
        raise ValueError("Choose the extracted prior artifact directory as RESTORE_FROM.")
    folders = [name for name in ("checkpoints", "results", "models", "data") if (source_root / name).is_dir()]
    if not folders:
        raise ValueError("Prior artifacts must contain checkpoints/, results/, models/, or data/.")
    if any((output_root / name).exists() for name in folders):
        raise FileExistsError("Restore into an empty output root; existing artifacts cannot be overwritten.")
    for name in folders:
        shutil.copytree(source_root / name, output_root / name)
    return [str(output_root / name) for name in folders]
