"""Audit extracted Kaggle NMT outputs without downloading data or training models.

Example: python scripts/analyze_nmt_runs.py --samples 256 --device cuda:0
"""

import argparse
import gc
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from nmt_assignment.config import STAGES
from nmt_assignment.data import TranslationCollator
from nmt_assignment.evaluation import evaluate, translation_metrics
from nmt_assignment.inference import Translator
from nmt_assignment.workflow import effective_model_config
from qa_assignment.data import QADataset
from qa_assignment.utils import environment_info


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def has_repeated_fourgram(text):
    words = text.split()
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    return any(count >= 2 for count in grams.values())


def length_group(row):
    words = len(row["source"].split())
    return "1-10 words" if words <= 10 else "11-20 words" if words <= 20 else "21-60 words"


def representative_rows(rows, count, seed=2026):
    """Deterministic random sample within proportional source-length strata."""
    if count < 1 or count > len(rows):
        raise ValueError("Sample size must be positive and fit the split.")
    groups = {name: [r for r in rows if length_group(r) == name]
              for name in ("1-10 words", "11-20 words", "21-60 words")}
    quotas = {name: count * len(values) // len(rows) for name, values in groups.items()}
    remaining = count - sum(quotas.values())
    order = sorted(groups, key=lambda name: (count * len(groups[name]) % len(rows), name), reverse=True)
    for name in order[:remaining]:
        quotas[name] += 1
    rng, selected = random.Random(seed), []
    for name, values in groups.items():
        selected.extend(rng.sample(values, quotas[name]))
    rng.shuffle(selected)
    return selected


def verify_corpus(directory, manifest):
    rows = {}
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.jsonl"
        if file_hash(path) != manifest["file_hashes"][path.name]:
            raise ValueError(f"Prepared corpus hash mismatch: {path}")
        rows[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    sources = [{row["source"] for row in rows[name]} for name in rows]
    if sources[0] & sources[1] or sources[0] & sources[2] or sources[1] & sources[2]:
        raise ValueError("Source overlap across prepared splits.")
    return rows


def audit_run(path, samples, device):
    summary = read_json(path)
    stage, seed = summary["stage"], summary["seed"]
    output_root = path.parents[3]
    export = output_root / "models" / stage / f"seed_{seed}"
    config = read_json(export / "translation_config.json")
    if config["data_fingerprint"] != summary["data_fingerprint"]:
        raise ValueError(f"Export and result use different data: {path}")
    matches = [p for p in (output_root / "data" / "prepared").glob("*/manifest.json")
               if read_json(p)["data_fingerprint"] == summary["data_fingerprint"]]
    if len(matches) != 1:
        raise ValueError("Expected exactly one matching prepared corpus.")
    manifest = read_json(matches[0])
    rows = verify_corpus(matches[0].parent, manifest)
    predictions = read_json(path.parent / "test_predictions.json")
    if [(r["id"], r["source"], r["reference"]) for r in predictions] != [
            (r["id"], r["source"], r["target"]) for r in rows["test"]]:
        raise ValueError("Saved predictions do not match the prepared test rows.")
    scores = translation_metrics([r["prediction"] for r in predictions], [r["reference"] for r in predictions])
    for metric in ("bleu", "chrf"):
        if abs(scores[metric] - summary["test"][metric]) > 1e-6:
            raise ValueError(f"Saved {metric} cannot be reproduced: {path}")
    history = read_json(path.parent / "history.json")
    record = {"summary_path": display_path(path), "stage": stage, "seed": seed,
              "data_fingerprint": summary["data_fingerprint"], "counts": summary["counts"],
              "test": summary["test"], "parameters": summary["parameters"],
              "training_seconds": summary["training_seconds"], "best_step": summary["best_step"],
              "optimizer_steps": summary["optimizer_steps"], "early_stopping": summary["early_stopping"],
              "benchmark": summary["benchmark"], "reported_model_config": summary["model_config"],
              "history": [{k: h[k] for k in ("step", "epoch_cursor", "bleu", "chrf", "teacher_forced_loss")}
                          for h in history], "test_scores_reproduced": True,
              "unique_test_predictions": len({r["prediction"] for r in predictions}),
              "repeated_fourgram_percent": 100 * sum(has_repeated_fourgram(r["prediction"]) for r in predictions) / len(predictions),
              "length_groups": {}}
    for name in ("1-10 words", "11-20 words", "21-60 words"):
        selected = [r for r in predictions if length_group(r) == name]
        if not selected:
            continue
        record["length_groups"][name] = {"examples": len(selected),
            **translation_metrics([r["prediction"] for r in selected], [r["reference"] for r in selected])}
    record["translation_examples"] = [next(r for r in predictions if length_group(r) == name)
                                       for name in record["length_groups"]]
    translator = Translator.from_export(export, device=device)
    record["effective_model_config"] = effective_model_config(translator.model)
    # These are the user's own trusted training checkpoints, including Python RNG state.
    checkpoint_dir = output_root / "checkpoints" / stage / f"seed_{seed}"
    for name in ("best", "last"):
        payload = torch.load(checkpoint_dir / f"{name}.pt", map_location="cpu", weights_only=False)
        record[f"{name}_checkpoint"] = {"step": payload["state"]["global_step"],
            "epoch_cursor": payload["state"]["epoch"], "next_batch": payload["state"]["next_batch"],
            "data_matches": payload["contract"]["data_fingerprint"] == summary["data_fingerprint"],
            "has_optimizer_scheduler_rng": all(k in payload for k in ("optimizer", "scheduler", "rng")),
            "scheduler_last_epoch": payload["scheduler"]["last_epoch"]}
        if not record[f"{name}_checkpoint"]["data_matches"]:
            raise ValueError("Checkpoint data contract differs from summary.")
        expected_step = summary["best_step"] if name == "best" else summary["optimizer_steps"]
        if payload["state"]["global_step"] != expected_step:
            raise ValueError("Checkpoint step differs from summary.")
        if name == "best":
            current = translator.model.state_dict()
            if current.keys() != payload["model"].keys() or any(
                    not torch.equal(value.cpu(), payload["model"][key]) for key, value in current.items()):
                raise ValueError("Export weights differ from the selected best checkpoint.")
            record["export_matches_best_checkpoint"] = True
        del payload
        gc.collect()
    if samples:
        record["sampled_evaluation"] = {}
        for split in ("train", "validation"):
            selected = representative_rows(rows[split], min(samples, len(rows[split])))
            if stage == "marian_en_vi":
                selected = [{**r, "input_ids": r["pretrained_input_ids"], "labels": r["pretrained_labels"]}
                            for r in selected]
            record["sampled_evaluation"][split] = evaluate(translator.model, QADataset(selected),
                TranslationCollator(translator.tokenizer), translator.tokenizer, device,
                translator.config.max_output_length, batch_size=8)
            record["sampled_evaluation"][split]["ids"] = [r["id"] for r in selected]
            print(stage, split, {k: round(v, 4) for k, v in record["sampled_evaluation"][split].items()
                               if k in ("teacher_forced_loss", "bleu", "chrf", "missing_eos_percent")}, flush=True)
    del translator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def plot_report(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    names = [r["stage"] for r in report["runs"]]
    for metric, axis in zip(("bleu", "chrf"), axes[0]):
        bars = axis.bar(names, [r["test"][metric] for r in report["runs"]])
        axis.bar_label(bars, fmt="%.2f")
        axis.set_title(f"Held-out test {metric}")
        axis.set_ylim(0, max(r["test"][metric] for r in report["runs"]) * 1.2)
    for run in report["runs"]:
        epochs = [h["epoch_cursor"] + 1 for h in run["history"]]
        axes[1, 0].plot(epochs, [h["chrf"] for h in run["history"]], marker="o", label=run["stage"])
        axes[1, 1].plot(epochs, [h["teacher_forced_loss"] for h in run["history"]], marker="o", label=run["stage"])
    axes[1, 0].set_title("Development chrF (fixed 512 rows)")
    axes[1, 1].set_title("Development token cross-entropy")
    for axis in axes[1]:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=.2)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "kaggle_runs")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "nmt_run_audit.json")
    parser.add_argument("--samples", type=int, default=256, help="Per-split evaluation rows; 0 skips generation.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    torch.set_num_threads(1)
    paths = sorted(args.runs.rglob("summary.json"), key=lambda p: STAGES.index(read_json(p)["stage"]))
    logs = {display_path(p): file_hash(p) for p in args.runs.rglob("*.log")}
    report = {"audit_environment": environment_info(ROOT), "logs_sha256": logs,
              "duplicated_logs": len(logs) > len(set(logs.values())),
              "sample_method": "seed 2026; proportional source-word-length strata, separately sampled train/dev",
              "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        print("Auditing", path, flush=True)
        report["runs"].append(audit_run(path, args.samples, args.device))
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len({r["data_fingerprint"] for r in report["runs"]}) != 1:
        raise ValueError("Runs do not share a single prepared corpus.")
    plot_report(report, args.output.with_suffix(".png"))
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
