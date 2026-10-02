"""Reference-aware QA evaluation and warmed, synchronized generation benchmarks."""

import statistics
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import move_batch
from .metrics import score_predictions
from .models import generate, generate_from_memory, model_backends
from .utils import autocast_context, memory_peak, reset_memory, synchronize, write_json


@torch.inference_mode()
def evaluate(model, dataset, collator, tokenizer, device, output_cap, batch_size=8,
             precision="fp32", limit=None, prediction_path=None):
    model.eval()
    subset = dataset if limit is None else dataset.first(limit)
    if not len(subset):
        raise ValueError("Cannot evaluate an empty dataset.")
    loader = DataLoader(subset, batch_size=batch_size, collate_fn=collator, num_workers=0)
    predictions, references, lengths = {}, {}, []
    reset_memory(device)
    synchronize(device)
    started = time.perf_counter()
    for batch in tqdm(loader, desc="QA evaluation", leave=False):
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            generated = generate(model, batch["input_ids"], batch["attention_mask"], output_cap)
        generated = generated.cpu().tolist()
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for key, text, answers, ids in zip(batch["ids"], texts, batch["answers"], generated):
            predictions[key], references[key] = text, answers
            lengths.append(ids.index(model.eos_id) + 1 if model.eos_id in ids else len(ids))
    synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = score_predictions(predictions, references)
    metrics.update({"evaluation_seconds": elapsed, "evaluation_ms_per_example": 1000 * elapsed / len(subset),
                    "mean_generated_tokens": statistics.mean(lengths), "peak_memory": memory_peak(device)})
    if prediction_path is not None:
        write_json(prediction_path, {"metrics": metrics, "predictions": predictions, "references": references})
    return metrics


@torch.inference_mode()
def benchmark(model, dataset, collator, device, output_cap, config, precision="fp32"):
    """Input tensors already reside on GPU; excludes tokenization and CPU transfer."""
    model.eval()
    subset = dataset.first(config.examples)
    if not len(subset):
        raise ValueError("Cannot benchmark an empty dataset.")
    if config.fixed_output_tokens > output_cap:
        raise ValueError("Fixed benchmark workload must not exceed the decoder output cap.")
    warm_batch = move_batch(collator(subset.rows[:config.throughput_batch_size]), device)
    for _ in range(config.warmup_runs):
        with autocast_context(device, precision):
            generate(model, warm_batch["input_ids"], warm_batch["attention_mask"], output_cap)
            generate(model, warm_batch["input_ids"], warm_batch["attention_mask"],
                     config.fixed_output_tokens, fixed_length=True)
    synchronize(device)
    del warm_batch
    results = {"example_ids": [row["id"] for row in subset.rows], "repeats": config.repeats,
               "precision": precision, "timing_scope": "model execution; excludes tokenization/transfers/loading",
               "decode_scope": "includes initial cache and cross-attention projection preparation",
               "cache_enabled": True, "model_backends": model_backends(model.config),
               "attention_backend": model_backends(model.config)["attention"]}
    for name, batch_size, fixed in (("qa_latency", 1, False),
                                   ("qa_throughput", config.throughput_batch_size, False),
                                   ("fixed_workload", config.throughput_batch_size, True)):
        repeat_rows = []
        for _ in range(config.repeats):
            loader = DataLoader(subset, batch_size=batch_size, collate_fn=collator, num_workers=0)
            durations, encoding_seconds, decoding_seconds, generated_lengths = [], 0.0, 0.0, []
            reset_memory(device)
            for batch in loader:
                batch = move_batch(batch, device)
                with autocast_context(device, precision):
                    synchronize(device)
                    start = time.perf_counter()
                    memory = model.encode(batch["input_ids"], batch["attention_mask"])
                    synchronize(device)
                    encoded = time.perf_counter()
                    tokens = generate_from_memory(model, memory,
                                                  config.fixed_output_tokens if fixed else output_cap,
                                                  fixed_length=fixed)
                    synchronize(device)
                    ended = time.perf_counter()
                durations.append(ended - start)
                encoding_seconds += encoded - start
                decoding_seconds += ended - encoded
                for ids in tokens.cpu().tolist():
                    generated_lengths.append(len(ids) if fixed or model.eos_id not in ids
                                             else ids.index(model.eos_id) + 1)
                del memory, tokens, batch
            total = sum(durations)
            repeat_rows.append({"seconds": total, "mean_ms_per_sample": 1000 * total / len(subset),
                                "median_batch_ms": 1000 * statistics.median(durations),
                                "samples_per_second": len(subset) / total,
                                "generated_tokens_per_second": sum(generated_lengths) / decoding_seconds,
                                "mean_generated_tokens": statistics.mean(generated_lengths),
                                "encoding_seconds": encoding_seconds, "decoding_seconds": decoding_seconds,
                                "peak_memory": memory_peak(device)})
        results[name] = {"batch_size": batch_size, "measurements": repeat_rows,
                         "mean_ms_per_sample": statistics.mean(r["mean_ms_per_sample"] for r in repeat_rows),
                         "samples_per_second": statistics.mean(r["samples_per_second"] for r in repeat_rows)}
    return results
