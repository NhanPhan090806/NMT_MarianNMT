"""Detokenized translation metrics and synchronized generation measurements."""

import time

import torch
from sacrebleu.metrics import BLEU, CHRF
from torch.nn import functional as F
from torch.utils.data import DataLoader

from qa_assignment.data import move_batch
from qa_assignment.utils import autocast_context, memory_peak, reset_memory, synchronize
from .models import generate


def translation_metrics(predictions, references):
    if not predictions or len(predictions) != len(references):
        raise ValueError("Translation predictions and references must have equal, nonzero lengths.")
    bleu, chrf = BLEU(tokenize="13a"), CHRF(char_order=6, word_order=0, beta=2)
    scores = {"bleu": bleu.corpus_score(predictions, [references]).score,
              "chrf": chrf.corpus_score(predictions, [references]).score}
    return {**scores, "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature())}


@torch.inference_mode()
def evaluate(model, dataset, collator, tokenizer, device, max_output_length,
             batch_size=16, precision="fp32", limit=None, include_predictions=False):
    model.eval()
    rows = dataset.rows if limit is None else dataset.rows[:limit]
    if not rows:
        raise ValueError("Evaluation split is empty.")
    predictions, references, records = [], [], []
    loss_sum, correct, tokens, first_eos, generated_tokens = 0.0, 0, 0, 0, 0
    missing_eos, at_cap = 0, 0
    synchronize(device)
    started = time.perf_counter()
    for batch in DataLoader(rows, batch_size=batch_size, collate_fn=collator, num_workers=0):
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            logits = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   batch["labels"].reshape(-1), ignore_index=-100, reduction="sum")
            output = generate(model, batch, max_output_length)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss.")
        valid = batch["labels"].ne(-100)
        loss_sum += loss.item()
        correct += int((logits.argmax(-1).eq(batch["labels"]) & valid).sum())
        tokens += int(valid.sum())
        first_eos += int(output[:, 0].eq(tokenizer.eos_token_id).sum())
        generated_tokens += int(output.ne(tokenizer.pad_token_id).sum())
        missing_eos += int((~output.eq(tokenizer.eos_token_id).any(dim=1)).sum())
        at_cap += int(output.ne(tokenizer.pad_token_id).sum(dim=1).ge(max_output_length).sum())
        decoded = tokenizer.batch_decode(output.cpu(), skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(batch["references"])
        records.extend({"id": key, "source": source, "reference": ref, "prediction": pred}
                       for key, source, ref, pred in zip(batch["ids"], batch["sources"], batch["references"], decoded))
    synchronize(device)
    metrics = {**translation_metrics(predictions, references), "examples": len(rows),
               "teacher_forced_loss": loss_sum / tokens, "token_accuracy": 100 * correct / tokens,
               "first_token_eos_percent": 100 * first_eos / len(rows),
               "empty_prediction_percent": 100 * sum(not p.strip() for p in predictions) / len(rows),
               "mean_generated_tokens": generated_tokens / len(rows),
               "missing_eos_percent": 100 * missing_eos / len(rows),
               "output_cap_percent": 100 * at_cap / len(rows),
               "evaluation_seconds": time.perf_counter() - started, "samples": records[:5]}
    if include_predictions:
        metrics["predictions"] = records
    return metrics


@torch.inference_mode()
def benchmark(model, bundle, collator, device, precision="fp32", examples=32, repeats=3):
    model.eval()
    rows = bundle.test.rows[:examples]
    measurements = {}
    for batch_size, name in ((1, "latency"), (8, "throughput")):
        batches = [move_batch(collator(rows[i:i + batch_size]), device) for i in range(0, len(rows), batch_size)]
        with autocast_context(device, precision):
            generate(model, batches[0], bundle.config.max_output_length)
        reset_memory(device)
        timings = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            with autocast_context(device, precision):
                for batch in batches:
                    generate(model, batch, bundle.config.max_output_length)
            synchronize(device)
            timings.append(time.perf_counter() - start)
        seconds = sum(timings) / repeats
        measurements[name] = {"batch_size": batch_size, "seconds": timings,
                              "ms_per_sentence": 1000 * seconds / len(rows),
                              "sentences_per_second": len(rows) / seconds, "peak_memory": memory_peak(device)}
    return {"examples": len(rows), "repeats": repeats, "decoding": "greedy, actual EOS stopping",
            "timing_scope": "model-only; excludes tokenization and transfer", **measurements}
