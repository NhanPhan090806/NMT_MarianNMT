"""Small learning diagnostics, deliberately separate from final experiments."""

import gc

import torch
from torch.nn import functional as F

from .config import TrainConfig
from .data import move_batch
from .evaluation import evaluate
from .models import build_model
from .runtime import require_mamba_kernels
from .training import optimizer_for
from .utils import seed_everything


def overfit_diagnostic(bundle, model_config, collator, device="cuda", steps=300, examples=4):
    if "mamba" in model_config.variant:
        require_mamba_kernels()
    seed_everything(42)
    dataset = bundle.train.first(examples)
    model = build_model(model_config, bundle.tokenizer, bundle.config).to(device)
    optimizer = optimizer_for(model, TrainConfig(learning_rate=3e-3, weight_decay=0.0))
    batch = move_batch(collator(dataset.rows), device)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite overfit diagnostic loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    metrics = evaluate(model, dataset, collator, bundle.tokenizer, device,
                       bundle.config.max_output_length, batch_size=examples)
    report = {"steps": steps, "examples": len(dataset), "first_loss": losses[0], "last_loss": losses[-1],
              "metrics": metrics, "overfit_demonstrated": metrics["f1"] >= 90,
              "note": "A failed tiny-subset check needs investigation or more steps; it is not final QA evidence."}
    del optimizer, model, batch, logits, loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
