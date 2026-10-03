"""Token-normalized training with resumable optimizer-boundary checkpoints."""

import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import move_batch
from .evaluation import evaluate
from .utils import (autocast_context, memory_peak, reset_memory, restore_rng, rng_state,
                    synchronize, write_json)


def optimizer_for(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or getattr(parameter, "_no_weight_decay", False):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=config.learning_rate)


def atomic_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Includes Python/NumPy RNG state: only load your own trusted Kaggle artifacts.
    return torch.load(path, map_location="cpu", weights_only=False)


def new_stopping_state(best_f1=-1.0):
    return {"best_f1": best_f1, "checks_without_improvement": 0, "stopped": False, "stop_step": None}


class Trainer:
    def __init__(self, model, bundle, collator, config, checkpoint_dir, device, source_revision=None):
        self.model, self.bundle, self.collator, self.config = model, bundle, collator, config
        self.checkpoint_dir, self.device = Path(checkpoint_dir), torch.device(device)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = optimizer_for(model, config)
        batches = math.ceil(len(bundle.train) / config.micro_batch_size)
        self.total_steps = math.ceil(batches / config.accumulation_steps) * config.epochs
        warmup = int(self.total_steps * config.warmup_fraction)

        def multiplier(step):
            if warmup and step < warmup:
                return (step + 1) / warmup
            return max(0.0, (self.total_steps - step) / max(1, self.total_steps - warmup))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, multiplier)
        self.scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
        self.state = {"epoch": 0, "next_batch": 0, "global_step": 0, "best_f1": -1.0,
                      "best_development_loss": None,
                      "history": [], "training_log": [], "examples_seen": 0, "training_seconds": 0.0,
                      "early_stopping": new_stopping_state(),
                      "training_peak_memory": {"allocated_mib": 0.0, "reserved_mib": 0.0}}
        self.contract = {"model": asdict(model.config), "train": asdict(config),
                         "source_revision": source_revision,
                         "data_fingerprint": bundle.manifest["data_fingerprint"],
                         "data_contract": bundle.manifest["contract"]}

    def payload(self):
        return {"format_version": 1, "contract": self.contract, "state": self.state,
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict(),
                "rng": rng_state()}

    def save(self, name="last.pt"):
        atomic_checkpoint(self.checkpoint_dir / name, self.payload())

    def resume(self, path):
        checkpoint = load_checkpoint(path)
        if checkpoint["contract"] != self.contract:
            raise ValueError("Checkpoint configuration/data differ. Resume with identical settings, "
                             "or use a new run directory for a new experiment.")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.state = checkpoint["state"]
        self.state.setdefault("early_stopping", new_stopping_state(self.state["best_f1"]))
        self.state.setdefault("best_development_loss", None)
        restore_rng(checkpoint["rng"])
        print(f"Resumed step {self.state['global_step']}; epoch {self.state['epoch'] + 1}, "
              f"next batch {self.state['next_batch']}.")

    def capture_training_memory(self):
        measured = memory_peak(self.device)
        for key, value in measured.items():
            self.state["training_peak_memory"][key] = max(self.state["training_peak_memory"][key], value)

    def development_evaluation(self):
        self.capture_training_memory()
        metrics = evaluate(self.model, self.bundle.development, self.collator, self.bundle.tokenizer,
                           self.device, self.bundle.config.max_output_length, self.config.eval_batch_size,
                           self.config.precision, self.config.development_limit,
                           include_teacher_forced=True, sample_limit=5)
        metrics.update({"step": self.state["global_step"], "epoch_cursor": self.state["epoch"],
                        "training_seconds": self.state["training_seconds"]})
        self.state["history"].append(metrics)
        development_loss = metrics.get("teacher_forced_loss")
        if development_loss is not None and not math.isfinite(development_loss):
            raise FloatingPointError("Development teacher-forced loss is nonfinite.")
        tied_f1_better_loss = (metrics["f1"] == self.state["best_f1"] and development_loss is not None and
                              (self.state["best_development_loss"] is None or
                               development_loss < self.state["best_development_loss"]))
        if metrics["f1"] > self.state["best_f1"] or tied_f1_better_loss:
            self.state["best_f1"] = metrics["f1"]
            self.state["best_development_loss"] = development_loss
            improved_checkpoint = True
        else:
            improved_checkpoint = False
        self.update_early_stopping(metrics["f1"])
        metrics["early_stopping"] = self.state["early_stopping"].copy()
        if improved_checkpoint:
            self.save("best.pt")
        self.save()
        write_json(self.checkpoint_dir / "history.json", self.state["history"])
        print(f"Internal development: EM={metrics['em']:.2f}, F1={metrics['f1']:.2f}")
        if development_loss is not None:
            print(f"Teacher-forced loss={development_loss:.4f}; "
                  f"token accuracy={metrics['teacher_forced_token_accuracy']:.2f}%; "
                  f"first-token EOS={metrics['first_token_eos_percent']:.2f}%")
            for sample in metrics["prediction_samples"][:3]:
                print(f"  {sample['id']}: {sample['prediction']!r}; gold={sample['answers'][0]!r}")
        self.model.train()
        reset_memory(self.device)

    def update_early_stopping(self, f1):
        config, stopping = self.config, self.state["early_stopping"]
        if not math.isfinite(f1):
            raise FloatingPointError("Development F1 is nonfinite; cannot select a checkpoint or stop training.")
        if not config.early_stopping_patience:
            return
        if f1 > stopping["best_f1"] + config.early_stopping_min_delta:
            stopping["best_f1"], stopping["checks_without_improvement"] = f1, 0
        elif self.state["global_step"] >= config.early_stopping_min_steps:
            stopping["checks_without_improvement"] += 1
        if self.state["global_step"] < config.early_stopping_min_steps:
            stopping["checks_without_improvement"] = 0
        if stopping["checks_without_improvement"] >= config.early_stopping_patience:
            stopping["stopped"], stopping["stop_step"] = True, self.state["global_step"]
            print(f"Early stopping at step {self.state['global_step']}: development F1 has not improved by "
                  f"more than {config.early_stopping_min_delta:g} points for "
                  f"{config.early_stopping_patience} evaluations. Best checkpoint F1={self.state['best_f1']:.2f}.")

    def fit(self):
        config = self.config
        if not len(self.bundle.train):
            raise ValueError("Training dataset is empty.")
        if self.state["early_stopping"]["stopped"]:
            print(f"Run already early-stopped at step {self.state['early_stopping']['stop_step']}; training skipped.")
            self.save()
            return self.state.copy()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        reset_memory(self.device)
        # Step snapshots are saved before development evaluation. If interrupted
        # there, complete the due check before another optimizer update so stopping
        # patience and checkpoint selection follow the uninterrupted run.
        step = self.state["global_step"]
        if step and config.eval_every_steps and step % config.eval_every_steps == 0 and (
                not self.state["history"] or self.state["history"][-1]["step"] != step):
            self.development_evaluation()
            if self.state["early_stopping"]["stopped"]:
                return self.state.copy()
        for epoch in range(self.state["epoch"], config.epochs):
            indices = list(range(len(self.bundle.train)))
            random.Random(config.sample_order_seed + epoch).shuffle(indices)
            batches = [indices[i:i + config.micro_batch_size]
                       for i in range(0, len(indices), config.micro_batch_size)]
            start_batch = self.state["next_batch"] if epoch == self.state["epoch"] else 0
            # An epoch-end last.pt can precede its development evaluation if interrupted.
            if start_batch == len(batches):
                if not self.state["history"] or self.state["history"][-1]["step"] != self.state["global_step"]:
                    self.development_evaluation()
                if self.state["early_stopping"]["stopped"]:
                    return self.state.copy()
                self.state["epoch"], self.state["next_batch"] = epoch + 1, 0
                self.save()
                continue
            loader = DataLoader(self.bundle.train, batch_sampler=batches[start_batch:],
                                collate_fn=self.collator, num_workers=0,
                                generator=torch.Generator().manual_seed(config.sample_order_seed + epoch))
            window_tokens, window_examples, window_loss = 0, 0, 0.0
            synchronize(self.device)
            window_start = time.perf_counter()
            for local_index, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")):
                batch_index = start_batch + local_index
                if window_tokens == 0:
                    window_end = min(len(batches), (batch_index // config.accumulation_steps + 1)
                                     * config.accumulation_steps)
                    window_token_budget = sum(len(self.bundle.train.rows[index]["labels"])
                                              for indices_in_batch in batches[batch_index:window_end]
                                              for index in indices_in_batch)
                batch = move_batch(batch, self.device)
                with autocast_context(self.device, config.precision):
                    logits = self.model(batch["input_ids"], batch["attention_mask"], batch["labels"])
                    loss_sum = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                               batch["labels"].reshape(-1), ignore_index=-100, reduction="sum")
                if not torch.isfinite(loss_sum):
                    raise FloatingPointError("Nonfinite loss: use fp32 and inspect the compatibility probe.")
                tokens = int(batch["labels"].ne(-100).sum())
                # Normalize before FP16 scaling to avoid magnifying a large summed loss.
                self.scaler.scale(loss_sum / window_token_budget).backward()
                window_tokens += tokens
                window_examples += len(batch["ids"])
                window_loss += loss_sum.detach().item()
                del logits, loss_sum, batch
                boundary = ((batch_index + 1) % config.accumulation_steps == 0 or batch_index == len(batches) - 1)
                if not boundary:
                    continue
                self.scaler.unscale_(self.optimizer)
                if window_tokens != window_token_budget:
                    raise ValueError("Prepared target lengths changed during gradient accumulation.")
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.gradient_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradients. The last saved optimizer boundary is resumable.")
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                synchronize(self.device)
                self.state["training_seconds"] += time.perf_counter() - window_start
                self.state["global_step"] += 1
                self.state["examples_seen"] += window_examples
                self.state["epoch"], self.state["next_batch"] = epoch, batch_index + 1
                self.capture_training_memory()
                step = self.state["global_step"]
                self.state["training_log"].append({"step": step, "loss": window_loss / window_tokens,
                                                   "learning_rate": self.scheduler.get_last_lr()[0]})
                if step % config.log_every_steps == 0:
                    print(f"step={step}/{self.total_steps} loss={window_loss / window_tokens:.4f} "
                          f"lr={self.scheduler.get_last_lr()[0]:.3g}")
                if config.save_every_steps and step % config.save_every_steps == 0:
                    self.save()
                    if config.keep_step_checkpoints:
                        self.save(f"step_{step:08d}.pt")
                        snapshots = sorted(self.checkpoint_dir.glob("step_*.pt"))
                        for obsolete in snapshots[:-config.keep_step_checkpoints]:
                            obsolete.unlink()
                if config.eval_every_steps and step % config.eval_every_steps == 0:
                    self.development_evaluation()
                    if self.state["early_stopping"]["stopped"]:
                        return self.state.copy()
                window_tokens, window_examples, window_loss = 0, 0, 0.0
                synchronize(self.device)
                window_start = time.perf_counter()
            if not self.state["history"] or self.state["history"][-1]["step"] != self.state["global_step"]:
                # Save the final optimizer boundary before potentially long evaluation.
                self.save()
                self.development_evaluation()
            if self.state["early_stopping"]["stopped"]:
                return self.state.copy()
            self.state["epoch"], self.state["next_batch"] = epoch + 1, 0
            self.save()
        self.capture_training_memory()
        self.save()
        return self.state.copy()
