"""Kaggle setup and runtime checks: fail explicitly on unsupported Mamba kernels."""

import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import move_batch
from .models import build_model, generate
from .training import optimizer_for
from .utils import (autocast_context, environment_info, memory_peak, reset_memory,
                    restore_rng, rng_state, seed_everything, synchronize, write_json)

MAMBA_REQUIREMENTS = ("causal-conv1d==1.5.3.post1", "mamba-ssm==2.2.6.post3")


def install_mamba():
    """Called only in the Kaggle notebook, before importing mamba_ssm."""
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Use a Linux GPU Kaggle runtime for the Mamba notebook.")
    # Pin the release/API; retain Kaggle's preinstalled torch/CUDA ABI.
    # PyPI build setup resolves matching wheels where available, otherwise compiles.
    environment = os.environ.copy()
    environment["MAX_JOBS"] = "2"
    for requirement in MAMBA_REQUIREMENTS:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
                        "--no-deps", requirement], check=True, env=environment)
    require_mamba_kernels()


def require_mamba_kernels():
    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("Mamba measurements require Linux and a CUDA GPU.")
    try:
        import selective_scan_cuda
        import causal_conv1d_cuda
        import triton
        from mamba_ssm.modules.mamba_simple import (causal_conv1d_fn, causal_conv1d_update,
                                                   selective_state_update)
        if any(kernel is None for kernel in (causal_conv1d_fn, causal_conv1d_update, selective_state_update)):
            raise ImportError("Optimized training or recurrent decoding kernels are unavailable.")
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError("Mamba CUDA extensions do not load in this runtime. Re-run setup in a "
                           "fresh GPU session. If a matching wheel is unavailable, inspect nvcc and "
                           "the build log; no slow reference fallback is used.") from exc
    return {"selective_scan": selective_scan_cuda.__file__, "causal_conv1d": causal_conv1d_cuda.__file__,
            "triton_version": triton.__version__, "mamba_training_path": "fused mamba_inner_fn",
            "mamba_decoding_path": "Mamba.step with recurrent state cache"}


def probe_model(model, batch, device, precision="fp32", output_cap=64):
    """Forward/backward, finite outputs, and cached/uncached logits agreement."""
    saved_rng = rng_state()
    try:
        batch = move_batch(batch, device)
        model.train()
        with autocast_context(device, precision):
            logits = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   batch["labels"].reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError("Compatibility probe produced nonfinite loss.")
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Compatibility probe produced nonfinite gradients.")
        model.zero_grad(set_to_none=True)
        model.eval()
        steps = min(5, output_cap)
        prefix = torch.full((batch["input_ids"].size(0), steps), model.start_id,
                            device=device, dtype=torch.long)
        if steps > 1:
            prefix[:, 1:] = batch["input_ids"][:, :steps - 1]
        max_difference = 0.0
        with torch.inference_mode(), autocast_context(device, precision):
            memory = model.encode(batch["input_ids"], batch["attention_mask"])
            full, _ = model.decode_tokens(prefix, memory)
            cache, pieces = None, []
            for index in range(steps):
                output, cache = model.decode_tokens(prefix[:, index:index + 1], memory, cache, True)
                pieces.append(output)
            incremental = torch.cat(pieces, dim=1)
            tolerance = 2e-4 if precision == "fp32" else 5e-2
            torch.testing.assert_close(incremental.float(), full.float(), atol=tolerance, rtol=tolerance)
            max_difference = (incremental.float() - full.float()).abs().max().item()
            generated = generate(model, batch["input_ids"], batch["attention_mask"], min(8, output_cap))
        return {"loss": loss.item(), "finite_backward": True, "cache_logit_max_difference": max_difference,
                "generated_shape": list(generated.shape), "precision": precision}
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng(saved_rng)


def run_pilot(bundle, model_config, train_config, collator, device="cuda", steps=50, repo_root=None):
    if steps < 1:
        raise ValueError("Pilot steps must be positive.")
    if "mamba" in model_config.variant:
        kernels = require_mamba_kernels()
    else:
        kernels = {"attention": "PyTorch SDPA"}
    seed_everything(train_config.seed)
    model = build_model(model_config, bundle.tokenizer, bundle.config).to(device)
    batch = move_batch(collator(bundle.train.rows[:train_config.micro_batch_size]), device)
    probe = probe_model(model, batch, device, train_config.precision, bundle.config.max_output_length)
    optimizer = optimizer_for(model, train_config)
    scaler = torch.amp.GradScaler("cuda", enabled=train_config.precision == "fp16")
    losses, durations = [], []
    reset_memory(device)
    model.train()
    for step in range(steps + 2):
        synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, train_config.precision):
            logits = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                   batch["labels"].reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite pilot loss; try fp32.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite pilot gradients; try fp32.")
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        if step >= 2:
            durations.append(time.perf_counter() - start)
            losses.append(loss.item())
    mean_seconds = sum(durations) / len(durations)
    microbatches = (len(bundle.train) + train_config.micro_batch_size - 1) // train_config.micro_batch_size
    result = {"variant": model_config.variant, "probe": probe, "kernel_paths": kernels,
              "pilot_microsteps": steps, "mean_microstep_seconds": mean_seconds,
              "approximate_epoch_training_seconds": mean_seconds * microbatches,
              "estimate_note": "Includes an optimizer update each pilot microstep; excludes evaluation/checkpointing.",
              "first_loss": losses[0], "last_loss": losses[-1], "peak_memory": memory_peak(device),
              "environment": environment_info(repo_root)}
    del optimizer, model, logits, loss, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def restore_artifacts(source_root, output_root):
    """Copy previously saved Kaggle output folders into writable /kaggle/working."""
    import shutil

    source_root, output_root = Path(source_root), Path(output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    candidates = [name for name in ("checkpoints", "results", "data") if (source_root / name).is_dir()]
    for name in candidates:
        if (output_root / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_root / name}; restore before creating artifacts.")
    restored = []
    for name in candidates:
        source, destination = source_root / name, output_root / name
        shutil.copytree(source, destination)
        restored.append(str(destination))
    if not restored:
        raise ValueError("Source must contain checkpoints/, results/, or data/ from a prior notebook output.")
    return restored
