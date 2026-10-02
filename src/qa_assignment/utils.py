"""Artifacts, randomness, and environment provenance."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def memory_peak(device):
    if torch.device(device).type != "cuda":
        return {"allocated_mib": 0.0, "reserved_mib": 0.0}
    return {"allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}


def reset_memory(device):
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def autocast_context(device, precision):
    device = torch.device(device)
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed precision is supported only on CUDA in this workflow.")
    if precision == "bf16" and torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("BF16 requires an Ampere-or-newer GPU; use fp32 or fp16.")
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def environment_info(repo_root=None):
    packages = {}
    for name in ("torch", "transformers", "sentencepiece", "numpy", "mamba-ssm",
                 "causal-conv1d", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cuda_runtime": torch.version.cuda, "gpu": None, "git_commit": None}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = {"name": p.name, "memory_mib": p.total_memory / 2**20,
                       "capability": [p.major, p.minor], "visible_count": torch.cuda.device_count()}
    if repo_root is not None:
        result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        info["git_commit"] = result.stdout.strip() if result.returncode == 0 else None
    return info


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    excluded = {id(p) for p in model.embedding.parameters()} if hasattr(model, "embedding") else set()
    if hasattr(model, "output_projection"):
        excluded.update(id(p) for p in model.output_projection.parameters())
    if hasattr(model, "hf_model"):
        excluded.update(id(p) for p in model.hf_model.shared.parameters())
        excluded.update(id(p) for p in model.hf_model.lm_head.parameters())
    return {"parameters": total,
            "parameters_without_embeddings_output": sum(p.numel() for p in model.parameters()
                                                         if p.requires_grad and id(p) not in excluded)}
