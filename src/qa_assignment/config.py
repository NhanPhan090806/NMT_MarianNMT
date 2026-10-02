"""Small, serializable configuration objects; notebooks only edit these."""

from dataclasses import asdict, dataclass
import math
from typing import Literal

VARIANTS = ("attn_attn", "mamba_attn", "attn_mamba", "mamba_mamba")
BASELINE_VARIANTS = ("rnn_rnn", "lstm_lstm", "attn_attn", "t5_small")


@dataclass(frozen=True)
class DataConfig:
    tokenizer_name: str = "google-t5/t5-small"
    tokenizer_revision: str = "main"
    max_input_length: int = 512
    max_output_length: int = 64
    development_fraction: float = 0.1
    split_seed: int = 2026
    max_train_examples: int | None = None

    def __post_init__(self):
        if min(self.max_input_length, self.max_output_length) < 2:
            raise ValueError("Sequence limits must be at least 2.")
        if not 0 < self.development_fraction < 1:
            raise ValueError("development_fraction must be between 0 and 1.")
        if self.max_train_examples is not None and self.max_train_examples < 1:
            raise ValueError("max_train_examples must be positive or None.")


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "attn_attn"
    d_model: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    tie_embeddings: bool = True
    t5_name: str = "google-t5/t5-small"
    t5_revision: str = "main"

    def __post_init__(self):
        if self.variant not in (*VARIANTS, *BASELINE_VARIANTS):
            raise ValueError(f"Unknown variant: {self.variant}")
        if min(self.d_model, self.heads, self.encoder_layers, self.decoder_layers,
               self.feedforward_dim, self.d_state, self.expand) < 1:
            raise ValueError("Model dimensions and layer counts must be positive.")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if self.d_conv not in (2, 3, 4):
            raise ValueError("Use a convolution width supported by causal-conv1d: 2, 3, or 4.")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    micro_batch_size: int = 4
    accumulation_steps: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    seed: int = 42
    sample_order_seed: int = 2026
    save_every_steps: int = 250
    eval_every_steps: int = 0  # 0 = internal development evaluation each epoch
    keep_step_checkpoints: int = 2
    development_limit: int | None = None  # None evaluates the whole eligible dev split
    log_every_steps: int = 25
    early_stopping_patience: int = 0  # 0 disables; counts development evaluations
    early_stopping_min_delta: float = 0.0  # absolute F1 points on the 0..100 scale
    early_stopping_min_steps: int = 0  # do not count failures before this optimizer step

    def __post_init__(self):
        if min(self.epochs, self.micro_batch_size, self.accumulation_steps,
               self.eval_batch_size, self.log_every_steps) < 1:
            raise ValueError("Epochs, batches, accumulation, and logging interval must be positive.")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("Unknown precision.")
        if not 0 <= self.warmup_fraction < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate or warmup fraction.")
        if min(self.save_every_steps, self.eval_every_steps, self.keep_step_checkpoints) < 0:
            raise ValueError("Checkpoint/evaluation intervals cannot be negative.")
        if self.development_limit is not None and self.development_limit < 1:
            raise ValueError("development_limit must be positive or None.")
        if min(self.early_stopping_patience, self.early_stopping_min_steps) < 0:
            raise ValueError("Early-stopping patience and minimum steps cannot be negative.")
        if not math.isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0:
            raise ValueError("Early-stopping minimum F1 improvement must be finite and nonnegative.")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkConfig:
    examples: int = 200
    repeats: int = 3
    warmup_runs: int = 2
    throughput_batch_size: int = 4
    fixed_output_tokens: int = 32

    def __post_init__(self):
        if min(self.examples, self.repeats, self.throughput_batch_size,
               self.fixed_output_tokens) < 1 or self.warmup_runs < 0:
            raise ValueError("Invalid benchmark configuration.")
