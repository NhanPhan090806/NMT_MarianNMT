"""Shared data contract and ordered historical model families."""

from dataclasses import dataclass

from qa_assignment.config import TrainConfig as BaseTrainConfig

STAGES = ("rnn", "lstm_attention", "transformer", "marian_en_vi")
MILESTONES = {
    "rnn": {"order": 1, "year": 1986, "label": "RNN: recurrent baseline", "pretrained": False},
    "lstm_attention": {"order": 2, "year": 1997, "label": "LSTM (1997) + attention (2015)", "pretrained": False},
    "transformer": {"order": 3, "year": 2017, "label": "Transformer: self/cross attention", "pretrained": False},
    "marian_en_vi": {"order": 4, "year": 2020, "label": "Marian: pretrained EN→VI reference", "pretrained": True},
}


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "Angelectronic/IWSLT15_English_Vietnamese"
    dataset_revision: str = "647d179736b0a4f2b62f860ffc962b067505311b"
    pretrained_name: str = "Helsinki-NLP/opus-mt-en-vi"
    pretrained_revision: str = "989c9fb9ec63987901022baf0182dcec3e149be6"
    validation_examples: int = 5000
    max_train_examples: int | None = 25000  # None uses the whole eligible training pool
    split_seed: int = 2026
    vocab_size: int = 8000
    max_input_length: int = 96
    max_output_length: int = 96
    max_words: int = 60
    target_prefix: str = ">>vie<< "

    def __post_init__(self):
        if min(self.validation_examples, self.max_words, self.max_input_length,
               self.max_output_length) < 2 or self.vocab_size < 32:
            raise ValueError("Invalid corpus, vocabulary, or sequence limits.")
        if self.max_train_examples is not None and self.max_train_examples < 1:
            raise ValueError("max_train_examples must be positive or None.")


@dataclass(frozen=True)
class ModelConfig:
    stage: str = "transformer"
    d_model: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    heads: int = 4
    feedforward_dim: int = 512
    dropout: float = 0.1
    tie_embeddings: bool = False

    @property
    def variant(self):
        # The existing cached SDPA encoder/decoder is task-independent.
        return "attn_attn" if self.stage == "transformer" else self.stage

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"Unknown translation stage: {self.stage}")
        if min(self.d_model, self.encoder_layers, self.decoder_layers, self.heads,
               self.feedforward_dim) < 1 or self.d_model % self.heads:
            raise ValueError("Invalid model dimensions.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")


@dataclass(frozen=True)
class TrainConfig(BaseTrainConfig):
    epochs: int = 8
    micro_batch_size: int = 16
    accumulation_steps: int = 2
    eval_batch_size: int = 16
    learning_rate: float = 1e-3
    warmup_fraction: float = 0.02
    development_limit: int | None = 512  # fixed subset, identical across every stage
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.2  # chrF points (0..100), not QA F1
    early_stopping_min_epochs: int = 3

    def __post_init__(self):
        super().__post_init__()
        if self.early_stopping_min_epochs < 0:
            raise ValueError("Minimum epochs cannot be negative.")
