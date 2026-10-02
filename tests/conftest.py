import pytest
import torch

from qa_assignment.config import DataConfig, ModelConfig
from qa_assignment.data import DataBundle, QACollator, QADataset
from qa_assignment.models import ScratchQA


@pytest.fixture(autouse=True)
def few_threads():
    torch.set_num_threads(1)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return 32

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[2 + len(word) % 29 for word in text.split()] + [1] for text in texts]}

    def batch_decode(self, sequences, skip_special_tokens=True):
        return [" ".join(f"w{token}" for token in sequence if token not in (0, 1)) for sequence in sequences]


@pytest.fixture
def tiny_bundle(tmp_path):
    tokenizer = TinyTokenizer()
    config = DataConfig(max_input_length=8, max_output_length=6)
    rows = [{"id": f"q{index}", "article_id": f"article{index}",
             "input_ids": [2 + index % 3, 3, 1], "labels": [4 + index % 3, 1],
             "answers": [f"w{4 + index % 3}"]} for index in range(7)]
    bundle = DataBundle(QADataset(rows), QADataset(rows[:2]), QADataset(rows[2:4]),
                        tokenizer, config, {"data_fingerprint": "tiny-data", "contract": {"tiny": True},
                                            "evaluation_scope": "synthetic test"}, tmp_path)
    return bundle


@pytest.fixture
def tiny_model():
    config = ModelConfig(d_model=16, encoder_layers=1, decoder_layers=1, heads=2,
                         feedforward_dim=32, dropout=0.1)
    return ScratchQA(config, 32, 0, 1, 8, 6)

