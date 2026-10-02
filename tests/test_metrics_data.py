import json

import pytest
import torch

from qa_assignment.config import DataConfig
from qa_assignment.data import (QACollator, article_split, download_squad, flatten_squad,
                                prepare_rows)
from qa_assignment.metrics import score_predictions, token_f1
from conftest import TinyTokenizer


def test_official_normalization_multiple_references_and_partial_overlap():
    metrics = score_predictions({"a": "The William Shakespeare!", "b": "Shakespeare"},
                                {"a": ["wrong", "William Shakespeare"], "b": ["William Shakespeare"]})
    assert metrics["em"] == 50
    assert metrics["f1"] == pytest.approx(100 * (1 + 2 / 3) / 2)
    assert token_f1("", "") == 0
    with pytest.raises(ValueError):
        score_predictions({"a": "answer"}, {"b": ["answer"]})


def test_article_split_has_no_article_leakage_and_is_seeded():
    rows = [{"id": str(i), "article_id": str(i // 3)} for i in range(30)]
    train, dev = article_split(rows, .2, 2026)
    assert not {row["article_id"] for row in train} & {row["article_id"] for row in dev}
    assert (train, dev) == article_split(rows, .2, 2026)
    assert len(train) + len(dev) == len(rows)


def test_length_filter_does_not_crop_to_gold_answer_and_keeps_validation_references():
    rows = [{"id": "long", "article_id": "a", "question": "who", "context": "one two three four five six",
             "answers": ["six"], "answer_starts": [24]},
            {"id": "short", "article_id": "b", "question": "who", "context": "one",
             "answers": ["one two three four"], "answer_starts": [0]}]
    config = DataConfig(max_input_length=6, max_output_length=3)
    train, stats = prepare_rows(rows, TinyTokenizer(), config, training=True)
    assert not train
    assert stats["removed_input_length"] == 1
    assert stats["removed_training_target_length"] == 1
    validation, _ = prepare_rows(rows, TinyTokenizer(), config, training=False)
    assert [row["id"] for row in validation] == ["short"]
    assert len(validation[0]["labels"]) > config.max_output_length


def test_collator_right_pads_and_masks_labels_without_truncation(tiny_bundle):
    collator = QACollator(tiny_bundle.tokenizer, tiny_bundle.config)
    batch = collator(tiny_bundle.train.rows[:2])
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].sum().item() == 6
    long_row = {**tiny_bundle.train.rows[0], "input_ids": list(range(9))}
    with pytest.raises(ValueError):
        collator([long_row])


def test_downloader_fetches_both_full_files_and_reuses_valid_downloads(tmp_path, monkeypatch):
    from qa_assignment import data

    raw = {"version": "1.1", "data": [{"title": "article", "paragraphs": [{"context": "word",
            "qas": [{"id": "q", "question": "what", "answers": [{"text": "word", "answer_start": 0}]}]}]}]}
    called = []

    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        def iter_content(self, size):
            yield json.dumps(raw).encode()

    def get(url, **kwargs):
        called.append(url)
        return Response()

    monkeypatch.setattr(data.requests, "get", get)
    paths = download_squad(tmp_path)
    assert set(paths) == {"train", "validation"}
    assert len(called) == 2
    download_squad(tmp_path)
    assert len(called) == 2
    assert flatten_squad(raw)[0]["answers"] == ["word"]

