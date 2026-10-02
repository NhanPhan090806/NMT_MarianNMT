"""Download all official SQuAD 1.1 data and prepare a shared, auditable subset."""

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .config import DataConfig
from .utils import fingerprint, write_json

SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "validation": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


def validate_raw(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != "1.1" or not raw.get("data"):
        raise ValueError(f"Not a SQuAD 1.1 dataset: {path}")
    return raw


def download_squad(data_root):
    """Download BOTH complete official files, regardless of later training limits."""
    raw_root = Path(data_root) / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, url in SQUAD_URLS.items():
        path = raw_root / url.rsplit("/", 1)[1]
        if not path.exists():
            temp = path.with_suffix(".json.part")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temp.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            validate_raw(temp)
            os.replace(temp, path)
        validate_raw(path)
        paths[split] = path
    return paths


def flatten_squad(raw):
    rows = []
    for article_index, article in enumerate(raw["data"]):
        article_id = f"{article_index}:{article['title']}"
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa.get("answers"):
                    raise ValueError("An unanswerable question was found in SQuAD 1.1.")
                rows.append({"id": qa["id"], "article_id": article_id,
                             "question": qa["question"], "context": paragraph["context"],
                             "answers": [answer["text"] for answer in qa["answers"]],
                             "answer_starts": [answer["answer_start"] for answer in qa["answers"]]})
    return rows


def article_split(rows, fraction, seed):
    articles = sorted({row["article_id"] for row in rows})
    if len(articles) < 2:
        raise ValueError("At least two training articles are needed for a grouped split.")
    random.Random(seed).shuffle(articles)
    count = min(len(articles) - 1, max(1, math.ceil(len(articles) * fraction)))
    dev_articles = set(articles[:count])
    return ([row for row in rows if row["article_id"] not in dev_articles],
            [row for row in rows if row["article_id"] in dev_articles])


def input_text(row):
    return f"question: {row['question']} context: {row['context']}"


def prepare_rows(rows, tokenizer, config, training=False):
    prepared, input_lengths, target_lengths = [], [], []
    removed_input = removed_target = invalid_spans = 0
    for offset in tqdm(range(0, len(rows), 256), desc="Tokenizing", leave=False):
        batch = rows[offset:offset + 256]
        inputs = tokenizer([input_text(row) for row in batch], truncation=False,
                           padding=False, verbose=False)["input_ids"]
        targets = tokenizer([row["answers"][0] for row in batch], truncation=False,
                            padding=False, verbose=False)["input_ids"]
        for row, ids, target in zip(batch, inputs, targets):
            input_lengths.append(len(ids))
            target_lengths.append(len(target))
            if len(ids) > config.max_input_length:
                removed_input += 1
                continue
            if training and len(target) > config.max_output_length:
                removed_target += 1
                continue
            # Diagnostics only: gold offsets never choose the visible input.
            valid = any(row["context"][start:start + len(answer)] == answer
                        for start, answer in zip(row["answer_starts"], row["answers"]))
            invalid_spans += int(not valid)
            prepared.append({"id": row["id"], "article_id": row["article_id"],
                             "input_ids": ids, "labels": target, "answers": row["answers"]})
    stats = {"original": len(rows), "retained_before_train_limit": len(prepared),
             "removed_input_length": removed_input, "removed_training_target_length": removed_target,
             "invalid_span_examples_retained": invalid_spans,
             "input_length_quantiles": np.quantile(input_lengths, [0, .5, .9, .99, 1]).tolist() if rows else [],
             "target_length_quantiles": np.quantile(target_lengths, [0, .5, .9, .99, 1]).tolist() if rows else [],
             "references_exceeding_output_cap": sum(length > config.max_output_length for length in target_lengths)}
    return prepared, stats


class QADataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def first(self, count):
        return QADataset(self.rows[:count])


@dataclass
class DataBundle:
    train: QADataset
    development: QADataset
    validation: QADataset
    tokenizer: object
    config: DataConfig
    manifest: dict
    directory: Path


def prepare_data(data_root, config=DataConfig()):
    from transformers import AutoTokenizer

    paths = download_squad(data_root)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name,
                                               revision=config.tokenizer_revision, use_fast=True)
    tokenizer.padding_side = "right"
    tokenizer_signature = fingerprint({"backend": tokenizer.backend_tokenizer.to_str(),
                                        "special_tokens": tokenizer.special_tokens_map,
                                        "pad": tokenizer.pad_token_id, "eos": tokenizer.eos_token_id})
    contract = {"schema_version": 1, "config": asdict(config), "tokenizer": tokenizer_signature,
                "raw_sha256": {split: hashlib.sha256(path.read_bytes()).hexdigest()
                               for split, path in paths.items()}}
    directory = Path(data_root) / "prepared" / fingerprint(contract)[:16]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = {split: [json.loads(line) for line in (directory / f"{split}.jsonl").read_text(
            encoding="utf-8").splitlines()] for split in ("train", "development", "validation")}
        if manifest["contract"] != contract or fingerprint(rows) != manifest["data_fingerprint"]:
            raise ValueError("Prepared cache is inconsistent; use a new data root or restore an intact cache.")
    else:
        train_raw = flatten_squad(validate_raw(paths["train"]))
        train_rows, dev_rows = article_split(train_raw, config.development_fraction, config.split_seed)
        validation_rows = flatten_squad(validate_raw(paths["validation"]))
        rows, statistics = {}, {}
        for split, original in (("train", train_rows), ("development", dev_rows),
                                ("validation", validation_rows)):
            rows[split], statistics[split] = prepare_rows(original, tokenizer, config, split == "train")
        if config.max_train_examples is not None:
            random.Random(config.split_seed).shuffle(rows["train"])
            rows["train"] = rows["train"][:config.max_train_examples]
        if any(not split_rows for split_rows in rows.values()):
            raise ValueError("A prepared split is empty; increase the length cap or dataset budget.")
        for split in rows:
            statistics[split]["retained"] = len(rows[split])
        manifest = {"contract": contract, "data_fingerprint": fingerprint(rows), "statistics": statistics,
                    "source_urls": SQUAD_URLS, "evaluation_scope": "length-filtered official validation",
                    "ids": {split: [row["id"] for row in values] for split, values in rows.items()}}
        directory.mkdir(parents=True, exist_ok=True)
        for split, split_rows in rows.items():
            with (directory / f"{split}.jsonl").open("w", encoding="utf-8") as output:
                for row in split_rows:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        tokenizer.save_pretrained(directory / "tokenizer")
        # Write last: an interrupted preparation must not appear complete.
        write_json(manifest_path, manifest)
    return DataBundle(*(QADataset(rows[split]) for split in ("train", "development", "validation")),
                      tokenizer, config, manifest, directory)


class QACollator:
    def __init__(self, tokenizer, config):
        self.pad = tokenizer.pad_token_id
        self.input_cap = config.max_input_length
        self.output_cap = config.max_output_length

    def __call__(self, rows):
        size = len(rows)
        input_ids = torch.full((size, self.input_cap), self.pad, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        # Keep complete development/final targets, even if the generation cap is shorter.
        target_length = max(len(row["labels"]) for row in rows)
        labels = torch.full((size, target_length), -100, dtype=torch.long)
        for index, row in enumerate(rows):
            if len(row["input_ids"]) > self.input_cap:
                raise ValueError("Input exceeds the frozen cap; do not silently truncate.")
            length = len(row["input_ids"])
            input_ids[index, :length] = torch.tensor(row["input_ids"])
            attention_mask[index, :length] = 1
            labels[index, :len(row["labels"])] = torch.tensor(row["labels"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "ids": [row["id"] for row in rows], "answers": [row["answers"] for row in rows]}


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}

