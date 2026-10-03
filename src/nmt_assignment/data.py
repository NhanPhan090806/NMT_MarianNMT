"""Pinned Parquet corpus, leakage checks, train-only vocabulary, common length scope."""

import hashlib
import html
import io
import json
import random
import os
import shutil
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import sentencepiece as spm
import torch

from qa_assignment.data import QADataset
from qa_assignment.utils import fingerprint, write_json
from .config import DataConfig


def load_pretrained_tokenizer(source, revision=None, cache_dir=None, local_files_only=False):
    """Keep the standard Marian tokenizer while avoiding native Unicode path IO on Windows."""
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    patterns = ["tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                "source.spm", "target.spm", "vocab.json", "target_vocab.json", "config.json"]
    path = Path(source)
    if not path.is_dir():
        path = Path(snapshot_download(str(source), revision=revision, cache_dir=cache_dir,
                                     allow_patterns=patterns, local_files_only=local_files_only))
    if os.name == "nt" and (path / "source.spm").is_file() and not str(path.resolve()).isascii():
        with tempfile.TemporaryDirectory(prefix="nmt_tokenizer_") as temporary:
            staging = Path(temporary).resolve()
            if not str(staging).isascii():
                raise RuntimeError("SentencePiece requires an ASCII temporary path on this Windows runtime.")
            for name in patterns:
                if (path / name).is_file():
                    shutil.copy2(path / name, staging / name)
            return AutoTokenizer.from_pretrained(staging, local_files_only=True)
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def normalize(text):
    if not isinstance(text, str):
        raise ValueError("Parallel corpus contains a non-string sentence.")
    return " ".join(unicodedata.normalize("NFC", html.unescape(text)).split())


def clean_split(rows, max_words):
    kept, seen, counts = [], set(), {"empty": 0, "long": 0, "duplicate_source": 0}
    for row in rows:
        pair = row.get("translation", row)
        source, target = normalize(pair["en"]), normalize(pair["vi"])
        if not source or not target:
            counts["empty"] += 1
        elif max(len(source.split()), len(target.split())) > max_words:
            counts["long"] += 1
        elif source in seen:
            counts["duplicate_source"] += 1
        else:
            seen.add(source)
            kept.append({"id": fingerprint(source), "source": source, "target": target})
    return kept, counts


def split_corpus(raw_splits, config):
    """A source sentence occurs in only one split, even with different translations."""
    if "train" not in raw_splits or "test" not in raw_splits:
        raise ValueError("The translation dataset must supply train and test splits.")
    test, test_counts = clean_split(raw_splits["test"], config.max_words)
    train, train_counts = clean_split(raw_splits["train"], config.max_words)
    test_sources = {r["source"] for r in test}
    overlap = sum(r["source"] in test_sources for r in train)
    train = [r for r in train if r["source"] not in test_sources]
    random.Random(config.split_seed).shuffle(train)
    if len(train) <= config.validation_examples:
        raise ValueError("Not enough training sources for the requested validation split.")
    validation, train = train[:config.validation_examples], train[config.validation_examples:]
    train = train[:config.max_train_examples] if config.max_train_examples is not None else train
    splits = {"train": train, "validation": validation, "test": test}
    if any(not rows for rows in splits.values()):
        raise ValueError("A cleaned corpus split is empty.")
    return splits, {"raw_counts": {k: len(v) for k, v in raw_splits.items()},
                    "train_filtering": train_counts, "test_filtering": test_counts,
                    "train_test_source_overlap_removed": overlap,
                    "split_counts_before_token_filter": {k: len(v) for k, v in splits.items()}}


class ScratchTokenizer:
    pad_token_id, eos_token_id, unk_token_id, bos_token_id = 0, 1, 2, 3

    def __init__(self, directory):
        self.directory = Path(directory)
        # Native SentencePiece file IO can fail on Vietnamese Windows paths.
        self.processor = spm.SentencePieceProcessor(model_proto=(self.directory / "joint.model").read_bytes())

    def __len__(self):
        return self.processor.get_piece_size()

    def encode(self, text):
        return self.processor.encode(text, out_type=int) + [self.eos_token_id]

    def batch_decode(self, sequences, skip_special_tokens=True):
        decoded = []
        for sequence in sequences:
            tokens = [int(t) for t in sequence]
            if self.eos_token_id in tokens:
                tokens = tokens[:tokens.index(self.eos_token_id)]
            decoded.append(self.processor.decode([t for t in tokens if t not in (0, 1, 3)]))
        return decoded


def train_tokenizer(rows, directory, vocab_size):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sentences = (r[key] for r in rows for key in ("source", "target"))
    serialized = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=sentences, model_writer=serialized,
        model_type="bpe", vocab_size=vocab_size, character_coverage=1.0,
        pad_id=0, eos_id=1, unk_id=2, bos_id=3, hard_vocab_limit=False,
        shuffle_input_sentence=False, num_threads=1, minloglevel=2)
    (directory / "joint.model").write_bytes(serialized.getvalue())
    processor = spm.SentencePieceProcessor(model_proto=serialized.getvalue())
    (directory / "joint.vocab").write_text("".join(
        f"{processor.id_to_piece(i)}\t{processor.get_score(i)}\n" for i in range(processor.get_piece_size())), encoding="utf-8")
    return ScratchTokenizer(directory)


def encode_splits(splits, scratch, pretrained, config):
    encoded, excluded = {}, {}
    for name, rows in splits.items():
        encoded[name], excluded[name] = [], 0
        # Batched HF tokenization avoids hundreds of thousands of Python calls.
        for start in range(0, len(rows), 256):
            chunk = rows[start:start + 256]
            src = pretrained([config.target_prefix + r["source"] for r in chunk],
                             truncation=False, verbose=False)["input_ids"]
            tgt = pretrained(text_target=[r["target"] for r in chunk],
                             truncation=False, verbose=False)["input_ids"]
            for row, source_ids, target_ids in zip(chunk, src, tgt):
                ids, labels = scratch.encode(row["source"]), scratch.encode(row["target"])
                if max(len(ids), len(source_ids)) > config.max_input_length or max(
                        len(labels), len(target_ids)) > config.max_output_length:
                    excluded[name] += 1
                    continue
                encoded[name].append({**row, "input_ids": ids, "labels": labels,
                                     "pretrained_input_ids": source_ids, "pretrained_labels": target_ids})
        if not encoded[name]:
            raise ValueError(f"No {name} examples fit both tokenizers. Increase caps or change the corpus.")
    return encoded, excluded


@dataclass
class DataBundle:
    train: QADataset
    development: QADataset
    test: QADataset
    tokenizer: object
    config: DataConfig
    manifest: dict
    directory: Path
    pretrained_tokenizer: object
    pretrained: bool = False

    def for_pretrained(self):
        def converted(dataset):
            return QADataset([{**r, "input_ids": r["pretrained_input_ids"],
                               "labels": r["pretrained_labels"]} for r in dataset.rows])
        return replace(self, train=converted(self.train), development=converted(self.development),
                       test=converted(self.test), tokenizer=self.pretrained_tokenizer, pretrained=True)


def prepare_data(data_root, config=DataConfig(), raw_splits=None, pretrained_tokenizer=None):
    """Download the entire pinned corpus once; preserve deterministic pilot/full scope."""
    directory = Path(data_root) / "prepared" / fingerprint(asdict(config))[:16]
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["contract"]["config"] != asdict(config):
            raise ValueError("Prepared configuration differs from the requested configuration.")
        encoded = {name: [json.loads(line) for line in (directory / f"{name}.jsonl").read_text(
                   encoding="utf-8").splitlines()] for name in ("train", "validation", "test")}
        scratch = ScratchTokenizer(directory / "scratch_tokenizer")
        pretrained_tokenizer = pretrained_tokenizer or load_pretrained_tokenizer(
            directory / "marian_tokenizer", local_files_only=True)
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.jsonl")}
        assets = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in ("scratch_tokenizer", "marian_tokenizer")
                  for p in sorted((directory / folder).glob("*")) if p.is_file()}
        if hashes != manifest["file_hashes"] or assets != manifest["tokenizer_hashes"]:
            raise ValueError("Prepared data/tokenizer files changed; regenerate in a new directory.")
    else:
        if raw_splits is None:
            from datasets import load_dataset
            raw_splits = load_dataset(config.dataset_name, revision=config.dataset_revision,
                                      cache_dir=str(Path(data_root) / "hf_cache"))
        splits, report = split_corpus(raw_splits, config)
        scratch = train_tokenizer(splits["train"], directory / "scratch_tokenizer", config.vocab_size)
        pretrained_tokenizer = pretrained_tokenizer or load_pretrained_tokenizer(
            config.pretrained_name, revision=config.pretrained_revision,
            cache_dir=str(Path(data_root).parent / "hf_cache"))
        supported = getattr(pretrained_tokenizer, "supported_language_codes", None)
        if supported is not None and config.target_prefix.strip() not in supported:
            raise ValueError("The selected Marian tokenizer does not support the target language prefix.")
        encoded, excluded = encode_splits(splits, scratch, pretrained_tokenizer, config)
        pretrained_tokenizer.save_pretrained(directory / "marian_tokenizer")
        for name, rows in encoded.items():
            (directory / f"{name}.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob("*.jsonl")}
        assets = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for folder in ("scratch_tokenizer", "marian_tokenizer")
                  for p in sorted((directory / folder).glob("*")) if p.is_file()}
        contract = {"format_version": 1, "config": asdict(config), "tokenizer_hashes": assets,
                    "file_hashes": hashes, "scratch_tokenizer_training": "selected training sources/targets only"}
        manifest = {"contract": contract, "data_fingerprint": fingerprint(contract), "audit": report,
                    "token_length_exclusions": excluded, "file_hashes": hashes, "tokenizer_hashes": assets,
                    "counts": {k: len(v) for k, v in encoded.items()},
                    "evaluation_scope": "cleaned, source-disjoint, word- and dual-tokenizer-length-filtered IWSLT test"}
        write_json(manifest_path, manifest)
    return DataBundle(QADataset(encoded["train"]), QADataset(encoded["validation"]),
                      QADataset(encoded["test"]), scratch, config, manifest, directory, pretrained_tokenizer)


class TranslationCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, rows):
        # Dynamic padding keeps recurrent and attention compute proportional to this batch.
        inputs = torch.full((len(rows), max(len(r["input_ids"]) for r in rows)), self.pad_id, dtype=torch.long)
        mask = torch.zeros_like(inputs)
        labels = torch.full((len(rows), max(len(r["labels"]) for r in rows)), -100, dtype=torch.long)
        for i, row in enumerate(rows):
            inputs[i, :len(row["input_ids"])] = torch.tensor(row["input_ids"])
            mask[i, :len(row["input_ids"])] = 1
            labels[i, :len(row["labels"])] = torch.tensor(row["labels"])
        return {"input_ids": inputs, "attention_mask": mask, "labels": labels,
                "ids": [r["id"] for r in rows], "sources": [r["source"] for r in rows],
                "references": [r["target"] for r in rows]}
