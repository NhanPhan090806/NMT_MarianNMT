"""Behavioral coverage for the offline run audit's sampling and integrity checks."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("nmt_audit", Path(__file__).resolve().parents[1] / "scripts" / "analyze_nmt_runs.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_audit_samples_proportional_lengths_reproducibly_without_duplicates():
    rows = [{"id": str(i), "source": "word " * (5 if i < 20 else 15 if i < 70 else 30)} for i in range(100)]
    selected = audit.representative_rows(rows, 10)
    assert selected == audit.representative_rows(rows, 10)
    assert len({r["id"] for r in selected}) == 10
    assert [sum(audit.length_group(r) == name for r in selected)
            for name in ("1-10 words", "11-20 words", "21-60 words")] == [2, 5, 3]
    assert audit.has_repeated_fourgram("a b c d a b c d")
    assert not audit.has_repeated_fourgram("a b c d e f g h")


def test_audit_rejects_modified_data_and_cross_split_sources(tmp_path):
    manifest = {"file_hashes": {}}
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps({"id": name, "source": name}) + "\n", encoding="utf-8")
        manifest["file_hashes"][path.name] = audit.file_hash(path)
    assert len(audit.verify_corpus(tmp_path, manifest)) == 3
    test = tmp_path / "test.jsonl"
    test.write_text('{"id":"another","source":"train"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit.verify_corpus(tmp_path, manifest)
    manifest["file_hashes"][test.name] = audit.file_hash(test)
    with pytest.raises(ValueError, match="Source overlap"):
        audit.verify_corpus(tmp_path, manifest)
