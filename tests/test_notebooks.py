"""Validate and execute the actual new notebook orchestration without network/GPU."""

import ast
from dataclasses import replace
from pathlib import Path

import nbformat
import pytest


ROOT = Path(__file__).resolve().parents[1]


def tagged(notebook, name):
    return next(c.source for c in notebook.cells if name in c.metadata.get("tags", []))


def test_training_and_local_testing_notebooks_are_valid():
    expected = {"01_nmt_scratch_kaggle.ipynb", "02_nmt_pretrained_kaggle.ipynb",
                "03_t5_question_answering_local.ipynb"}
    paths = sorted(ROOT.glob("0*_*.ipynb"))
    assert {p.name for p in paths} == expected
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                if not path.name.startswith("03"):
                    assert cell.outputs == [] and cell.execution_count is None
        text = "\n".join(c.source for c in notebook.cells)
        if path.name.startswith("03"):
            assert "T5Answerer.from_export" in text and "models/t5_small" in text
            assert "/kaggle/" not in text
        else:
            assert '"git", "clone"' in text and 'resume="auto"' in text
            assert "install_mamba" not in text and "require_mamba_kernels" not in text
            assert "DataConfig(max_train_examples=25000" in text


def test_scratch_notebook_order_capacity_stopping_and_diagnostic_gate(tmp_path, monkeypatch):
    from nmt_assignment import workflow
    from nmt_assignment.config import ModelConfig, TrainConfig
    notebook = nbformat.read(ROOT / "01_nmt_scratch_kaggle.ipynb", as_version=4)
    calls = []
    def completed(bundle, model_config, train_config, root, **kwargs):
        calls.append(model_config.stage)
        assert train_config.early_stopping_patience == 3 and train_config.epochs == 8
        assert kwargs["resume"] == "auto"
        return {"stage": model_config.stage, "test": {"bleu": 1, "chrf": 2}}
    namespace = {"ModelConfig": ModelConfig, "TrainConfig": TrainConfig, "replace": replace,
                 "bundle": object(), "OUTPUT_ROOT": tmp_path, "DEVICE": "cpu", "REPO_ROOT": ROOT,
                 "run_experiment": completed, "archive_translation_outputs": lambda root: root / "archive.zip"}
    exec(tagged(notebook, "configuration"), namespace)
    assert tuple(namespace["STAGE_ORDER"]) == ("rnn", "lstm_attention", "transformer")
    assert [namespace["MODEL_CONFIGS"][s].d_model for s in namespace["STAGE_ORDER"]] == [128, 192, 256]
    namespace["diagnostics"] = {s: {"overfit_demonstrated": s != "rnn"} for s in namespace["STAGE_ORDER"]}
    exec(tagged(notebook, "train"), namespace)
    assert calls == ["lstm_attention", "transformer"]


def test_pretrained_notebook_recipe_and_restoration_controls():
    notebook = nbformat.read(ROOT / "02_nmt_pretrained_kaggle.ipynb", as_version=4)
    text = tagged(notebook, "configuration")
    assert 'stage="marian_en_vi"' in text
    assert "epochs=3" in text and "learning_rate=2e-5" in text
    assert "early_stopping_patience=2" in text and "bundle.for_pretrained()" in text
    assert "RESTORE_FROM" in tagged(notebook, "setup")
    assert "PyTorch >=2.6" in text


def test_generator_preserves_user_controls_and_never_edits_qa(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("nmt_generator", ROOT / "scripts" / "build_nmt_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    qa = tmp_path / "03_t5_question_answering_local.ipynb"
    qa.write_bytes(b"user-owned notebook bytes")
    module.main()
    path = tmp_path / "01_nmt_scratch_kaggle.ipynb"
    notebook = nbformat.read(path, as_version=4)
    config = next(c for c in notebook.cells if "configuration" in c.metadata.get("tags", []))
    config.source = config.source.replace("epochs=8", "epochs=6")
    notebook.cells.append(nbformat.v4.new_code_cell('print("custom translation")'))
    nbformat.write(notebook, path)
    module.main()
    refreshed = nbformat.read(path, as_version=4)
    assert "epochs=6" in tagged(refreshed, "configuration")
    assert refreshed.cells[-1].source == 'print("custom translation")'
    assert qa.read_bytes() == b"user-owned notebook bytes"
