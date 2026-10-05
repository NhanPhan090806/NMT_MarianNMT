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


@pytest.mark.parametrize("legacy_checkout", [False, True])
def test_scratch_notebook_order_capacity_stopping_and_diagnostic_gate(tmp_path, monkeypatch, legacy_checkout):
    from nmt_assignment import workflow
    if legacy_checkout:
        # The exact API missing from the a1465b4 Kaggle checkout in the error log.
        monkeypatch.delattr(workflow, "review_learning_diagnostics")
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
    assert namespace["DIAGNOSTIC_STEPS"] == 1000
    assert namespace["REQUIRE_DIAGNOSTIC_PASS"] is False
    assert namespace["TRAIN_STAGES"] == namespace["STAGE_ORDER"]
    assert tuple(namespace["STAGE_ORDER"]) == ("rnn", "lstm_attention", "transformer")
    assert [namespace["MODEL_CONFIGS"][s].d_model for s in namespace["STAGE_ORDER"]] == [128, 192, 256]
    namespace["diagnostics"] = {s: {"overfit_demonstrated": s != "lstm_attention", "target_chrf": 90,
        "final": {"chrf": 73.98 if s == "lstm_attention" else 100, "teacher_forced_loss": .102,
                  "token_accuracy": 98.34}} for s in namespace["STAGE_ORDER"]}
    exec(tagged(notebook, "train"), namespace)
    assert calls == ["rnn", "lstm_attention", "transformer"]
    # Recover the missing stage without touching completed RNN/Transformer runs.
    calls.clear()
    namespace["TRAIN_STAGES"] = ("lstm_attention",)
    exec(tagged(notebook, "train"), namespace)
    assert calls == ["lstm_attention"]
    # Strict gating is explicit and stops before ANY model is trained.
    calls.clear()
    namespace["TRAIN_STAGES"] = namespace["STAGE_ORDER"]
    namespace["REQUIRE_DIAGNOSTIC_PASS"] = True
    with pytest.raises(RuntimeError, match="Strict.*lstm_attention"):
        exec(tagged(notebook, "train"), namespace)
    assert calls == []
    namespace["REQUIRE_DIAGNOSTIC_PASS"] = False
    namespace["diagnostics"]["lstm_attention"]["final"]["chrf"] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        exec(tagged(notebook, "train"), namespace)
    assert calls == []


def test_lstm_only_selection_runs_only_its_diagnostics(tmp_path):
    from nmt_assignment.config import ModelConfig, TrainConfig
    notebook = nbformat.read(ROOT / "01_nmt_scratch_kaggle.ipynb", as_version=4)
    pilots, probes = [], []
    def pilot(bundle, config, train, device):
        pilots.append(config.stage)
        return {}
    def probe(bundle, config, root, device, **kwargs):
        probes.append(config.stage)
        return {"overfit_demonstrated": False}
    namespace = {"ModelConfig": ModelConfig, "TrainConfig": TrainConfig, "bundle": object(),
                 "OUTPUT_ROOT": tmp_path, "DEVICE": "cpu", "resource_pilot": pilot,
                 "learning_diagnostic": probe, "write_json": lambda *args: None,
                 "archive_translation_outputs": lambda root: root / "archive.zip"}
    exec(tagged(notebook, "configuration"), namespace)
    namespace["TRAIN_STAGES"] = ("lstm_attention",)
    exec(tagged(notebook, "diagnostics"), namespace)
    assert pilots == probes == ["lstm_attention"]
    assert set(namespace["diagnostics"]) == {"lstm_attention"}


def test_temporary_lstm_notebook_is_valid_and_matches_main_recipe():
    from dataclasses import asdict
    from nmt_assignment.config import DataConfig, ModelConfig, TrainConfig
    temporary = nbformat.read(ROOT / "temp_lstm_only_kaggle.ipynb", as_version=4)
    main = nbformat.read(ROOT / "01_nmt_scratch_kaggle.ipynb", as_version=4)
    nbformat.validate(temporary)
    for cell in temporary.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.outputs == [] and cell.execution_count is None
    controls = {"ModelConfig": ModelConfig, "TrainConfig": TrainConfig}
    exec(tagged(main, "configuration"), controls)
    settings = {}
    exec(tagged(temporary, "configuration"), settings)
    assert asdict(settings["MODEL_CONFIG"]) == asdict(controls["MODEL_CONFIGS"]["lstm_attention"])
    assert asdict(settings["TRAIN_CONFIG"]) == asdict(controls["TRAIN_CONFIGS"]["lstm_attention"])
    data_node = next(n for n in ast.parse(tagged(main, "data")).body if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "DATA_CONFIG")
    data_namespace = {"DataConfig": DataConfig}
    exec(compile(ast.Module(body=[data_node], type_ignores=[]), "main data settings", "exec"), data_namespace)
    assert asdict(settings["DATA_CONFIG"]) == asdict(data_namespace["DATA_CONFIG"])
    assert settings["EXPECTED_MODEL_PARAMETERS"] == 4672256
    from types import SimpleNamespace
    from nmt_assignment.models import build_model
    from nmt_assignment.workflow import model_parameter_counts
    class FullVocabulary:
        pad_token_id, eos_token_id, bos_token_id = 0, 1, 3
        def __len__(self):
            return settings["DATA_CONFIG"].vocab_size
    actual = build_model(settings["MODEL_CONFIG"], SimpleNamespace(
        tokenizer=FullVocabulary(), config=settings["DATA_CONFIG"], pretrained=False))
    assert model_parameter_counts(actual)["parameters"] == settings["EXPECTED_MODEL_PARAMETERS"]
    assert 'EXPECTED_DATA_FINGERPRINT' in tagged(temporary, "data")
    training = tagged(temporary, "train")
    assert "run_experiment(" in training and "continue" not in training
    assert "REQUIRE_DIAGNOSTIC_PASS" not in training and "overfit_demonstrated" not in training
    # Clone/settings are inherited, except for the isolated temporary output directory.
    assert tagged(temporary, "clone") == tagged(main, "clone").replace(
        'OUTPUT_ROOT = WORKSPACE / "nmt"', 'OUTPUT_ROOT = WORKSPACE / "nmt_lstm_only"')
    assert tagged(temporary, "setup") == tagged(main, "setup")


def test_pretrained_notebook_recipe_and_restoration_controls():
    notebook = nbformat.read(ROOT / "02_nmt_pretrained_kaggle.ipynb", as_version=4)
    text = tagged(notebook, "configuration")
    assert 'stage="marian_en_vi"' in text
    assert "epochs=3" in text and "learning_rate=2e-5" in text
    assert "early_stopping_patience=2" in text and "bundle.for_pretrained()" in text
    assert "RESTORE_FROM" in tagged(notebook, "setup")
    assert "PyTorch >=2.6" in text


def test_setup_requires_fresh_kernel_and_hub_compatible_with_kaggle():
    for name in ("01_nmt_scratch_kaggle.ipynb", "02_nmt_pretrained_kaggle.ipynb"):
        source = tagged(nbformat.read(ROOT / name, as_version=4), "setup")
        assert source.index("sys.modules") < source.index('"pip", "install"')
        assert 'version("transformers") != "5.17.0"' in source
    requirements = (ROOT / "requirements-kaggle.txt").read_text()
    assert "transformers==5.17.0" in requirements and "huggingface-hub>=1.23,<2" in requirements


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
