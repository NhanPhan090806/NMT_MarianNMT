import ast
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import nbformat


def test_two_kaggle_notebooks_are_valid_and_have_repo_variables():
    paths = sorted(Path(__file__).resolve().parents[1].glob("*_kaggle.ipynb"))
    assert len(paths) == 2
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.outputs == []
                assert cell.execution_count is None
        combined = "\n".join(cell.source for cell in notebook.cells)
        assert 'REPO_URL = ' in combined
        assert '"git", "clone"' in combined
        assert 'resume="auto"' in combined


def test_attention_mamba_notebook_passes_stopping_policy_to_every_run(tmp_path, monkeypatch):
    from qa_assignment import workflow
    from qa_assignment.config import VARIANTS

    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "01_attention_mamba_kaggle.ipynb", as_version=4)
    namespace = {"bundle": object(), "OUTPUT_ROOT": tmp_path / "outputs", "DEVICE": "cuda:0",
                 "REPO_ROOT": root}
    exec(notebook.cells[9].source, namespace)
    calls = []
    def completed_run(bundle, model_config, train_config, output_root, **kwargs):
        assert train_config.epochs == 20
        assert train_config.eval_every_steps == 1000
        assert train_config.early_stopping_patience == 5
        assert train_config.early_stopping_min_delta == 0.1
        assert train_config.early_stopping_min_steps == 5000
        assert kwargs["resume"] == "auto"
        calls.append((model_config.variant, train_config.seed))
        return {"variant": model_config.variant, "early_stopping": {"stopped": True}}
    monkeypatch.setattr(workflow, "run_experiment", completed_run)
    # Exercise the actual notebook loop; a stopped run must let subsequent runs proceed.
    exec(notebook.cells[15].source, namespace)
    assert calls == [(variant, seed) for variant in VARIANTS for seed in namespace["SEEDS"]]
    assert len(namespace["summaries"]) == len(calls)


def test_setup_exposes_sources_in_an_already_running_interpreter(tmp_path):
    """Reproduce the logged failure without installing anything into the real venv."""
    root = Path(__file__).resolve().parents[1]
    for name in ("01_attention_mamba_kaggle.ipynb", "02_t5_transfer_learning_kaggle.ipynb"):
        notebook = nbformat.read(root / name, as_version=4)
        tree = ast.parse(notebook.cells[3].source)
        # Exercise pip and ALL helper imports, stopping before GPU selection.
        prefix = []
        for node in tree.body:
            if isinstance(node, ast.If) and "torch.cuda.is_available" in ast.unparse(node.test):
                break
            prefix.append(node)
        setup_source = ast.unparse(ast.Module(body=prefix, type_ignores=[]))
        script = '''
import importlib.util
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(sys.argv[1])
mock_site = Path(sys.argv[2])
mock_site.mkdir(exist_ok=True)
calls = []
def fake_install(command, **kwargs):
    calls.append(command)
    # Simulate pip publishing an editable-install .pth AFTER interpreter startup.
    (mock_site / 'qa_assignment.pth').write_text(str(REPO_ROOT / 'src'), encoding='utf-8')
subprocess.run = fake_install
assert importlib.util.find_spec('qa_assignment') is None
exec(sys.argv[3])
assert len(calls) == 2
assert Path(qa_assignment.__file__).resolve() == REPO_ROOT / 'src' / 'qa_assignment' / '__init__.py'
print('Live-kernel setup passed')
'''
        result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", script,
                                 str(root), str(tmp_path), setup_source],
                                capture_output=True, text=True, encoding="utf-8", timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Live-kernel setup passed" in result.stdout


def test_baseline_notebook_runs_all_four_variants_without_mamba(tiny_bundle, tmp_path, monkeypatch):
    """Execute the actual notebook orchestration on tiny data, without network calls."""
    import json
    import matplotlib
    from transformers import T5Config, T5ForConditionalGeneration
    from qa_assignment import runtime
    from qa_assignment.config import BASELINE_VARIANTS, BenchmarkConfig
    from qa_assignment.data import QACollator

    matplotlib.use("Agg")
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "02_t5_transfer_learning_kaggle.ipynb", as_version=4)
    combined = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "install_mamba" not in combined and "require_mamba_kernels" not in combined
    # A Mamba dependency on any notebook-2 execution path must fail this test.
    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline notebook must not load or install Mamba kernels")
    monkeypatch.setattr(runtime, "require_mamba_kernels", forbidden)
    monkeypatch.setattr(runtime, "install_mamba", forbidden)
    hf_config = T5Config(vocab_size=32, d_model=16, d_ff=32, d_kv=8, num_layers=1,
                         num_decoder_layers=1, num_heads=2, dropout_rate=0,
                         pad_token_id=0, eos_token_id=1, decoder_start_token_id=0)
    monkeypatch.setattr(T5ForConditionalGeneration, "from_pretrained",
                        lambda *args, **kwargs: T5ForConditionalGeneration(hf_config))
    def save_tokenizer(directory):
        (Path(directory) / "tokenizer_config.json").write_text('{}', encoding="utf-8")
    monkeypatch.setattr(tiny_bundle.tokenizer, "save_pretrained", save_tokenizer, raising=False)
    namespace = {"bundle": tiny_bundle, "collator": QACollator(tiny_bundle.tokenizer, tiny_bundle.config),
                 "OUTPUT_ROOT": tmp_path / "qa_baselines", "DEVICE": "cpu", "REPO_ROOT": None,
                 "display": lambda *args: None}
    exec(notebook.cells[7].source, namespace)
    assert tuple(namespace["MODEL_VARIANTS"]) == BASELINE_VARIANTS
    assert namespace["SCRATCH_TRAIN_CONFIG"].epochs == 20
    assert namespace["T5_TRAIN_CONFIG"].epochs == 5
    assert all(config.early_stopping_patience == 5 for config in namespace["TRAIN_CONFIGS"].values())
    assert all(config.eval_every_steps == 1000 for config in namespace["TRAIN_CONFIGS"].values())
    for variant in BASELINE_VARIANTS:
        namespace["MODEL_CONFIGS"][variant] = replace(namespace["MODEL_CONFIGS"][variant],
            d_model=16, encoder_layers=1, decoder_layers=1, heads=2, feedforward_dim=32, dropout=0)
        namespace["TRAIN_CONFIGS"][variant] = replace(namespace["TRAIN_CONFIGS"][variant],
            epochs=1, micro_batch_size=2, accumulation_steps=2, eval_batch_size=2,
            save_every_steps=1, log_every_steps=1)
    namespace["BENCHMARK_CONFIG"] = BenchmarkConfig(examples=2, repeats=1, warmup_runs=1,
        throughput_batch_size=2, fixed_output_tokens=3)
    # Only shorten the test pilot; execute the notebook's actual model loops.
    original_pilot = runtime.run_pilot
    monkeypatch.setattr(runtime, "run_pilot", lambda *args, **kwargs: original_pilot(
        *args, **{**kwargs, "steps": 3}))
    for index in (9, 11, 13, 15, 16):
        exec(notebook.cells[index].source, namespace)
    assert [summary["variant"] for summary in namespace["summaries"]] == list(BASELINE_VARIANTS)
    assert namespace["table"].variant.tolist() == list(BASELINE_VARIANTS)
    assert namespace["table"].pretrained.tolist() == [False, False, False, True]
    for variant in BASELINE_VARIANTS:
        checkpoint = namespace["OUTPUT_ROOT"] / "checkpoints" / variant / "seed_42"
        assert (checkpoint / "last.pt").is_file() and (checkpoint / "best.pt").is_file()
        results = namespace["OUTPUT_ROOT"] / "results" / variant / "seed_42"
        summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
        assert summary["validation"]["examples"] == 2
        assert summary["benchmark"]["fixed_workload"]["measurements"][0]["mean_generated_tokens"] == 3
    assert (namespace["OUTPUT_ROOT"] / "checkpoints" / "t5_small" / "seed_42" /
            "hf_export" / "config.json").is_file()
    assert (namespace["OUTPUT_ROOT"] / "results" / "seq2seq_baselines_comparison.png").is_file()
    assert namespace["archive_path"].is_file()
