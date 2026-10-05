"""Build a temporary LSTM-only interface from the supplied run's frozen recipe."""

import ast
import json
import sys
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nmt_assignment.config import ModelConfig, TrainConfig
from build_nmt_notebooks import notebook_helper_import


def code(source, tag):
    return nbf.v4.new_code_cell(dedent(source).strip(), metadata={"tags": [tag]})


def md(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def tagged(notebook, tag):
    return next(c.source for c in notebook.cells if tag in c.metadata.get("tags", []))


def main():
    reference = ROOT / "kaggle_runs" / "rnn_lstm_trsf"
    contract = json.loads((reference / "checkpoints" / "rnn" / "seed_42" / "run_config.json").read_text(encoding="utf-8"))
    pilot = json.loads((reference / "results" / "pilots" / "lstm_attention.json").read_text(encoding="utf-8"))
    original = nbf.read(ROOT / "01_nmt_scratch_kaggle.ipynb", as_version=4)
    controls = {"ModelConfig": ModelConfig, "TrainConfig": TrainConfig}
    exec(compile(tagged(original, "configuration"), "notebook configuration", "exec"), controls)
    model_settings = asdict(controls["MODEL_CONFIGS"]["lstm_attention"])
    if asdict(controls["TRAIN_CONFIGS"]["lstm_attention"]) != contract["train"]:
        raise ValueError("Notebook 1's LSTM recipe differs from the supplied run; resolve this before rebuilding.")
    clone = tagged(original, "clone").replace('OUTPUT_ROOT = WORKSPACE / "nmt"',
                                               'OUTPUT_ROOT = WORKSPACE / "nmt_lstm_only"')
    configuration = "\n\n".join([
        "# Frozen settings copied from the supplied run and notebook 1.",
        "REFERENCE_MODEL_CONFIG = " + pformat(model_settings, sort_dicts=False),
        "REFERENCE_TRAIN_CONFIG = " + pformat(contract["train"], sort_dicts=False),
        "REFERENCE_DATA_CONFIG = " + pformat(contract["data_contract"]["config"], sort_dicts=False),
        "EXPECTED_DATA_FINGERPRINT = " + repr(contract["data_fingerprint"]),
        "EXPECTED_MODEL_PARAMETERS = " + repr(pilot["parameters"]),
        "REFERENCE_SOURCE_REVISION = " + repr(contract["source_revision"]),
        dedent('''
        from dataclasses import asdict
        from nmt_assignment.config import DataConfig, ModelConfig, TrainConfig
        from nmt_assignment.data import prepare_data
        from nmt_assignment.workflow import (resource_pilot, learning_diagnostic, run_experiment,
            collect_results, plot_comparison, archive_translation_outputs)
        from qa_assignment.utils import write_json

        MODEL_CONFIG = ModelConfig(**REFERENCE_MODEL_CONFIG)
        TRAIN_CONFIG = TrainConfig(**REFERENCE_TRAIN_CONFIG)
        DATA_CONFIG = DataConfig(**REFERENCE_DATA_CONFIG)
        RUN_RESOURCE_PILOT = True
        RUN_LEARNING_DIAGNOSTIC = True
        DIAGNOSTIC_STEPS = 1000
        BENCHMARK_EXAMPLES = 32
        print("Model:", asdict(MODEL_CONFIG))
        print("Training:", asdict(TRAIN_CONFIG))
        ''').strip(),
        notebook_helper_import("archive_stage_outputs"),
    ])
    notebook = nbf.v4.new_notebook(cells=[
        md('''
        # Temporary LSTM-only recovery — English → Vietnamese

        Enable Kaggle **Internet** and a **GPU**, then run all cells. Only LSTM +
        attention is trained. The merge utility is bundled for compatible older
        GitHub checkouts; model/training helpers still load from the repository.
        No earlier Kaggle outputs need to be attached: this notebook downloads the
        same pinned corpus and checks its fingerprint before any training.

        The recipe matches the supplied run: width 192, two encoder/two decoder layers,
        eight epochs maximum, FP32, effective batch 32, LR 0.001, seed 42, the same
        early stopping, 25,000 training pairs and fixed 512-example validation scoring.
        The short memorization diagnostic is advisory and has **no skip path**.

        Download the final **merge-ready ZIP** and extract its contents directly into
        `kaggle_runs/rnn_lstm_trsf/`. It contains only LSTM-specific folders, so your
        completed RNN/Transformer models and shared prepared data are retained.
        '''),
        code(clone, "clone"),
        md("## Install the same dependencies and load helpers"),
        code(tagged(original, "setup"), "setup"),
        md("## Frozen model, optimizer and data settings"),
        code(configuration, "configuration"),
        md('''
        ## Prepare exactly the same corpus

        The check includes split files and both tokenizers. If the fingerprint differs,
        this cell stops instead of training a model that would not match the comparison.
        '''),
        code('''
        bundle = prepare_data(OUTPUT_ROOT / "data", DATA_CONFIG)
        actual_fingerprint = bundle.manifest["data_fingerprint"]
        if actual_fingerprint != EXPECTED_DATA_FINGERPRINT:
            raise ValueError("Prepared data differs from the supplied run. Do not merge these outputs. "
                "Use the original prepared data with identical DATA_CONFIG before retrying.")
        print("Verified identical prepared data:", actual_fingerprint)
        print("Counts:", bundle.manifest["counts"])
        write_json(OUTPUT_ROOT / "results" / MODEL_CONFIG.stage / f"seed_{TRAIN_CONFIG.seed}" / "recovery_recipe.json",
            {"reference_source_revision": REFERENCE_SOURCE_REVISION,
             "expected_data_fingerprint": EXPECTED_DATA_FINGERPRINT,
             "model": asdict(MODEL_CONFIG), "train": asdict(TRAIN_CONFIG), "data": asdict(DATA_CONFIG),
             "purpose": "recover the missing LSTM stage; same training recipe, advisory memorization probe"})
        ''', "data"),
        md('''
        ## LSTM resource and learning probes

        Probe weights are discarded. Missing chrF 90 reports the actual result and
        proceeds to main training; nonfinite losses/gradients still raise an error.
        '''),
        code('''
        try:
            if RUN_RESOURCE_PILOT:
                pilot = resource_pilot(bundle, MODEL_CONFIG, TRAIN_CONFIG, DEVICE)
                if pilot["parameters"] != EXPECTED_MODEL_PARAMETERS:
                    raise ValueError("LSTM parameter count differs from the original recipe; inspect the model settings.")
                print(pilot)
                write_json(OUTPUT_ROOT / "results" / "pilots" / "lstm_attention.json", pilot)
            if RUN_LEARNING_DIAGNOSTIC:
                diagnostic = learning_diagnostic(bundle, MODEL_CONFIG, OUTPUT_ROOT,
                                                DEVICE, steps=DIAGNOSTIC_STEPS)
                print("Diagnostic chrF:", diagnostic["final"]["chrf"])
                if not diagnostic["overfit_demonstrated"]:
                    print("Memorization target not reached in the short probe. Main LSTM training remains enabled.")
        finally:
            print("Partial/recovery archive:", archive_translation_outputs(OUTPUT_ROOT))
        ''', "diagnostics"),
        md('''
        ## Train LSTM, select its best checkpoint, export and evaluate

        This cell always calls the LSTM training helper after successful data preparation.
        Its learning curve uses the same validation chrF selection and early stopping.
        Checkpoints, test predictions, benchmark metrics and the portable export use
        the standard `lstm_attention/seed_42/` paths.
        '''),
        code('''
        try:
            summary = run_experiment(bundle, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_ROOT,
                device=DEVICE, repo_root=REPO_ROOT, resume="auto", benchmark_examples=BENCHMARK_EXAMPLES)
            print(summary["test"])
        finally:
            print("Full recovery archive:", archive_translation_outputs(OUTPUT_ROOT))
        ''', "train"),
        md("## Inspect results and try a translation"),
        code('''
        table = collect_results(OUTPUT_ROOT)
        display(table)
        figure = plot_comparison(table, OUTPUT_ROOT)
        if figure is not None:
            display(figure)
        from nmt_assignment.inference import Translator
        MODEL_DIR = OUTPUT_ROOT / "models" / "lstm_attention" / f"seed_{TRAIN_CONFIG.seed}"
        translator = Translator.from_export(MODEL_DIR, device=DEVICE)
        ENGLISH = "Scientists are studying how the climate is changing."
        print("EN:", ENGLISH)
        print("VI:", translator.translate(ENGLISH))
        ''', "comparison"),
        md('''
        ## Download and merge

        Download the merge-ready ZIP below and extract **inside** your local
        `kaggle_runs/rnn_lstm_trsf/` folder. Its top level is `checkpoints/`, `results/`
        and `models/`; there is no extra `nmt_lstm_only/` enclosing directory.

        Your new summary will be `results/lstm_attention/seed_42/summary.json`.
        The archive includes LSTM's checkpoints, model/tokenizer, predictions,
        learning curves, diagnostics and recipe metadata. It omits shared data and
        top-level comparison CSV/plots, avoiding replacement with a LSTM-only table.
        The previous combined CSV/plot can be regenerated during the next local analysis.
        Save this Kaggle run's console log as `lstm_only.log` alongside your earlier log.
        '''),
        code('''
        merge_archive = archive_stage_outputs(OUTPUT_ROOT, "lstm_attention", seed=TRAIN_CONFIG.seed)
        print("DOWNLOAD THIS ZIP:", merge_archive)
        print("Extract its contents into: kaggle_runs/rnn_lstm_trsf/")
        from IPython.display import FileLink
        display(FileLink(str(merge_archive)))
        ''', "merge_archive"),
    ])
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.13"}
    notebook.metadata["nmt_assignment"] = {"workflow": "temporary_lstm_recovery",
        "reference_model_config": model_settings, "reference_train_config": contract["train"],
        "reference_data_config": contract["data_contract"]["config"],
        "expected_data_fingerprint": contract["data_fingerprint"]}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    nbf.validate(notebook)
    destination = ROOT / "temp_lstm_only_kaggle.ipynb"
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        nbf.write(notebook, stream)
    print(destination)


if __name__ == "__main__":
    main()
