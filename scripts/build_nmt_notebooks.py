"""Generate two Kaggle NMT interfaces; preserve editable controls and the local QA notebook."""

import ast
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source, tag):
    return nbf.v4.new_code_cell(dedent(source).strip(), metadata={"tags": [tag]})


def common_cells(pretrained=False):
    scope = "4: pretrained Marian reference" if pretrained else "1–3: RNN → LSTM + attention → Transformer"
    return [
        md(f"""
        # English → Vietnamese NMT — stages {scope}

        Enable Kaggle **Internet** and a **GPU**. Helpers live in `src/nmt_assignment/`;
        this notebook only controls preparation and experiments. Both notebooks use
        the same pinned IWSLT Parquet corpus, source-disjoint splits, and evaluation scope.

        | Stage | Milestone | Model | Training |
        |---|---|---|---|
        | 1 | RNN/backprop era, 1986 | Tanh RNN encoder–decoder | Scratch |
        | 2 | LSTM 1997; attention 2015 | LSTM encoder–decoder + dot attention | Scratch |
        | 3 | Transformer 2017 | Self-attention encoder + causal self/cross-attention decoder | Scratch |
        | 4 | Marian checkpoint 2020 | Helsinki-NLP/opus-mt-en-vi | Pretrained + fine-tuning |

        These dates describe components/checkpoint release, not the invention of
        encoder–decoder translation (2014). Increasing capability is the motivation;
        measured BLEU, chrF and speed need not increase in this order. LSTM + attention
        changes both recurrence and conditioning, so this is a progression of recipes.
        Marian uses external pretraining; prior IWSLT exposure cannot be ruled out.

        References: [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf),
        [dot attention](https://arxiv.org/abs/1508.04025),
        [Transformer](https://arxiv.org/abs/1706.03762),
        [Marian model card](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi).
        """),
        code('''
        from pathlib import Path
        import subprocess
        import sys

        REPO_URL = ""  # your GitHub repository URL
        REPO_REF = "main"  # use the SAME source commit in both notebooks
        SOURCE_SUBDIR = "."
        WORKSPACE = Path("/kaggle/working")
        CLONE_ROOT = WORKSPACE / "nmt_source"
        OUTPUT_ROOT = WORKSPACE / "nmt"
        RESTORE_FROM = None  # extracted prior output root containing checkpoints/results/data/models

        if not REPO_URL.strip():
            raise ValueError("Fill in REPO_URL after pushing the source to GitHub.")
        if not WORKSPACE.is_dir():
            raise RuntimeError("Run this training interface on Kaggle.")
        if not (CLONE_ROOT / ".git").is_dir():
            subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", REPO_URL, str(CLONE_ROOT)], check=True)
        subprocess.run(["git", "-C", str(CLONE_ROOT), "fetch", "--depth", "1", "origin", REPO_REF], check=True)
        subprocess.run(["git", "-C", str(CLONE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
        REPO_ROOT = (CLONE_ROOT / SOURCE_SUBDIR).resolve()
        if not (REPO_ROOT / "pyproject.toml").is_file():
            raise FileNotFoundError("Check SOURCE_SUBDIR.")
        print("Source:", REPO_ROOT)
        print("Artifacts:", OUTPUT_ROOT)
        ''', "clone"),
        md("""
        ## Install and expose helpers in the running kernel

        Setup preserves Kaggle's CUDA PyTorch. No Mamba/compiler installation is used.
        If Transformers was imported before setup, restart the session first.
        The upstream pretrained Marian `.bin` export requires **PyTorch ≥2.6** with
        the pinned Transformers release; scratch experiments also work on PyTorch 2.5.
        """),
        code('''
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(REPO_ROOT / "requirements-kaggle.txt")], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_ROOT)], check=True)
        import importlib
        SOURCE_ROOT = (REPO_ROOT / "src").resolve()
        if str(SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(SOURCE_ROOT))
        importlib.invalidate_caches()
        import nmt_assignment
        if Path(nmt_assignment.__file__).resolve().parent != SOURCE_ROOT / "nmt_assignment":
            raise RuntimeError("Helpers came from another checkout; restart the session.")
        import torch
        from nmt_assignment.workflow import restore_artifacts
        from qa_assignment.utils import environment_info
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a Kaggle GPU accelerator.")
        DEVICE = "cuda:0"
        if RESTORE_FROM is not None:
            print(restore_artifacts(RESTORE_FROM, OUTPUT_ROOT))
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print("Loaded helpers:", nmt_assignment.__file__)
        print(environment_info(REPO_ROOT))
        ''', "setup"),
        md("""
        ## Download the corpus and freeze common splits

        The whole pinned corpus is downloaded; the default **25,000-pair training pilot**
        is selected after reserving validation. Set `max_train_examples=None` for the full
        eligible corpus in BOTH notebooks, using a NEW output root for changed settings.
        Source duplicates are removed, test sources are excluded from training, and
        validation is reserved before sampling. Unicode accents and case are preserved;
        HTML escapes and whitespace are normalized.

        Scratch SentencePiece BPE is learned only from selected training sentences.
        Marian keeps its pretrained source/target tokenizer, including `>>vie<<`.
        Length eligibility is checked with BOTH tokenizers, so every stage uses the
        same example IDs. Nothing is silently truncated. Scores refer to the retained
        IWSLT test subset; the audit reports excluded examples and the data fingerprint.
        The community mirror has sparse licensing/provenance documentation.
        """),
        code('''
        from dataclasses import replace
        from nmt_assignment.config import DataConfig, ModelConfig, TrainConfig, MILESTONES
        from nmt_assignment.data import prepare_data
        from nmt_assignment.workflow import (resource_pilot, learning_diagnostic, run_experiment,
                                             collect_results, plot_comparison, archive_translation_outputs)
        from qa_assignment.utils import write_json

        DATA_CONFIG = DataConfig(max_train_examples=25000, validation_examples=5000,
                                 vocab_size=8000, max_input_length=96, max_output_length=96)
        bundle = prepare_data(OUTPUT_ROOT / "data", DATA_CONFIG)
        print(bundle.manifest)
        print("Scratch vocabulary:", len(bundle.tokenizer))
        print("Marian vocabulary:", len(bundle.pretrained_tokenizer))
        print("EN:", bundle.train.rows[0]["source"])
        print("VI:", bundle.train.rows[0]["target"])
        ''', "data"),
    ]


def scratch_cells():
    return common_cells() + [
        md("""
        ## Ordered experiments

        Scratch capacity increases: RNN width 128 / one layer, attentive LSTM width
        192 / two layers, Transformer width 256 / two layers. They share a train-only
        vocabulary, split IDs, effective batch 32, and greedy decoding. This compares
        declared recipes, not architecture at matched parameter counts. Defaults are
        at most eight epochs, with chrF early stopping.
        Patience counts development evaluations, not noisy training-loss log entries.
        With epoch-end evaluation, stopping starts counting at epoch three. Set
        patience to zero to disable it, and lower the epoch budget if desired.
        Validation uses the same fixed 512-example subset for every model; set
        `development_limit=None` in BOTH notebooks to use the full validation set.
        """),
        code('''
        STAGE_ORDER = ("rnn", "lstm_attention", "transformer")
        MODEL_CONFIGS = {
            "rnn": ModelConfig(stage="rnn", d_model=128, encoder_layers=1, decoder_layers=1),
            "lstm_attention": ModelConfig(stage="lstm_attention", d_model=192, feedforward_dim=768),
            "transformer": ModelConfig(stage="transformer", d_model=256, feedforward_dim=1024),
        }
        TRAIN_CONFIG = TrainConfig(epochs=8, micro_batch_size=16, accumulation_steps=2,
            learning_rate=1e-3, precision="fp32", development_limit=512,
            save_every_steps=250, early_stopping_patience=3,
            early_stopping_min_delta=0.2, early_stopping_min_epochs=3)
        TRAIN_CONFIGS = {stage: TRAIN_CONFIG for stage in STAGE_ORDER}
        SEEDS = [42]
        RUN_RESOURCE_PILOT = True
        RUN_LEARNING_DIAGNOSTIC = True
        REQUIRE_DIAGNOSTIC_PASS = True
        DIAGNOSTIC_STEPS = 500
        BENCHMARK_EXAMPLES = 32
        ''', "configuration"),
        md("""
        ## Resource and learning diagnostics

        Resource pilots run real forward/backward/generation on the selected accelerator.
        Their short-batch time estimates exclude evaluation, checkpointing and sentence
        length variation. The learning diagnostic memorizes eight short TRAIN examples
        with dropout off, FP32 and LR 0.003. Its weights are discarded. chrF ≥90 is the
        pass criterion. A failure skips expensive main training for that model by default;
        inspect `results/diagnostics/` before changing the gate.
        """),
        code('''
        diagnostics = {}
        try:
            for stage in STAGE_ORDER:
                if RUN_RESOURCE_PILOT:
                    report = resource_pilot(bundle, MODEL_CONFIGS[stage], TRAIN_CONFIGS[stage], DEVICE)
                    print(report)
                    write_json(OUTPUT_ROOT / "results" / "pilots" / f"{stage}.json", report)
                if RUN_LEARNING_DIAGNOSTIC:
                    diagnostics[stage] = learning_diagnostic(bundle, MODEL_CONFIGS[stage], OUTPUT_ROOT,
                                                            DEVICE, steps=DIAGNOSTIC_STEPS)
        finally:
            archive_path = archive_translation_outputs(OUTPUT_ROOT)
            print("Diagnostic artifacts:", archive_path)
        ''', "diagnostics"),
        md("""
        ## Train, checkpoint and evaluate

        Each stage/seed saves `last.pt`, the highest-validation-chrF `best.pt`, and up to
        two step snapshots. Resume includes optimizer, scheduler, RNG, accumulation
        boundary, stopping state and source/data contracts. Restore the ENTIRE stage
        checkpoint folder. Test scores are computed after selecting the best checkpoint.
        Exports go to `models/<stage>/seed_42/` and include their tokenizer/configuration.
        """),
        code('''
        summaries = []
        try:
            for stage in STAGE_ORDER:
                if REQUIRE_DIAGNOSTIC_PASS and not diagnostics.get(stage, {}).get("overfit_demonstrated", False):
                    print("Skipping", stage, "because its learning diagnostic did not pass.")
                    continue
                for seed in SEEDS:
                    summary = run_experiment(bundle, MODEL_CONFIGS[stage],
                        replace(TRAIN_CONFIGS[stage], seed=seed), OUTPUT_ROOT, device=DEVICE,
                        repo_root=REPO_ROOT, resume="auto", benchmark_examples=BENCHMARK_EXAMPLES)
                    summaries.append(summary)
                    print(stage, summary["test"])
        finally:
            archive_path = archive_translation_outputs(OUTPUT_ROOT)
            print("Download or save:", archive_path)
        ''', "train"),
    ] + ending_cells()


def pretrained_cells():
    return common_cells(pretrained=True) + [
        md("""
        ## Pretrained translation reference

        This uses Marian EN→VI, not the saved English QA T5. It retains Marian's
        pretrained vocabulary and Vietnamese language prefix. Fine-tuning defaults to
        three epochs, LR 0.00002, and the same chrF/greedy evaluation conventions.
        Prior exposure to IWSLT through external pretraining is unknown; this is a
        practical reference rather than evidence of an architecture-only improvement.

        To combine all four stages, attach notebook 1's extracted artifacts and set
        `RESTORE_FROM` before setup. Keep the source commit, DATA_CONFIG, validation
        limit and decoding settings identical. Results otherwise contain Marian alone.
        """),
        code('''
        from packaging.version import Version
        if Version(torch.__version__.split("+")[0]) < Version("2.6"):
            raise RuntimeError("The upstream Marian .bin requires PyTorch >=2.6. Select a current Kaggle GPU runtime.")
        pretrained_bundle = bundle.for_pretrained()
        MODEL_CONFIG = ModelConfig(stage="marian_en_vi")
        TRAIN_CONFIG = TrainConfig(epochs=3, micro_batch_size=8, accumulation_steps=4,
            learning_rate=2e-5, warmup_fraction=0.05, precision="fp32", development_limit=512,
            save_every_steps=250, early_stopping_patience=2,
            early_stopping_min_delta=0.2, early_stopping_min_epochs=1)
        SEEDS = [42]
        RUN_RESOURCE_PILOT = True
        BENCHMARK_EXAMPLES = 32
        ''', "configuration"),
        md("""
        ## Probe the actual pretrained model

        This checks forward/backward and greedy generation before fine-tuning. The pilot
        weights are discarded and the pinned pretrained weights are loaded again.
        Token cross-entropy values across different vocabularies are not directly
        comparable; compare detokenized BLEU/chrF on the identical test IDs instead.
        """),
        code('''
        if RUN_RESOURCE_PILOT:
            report = resource_pilot(pretrained_bundle, MODEL_CONFIG, TRAIN_CONFIG, DEVICE)
            print(report)
            write_json(OUTPUT_ROOT / "results" / "pilots" / "marian_en_vi.json", report)
        ''', "diagnostics"),
        md("""
        ## Fine-tune, restore the best checkpoint, export and score

        Save Kaggle outputs for persistence. The exported safe-tensor model has its own
        tokenizer and `translation_config.json`; it can be moved to a short local folder.
        The archive contains checkpoints/results/models/prepared data, excluding HF caches.
        """),
        code('''
        summaries = []
        try:
            for seed in SEEDS:
                summary = run_experiment(pretrained_bundle, MODEL_CONFIG, replace(TRAIN_CONFIG, seed=seed),
                    OUTPUT_ROOT, device=DEVICE, repo_root=REPO_ROOT, resume="auto",
                    benchmark_examples=BENCHMARK_EXAMPLES)
                summaries.append(summary)
                print(summary["test"])
        finally:
            archive_path = archive_translation_outputs(OUTPUT_ROOT)
            print("Download or save:", archive_path)
        ''', "train"),
    ] + ending_cells()


def ending_cells():
    return [
        md("""
        ## Historical comparison and manual translation

        Comparison rows follow the declared chronological progression. Log BLEU/chrF
        signatures, teacher-forced loss/token accuracy, examples, elapsed training time,
        parameters, inference latency/throughput, CUDA peak memory and decoded examples.
        A newer model need not beat the previous stage under a small training budget.
        Manual translation loads the exported selected checkpoint once.
        """),
        code('''
        table = collect_results(OUTPUT_ROOT)
        display(table)
        figure = plot_comparison(table, OUTPUT_ROOT)
        if figure is not None:
            display(figure)
        archive_path = archive_translation_outputs(OUTPUT_ROOT)
        print("Final archive:", archive_path)
        ''', "comparison"),
        code('''
        from nmt_assignment.inference import Translator
        # Pick a completed stage. Locally, copy its export to e.g. models/translator/.
        TEST_STAGE = summaries[-1]["stage"] if summaries else None
        if TEST_STAGE is not None:
            MODEL_DIR = OUTPUT_ROOT / "models" / TEST_STAGE / f"seed_{summaries[-1]['seed']}"
            translator = Translator.from_export(MODEL_DIR, device=DEVICE)
            ENGLISH = "Scientists are studying how the climate is changing."
            print("EN:", ENGLISH)
            print("VI:", translator.translate(ENGLISH))
        else:
            print("No completed export; inspect the diagnostic reports first.")
        ''', "translate"),
    ]


def clone_settings(path):
    settings = {}
    if path.exists():
        notebook = nbf.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            for node in ast.parse(cell.source).body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in ("REPO_URL", "REPO_REF", "SOURCE_SUBDIR") and isinstance(node.value, ast.Constant):
                        settings[node.targets[0].id] = node.value.value
    return settings


def main():
    specs = (("01_nmt_scratch_kaggle.ipynb", "01_attention_mamba_kaggle.ipynb", scratch_cells),
             ("02_nmt_pretrained_kaggle.ipynb", "02_t5_transfer_learning_kaggle.ipynb", pretrained_cells))
    for name, legacy, builder in specs:
        destination = ROOT / name
        cells = builder()
        settings = clone_settings(destination if destination.exists() else ROOT / legacy)
        for setting, value in settings.items():
            tree = ast.parse(cells[1].source)
            lines = cells[1].source.splitlines()
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == setting:
                    lines[node.lineno - 1] = f"{setting} = {value!r}"
            cells[1].source = "\n".join(lines)
        if destination.exists():
            # Preserve editable control cells and user-added cells on subsequent builds.
            existing = nbf.read(destination, as_version=4)
            controls = {"clone", "data", "configuration"}
            refreshed = {cell.metadata["tags"][0]: cell for cell in cells if cell.cell_type == "code"}
            for cell in existing.cells:
                tags = cell.metadata.get("tags", [])
                if cell.cell_type == "code" and tags and tags[0] in refreshed and tags[0] not in controls:
                    cell.source = refreshed[tags[0]].source
            notebook = existing
        else:
            notebook = nbf.v4.new_notebook(cells=cells)
            notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
            notebook.metadata.language_info = {"name": "python", "version": "3.11"}
        notebook.metadata["nmt_assignment"] = {"version": 1, "workflow": "pretrained" if name.startswith("02") else "scratch"}
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
                ast.parse(cell.source)
        nbf.validate(notebook)
        nbf.write(notebook, destination)
        print(destination)
    # The saved QA notebook is intentionally never read, written or regenerated here.


if __name__ == "__main__":
    main()
