"""Regenerate the two lightweight Kaggle interfaces without notebook outputs."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


def common_cells(title, description):
    return [
        md(f"""
        # {title}

        {description}

        Enable **Internet** and a **GPU accelerator** in Kaggle settings. Edit `REPO_URL`
        below after pushing this project to GitHub. The workflow uses GPU 0, even if
        Kaggle exposes two GPUs. Long implementations live in `src/qa_assignment/`.

        Artifacts go to `/kaggle/working/qa_assignment/`. Each variant and seed has its
        own checkpoint directory. Kaggle working files are temporary: download the
        archive or save the notebook outputs, then attach those outputs to resume a
        later session. Checkpoints are written every 250 optimizer steps and each epoch.
        """),
        code("""
        from pathlib import Path
        import subprocess
        import sys

        REPO_URL = ""  # fill in your GitHub repository URL after pushing the project
        REPO_REF = "main"  # preferably the same commit SHA in BOTH notebooks
        SOURCE_SUBDIR = "."  # edit if this assignment lives below the repository root
        WORKSPACE = Path("/kaggle/working")
        CLONE_ROOT = WORKSPACE / "qa_assignment_source"
        OUTPUT_ROOT = WORKSPACE / "qa_assignment"
        RESTORE_FROM = None  # e.g. Path('/kaggle/input/prior-qa-output/qa_assignment')

        if not REPO_URL.strip():
            raise ValueError("Set REPO_URL to the GitHub repository you pushed this project to.")
        if not WORKSPACE.is_dir():
            raise RuntimeError("This notebook is intended for Kaggle. Run local tests with pytest.")
        if not (CLONE_ROOT / ".git").is_dir():
            subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", REPO_URL, str(CLONE_ROOT)], check=True)
        subprocess.run(["git", "-C", str(CLONE_ROOT), "fetch", "--depth", "1", "origin", REPO_REF], check=True)
        subprocess.run(["git", "-C", str(CLONE_ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
        REPO_ROOT = (CLONE_ROOT / SOURCE_SUBDIR).resolve()
        assert (REPO_ROOT / "pyproject.toml").is_file(), "Check SOURCE_SUBDIR."
        print("Source:", REPO_ROOT)
        print("Artifacts:", OUTPUT_ROOT)
        subprocess.run(["git", "-C", str(CLONE_ROOT), "rev-parse", "HEAD"], check=True)
        """),
        md("""
        ## Install the Python helpers

        Run setup before importing project modules. These requirements pin the Hugging
        Face interface used here and preserve Kaggle's preinstalled CUDA-enabled PyTorch.
        Setup also adds the cloned `src/` directory to this running kernel's import path;
        a subprocess editable install alone cannot refresh an already-started kernel.
        If you previously imported Transformers in this session, restart the session
        after installing and rerun from the top. A successful import is followed by an
        actual forward/backward and generation probe before training.
        """),
        code("""
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(REPO_ROOT / "requirements-kaggle.txt")], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                        "-e", str(REPO_ROOT)], check=True)

        # Editable-install .pth files are read at interpreter startup. This kernel
        # predates the pip subprocess, so explicitly expose the cloned sources now.
        import importlib

        SOURCE_ROOT = (REPO_ROOT / "src").resolve()
        if not (SOURCE_ROOT / "qa_assignment" / "__init__.py").is_file():
            raise FileNotFoundError(f"Missing qa_assignment sources under {SOURCE_ROOT}; check SOURCE_SUBDIR.")
        if str(SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(SOURCE_ROOT))
        importlib.invalidate_caches()

        import qa_assignment
        if Path(qa_assignment.__file__).resolve().parent != SOURCE_ROOT / "qa_assignment":
            raise RuntimeError("qa_assignment was loaded from another checkout. Restart the session and rerun setup.")
        print("Loaded helpers:", qa_assignment.__file__)

        import torch
        from qa_assignment.utils import environment_info
        from qa_assignment.runtime import restore_artifacts

        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU accelerator in Kaggle settings.")
        DEVICE = "cuda:0"
        if RESTORE_FROM is not None:
            print(restore_artifacts(RESTORE_FROM, OUTPUT_ROOT))
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        environment_info(REPO_ROOT)
        """),
    ]


def data_cells():
    return [
        md("""
        ## Download the whole dataset and prepare identical splits

        Both official SQuAD 1.1 files are downloaded in full. `MAX_TRAIN_EXAMPLES=None`
        trains on all eligible training examples; set 10,000 for a shorter pilot project.
        Examples are retained only when the full formatted question and context fit
        the input cap. No answer-guided cropping occurs. Internal development articles
        come exclusively from training, and official validation is reserved for final
        reporting. Scores describe the retained subset.

        Copy the exact `DataConfig` between notebooks. Compare `data_fingerprint` values
        before comparing results. You can restore `data/` from previous output to reuse
        tokenization, but both notebooks can independently reproduce the same split.
        """),
        code("""
        from qa_assignment.config import DataConfig
        from qa_assignment.data import prepare_data, QACollator

        DATA_CONFIG = DataConfig(
            max_input_length=512,
            max_output_length=64,
            development_fraction=0.1,
            split_seed=2026,
            max_train_examples=None,
        )
        bundle = prepare_data(OUTPUT_ROOT / "data", DATA_CONFIG)
        collator = QACollator(bundle.tokenizer, DATA_CONFIG)
        print("Data fingerprint:", bundle.manifest["data_fingerprint"])
        print("Prepared cache:", bundle.directory)
        print("Train / development / validation:", len(bundle.train), len(bundle.development), len(bundle.validation))
        bundle.manifest["statistics"]
        """),
    ]


def ending_cells(baselines=False):
    return [
        md("""
        ## Results and output archive

        Results include normalized EM/F1, predictions, selected checkpoint, resource
        measurements, and warmed timing repeats. Actual QA latency and forced 32-token
        decoding are separate measurements. T5 and the controlled models are plotted
        separately. Import prior `results/` and `checkpoints/` to see runs from the other
        notebook. Matching data fingerprints are required for comparison.
        """ if not baselines else """
        ## Compare all four encoder-decoder models

        The ordered table and combined plots include RNN, LSTM, attention, and T5.
        Every run logs normalized EM/F1, training loss/time, parameter count, GPU
        memory, batch-one QA latency, throughput, and forced 32-token decoding.
        T5 is labeled pretrained; epochs and learning rates remain visible in the
        table. This measures the declared training recipes, rather than attributing
        the benefit of pretraining to architecture alone. Only the selected variants
        with matching data fingerprints enter this comparison.
        """),
        code("""
        from qa_assignment.workflow import collect_results, plot_results, archive_outputs

        table = collect_results(OUTPUT_ROOT)
        display(table)
        if not table.empty:
            figures = plot_results(table, OUTPUT_ROOT)
            import matplotlib.pyplot as plt
            plt.show()
        """ if not baselines else """
        from qa_assignment.workflow import collect_results, plot_results, archive_outputs

        table = collect_results(OUTPUT_ROOT, variants=MODEL_VARIANTS)
        display(table)
        if not table.empty:
            figures = plot_results(table, OUTPUT_ROOT, combine_groups=True)
            import matplotlib.pyplot as plt
            plt.show()
        """),
        code("""
        INCLUDE_DATA_IN_ARCHIVE = True  # enables reuse in the second notebook
        archive_path = archive_outputs(OUTPUT_ROOT, include_data=INCLUDE_DATA_IN_ARCHIVE)
        print("Download from Kaggle Output:", archive_path)
        print("Checkpoint root:", OUTPUT_ROOT / "checkpoints")
        """),
        md("""
        ## Resume in another Kaggle session

        Save/download these outputs, attach the prior output as a Kaggle input, and set
        `RESTORE_FROM` to the folder containing `checkpoints/`, `results/`, and optionally
        `data/`. Restore before preparing new artifacts. The helper copies read-only
        Kaggle inputs into writable working storage; existing folders are never overwritten.
        `resume='auto'` loads each variant's `last.pt`. Keep the complete checkpoint folder,
        including `best.pt`, and use identical model/data/training settings and source commit.

        To start a different experiment, change `OUTPUT_ROOT`; changing settings inside an
        existing run is rejected. `last.pt` resumes training; `best.pt` is selected using
        internal-development F1. Load only checkpoints you created or otherwise trust.
        """),
    ]


def build_attention_mamba():
    cells = common_cells("Attention / Mamba encoder-decoder experiments on Kaggle",
                         "Train the four scratch variants, retaining shared decoder cross-attention.")
    cells += [
        md("""
        ## Mamba CUDA setup

        Install the pinned Mamba-1 implementation against Kaggle's existing PyTorch.
        This may download a matching binary wheel or compile an extension; setup time
        is excluded from model benchmarks. The helper requires real selective-scan and
        causal-convolution CUDA extensions. It never substitutes a slow Python model.
        Select a modern GPU runtime; on older accelerators a Triton/kernel probe can
        fail. In that case stop and select a compatible runtime, rather than reporting
        fallback timing as Mamba efficiency. See README for troubleshooting.
        """),
        code("""
        from qa_assignment.runtime import install_mamba, require_mamba_kernels

        INSTALL_MAMBA = True
        if INSTALL_MAMBA:
            install_mamba()
        print(require_mamba_kernels())
        """),
    ] + data_cells() + [
        md("""
        ## Experiment configuration

        Start small and compare the pilot estimates before training. All variants use
        identical dimensions, feed-forward shells, positional embeddings, tokenizer,
        data, and training recipe. Mamba encoders scan both directions; Mamba decoders
        are causal. `mamba_mamba` still uses cross-attention.

        The default runs all four variants sequentially, one on GPU at a time. Set
        `MODEL_VARIANTS=['attn_attn', 'mamba_attn']` for the core scope. Use the same
        precision for all variants. FP32 is the initial compatibility setting; FP16
        needs a successful probe. Multiple seeds multiply training cost.
        """),
        code("""
        from dataclasses import replace
        from qa_assignment.config import ModelConfig, TrainConfig, BenchmarkConfig, VARIANTS

        MODEL_VARIANTS = list(VARIANTS)
        SEEDS = [42]  # extend to [42, 43, 44] if your budget permits
        MODEL_CONFIG = ModelConfig(d_model=128, encoder_layers=2, decoder_layers=2,
                                   heads=4, feedforward_dim=512, dropout=0.1)
        TRAIN_CONFIG = TrainConfig(epochs=5, micro_batch_size=4, accumulation_steps=4,
                                   eval_batch_size=8, learning_rate=3e-4, precision="fp32",
                                   save_every_steps=250, development_limit=None)
        BENCHMARK_CONFIG = BenchmarkConfig(examples=200, repeats=3,
                                           throughput_batch_size=4, fixed_output_tokens=32)
        print("Effective batch:", TRAIN_CONFIG.micro_batch_size * TRAIN_CONFIG.accumulation_steps)
        print("Variants:", MODEL_VARIANTS, "Seeds:", SEEDS)
        """),
        md("""
        ## Compatibility and resource pilot

        The pilot verifies finite training gradients and cached/uncached decoding logits,
        then times 50 warmed training microsteps on a repeated small batch. Pilot models
        are discarded; main runs start fresh. The epoch estimate excludes development
        evaluation, checkpoints, and final generation. Inspect the memory and timing
        results before proceeding. Change shared configuration before main runs if needed.
        """),
        code("""
        from qa_assignment.runtime import run_pilot
        from qa_assignment.utils import write_json

        pilots = {}
        for variant in MODEL_VARIANTS:
            pilots[variant] = run_pilot(bundle, replace(MODEL_CONFIG, variant=variant),
                                        TRAIN_CONFIG, collator, DEVICE, steps=50, repo_root=REPO_ROOT)
            print(variant, pilots[variant])
        write_json(OUTPUT_ROOT / "results" / "controlled_pilots.json", pilots)
        """),
        md("""
        ## Optional tiny-subset learning diagnostic

        Enable this to check that a fresh model can learn four fixed training examples.
        This diagnostic is separate from the main experiment and final validation.
        Increase steps if it has not overfit; inspect the answer examples and losses.
        A successful resource probe alone does not show that a model learned QA.
        """),
        code("""
        from qa_assignment.diagnostics import overfit_diagnostic

        RUN_OVERFIT_DIAGNOSTIC = False
        if RUN_OVERFIT_DIAGNOSTIC:
            for variant in MODEL_VARIANTS:
                report = overfit_diagnostic(bundle, replace(MODEL_CONFIG, variant=variant),
                                             collator, DEVICE, steps=300)
                print(variant, report)
        """),
        md("""
        ## Train, checkpoint, final evaluation, and benchmark

        Training is enabled below, as in the sample notebook. Run this cell after
        reviewing the pilot results. Checkpoints are isolated by variant and seed.
        Existing matching runs resume automatically; optimizer, scheduler, scaler,
        random state, and within-epoch cursor are restored. Final validation runs only
        after training and checkpoint selection finish.
        """),
        code("""
        from qa_assignment.workflow import run_experiment, archive_outputs

        RUN_TRAINING = True
        summaries = []
        if RUN_TRAINING:
            try:
                for variant in MODEL_VARIANTS:
                    for seed in SEEDS:
                        summaries.append(run_experiment(
                            bundle, replace(MODEL_CONFIG, variant=variant), replace(TRAIN_CONFIG, seed=seed),
                            OUTPUT_ROOT, device=DEVICE, resume="auto",
                            benchmark_config=BENCHMARK_CONFIG, repo_root=REPO_ROOT,
                        ))
            finally:
                print("Available checkpoints/results archived at:", archive_outputs(OUTPUT_ROOT))
        else:
            print("Training disabled; set RUN_TRAINING=True after inspecting pilots.")
        """),
    ] + ending_cells()
    return cells


def build_baselines():
    cells = common_cells("RNN → LSTM → Attention → T5 encoder-decoder QA on Kaggle",
                         "Train three scratch encoder-decoders and fine-tune T5-small using one shared QA workflow.")
    cells[1].source = cells[1].source.replace('OUTPUT_ROOT = WORKSPACE / "qa_assignment"',
                                             'OUTPUT_ROOT = WORKSPACE / "qa_baselines"')
    cells[0].source = cells[0].source.replace("/kaggle/working/qa_assignment/", "/kaggle/working/qa_baselines/")
    cells += data_cells() + [
        md("""
        ## Four models, one dataset and metric pipeline

        RNN and LSTM use PyTorch's built-in `nn.RNN` (tanh) and `nn.LSTM` for both
        encoder and decoder. The encoder's final state initializes the decoder;
        packed input sequences exclude right padding from that state. These two
        baselines have no cross-attention. The attention model uses PyTorch SDPA
        with a bidirectional self-attention encoder, causal self-attention decoder,
        and decoder cross-attention. T5-small uses Hugging Face pretrained weights.

        No Mamba installation or CUDA-extension compilation is needed. All four
        use the same tokenizer, retained example IDs, input/output caps, article
        partition, greedy generation, and metrics. Scratch models share width,
        depth, optimizer settings, and sample order; parameter counts differ.
        T5 has its own declared fine-tuning recipe and is labeled pretrained.
        Runs execute sequentially in the order below, one model on GPU at a time.
        """),
        code("""
        from dataclasses import replace
        from qa_assignment.config import ModelConfig, TrainConfig, BenchmarkConfig, BASELINE_VARIANTS

        MODEL_VARIANTS = list(BASELINE_VARIANTS)  # rnn_rnn, lstm_lstm, attn_attn, t5_small
        SEEDS = [42]
        PRECISION = "fp32"  # keep common; change only after every model passes its probe
        SCRATCH_MODEL_CONFIG = ModelConfig(d_model=128, encoder_layers=2, decoder_layers=2,
                                           heads=4, feedforward_dim=512, dropout=0.1)
        EARLY_STOPPING_PATIENCE = 5  # development checks, not epochs
        EARLY_STOPPING_MIN_DELTA = 0.1  # absolute F1 points; scores run from 0 to 100
        EARLY_STOPPING_MIN_STEPS = 5000  # let scratch models begin learning before counting failures
        EVAL_EVERY_STEPS = 1000  # also evaluates at epoch end; optimizer steps, not microbatches
        SCRATCH_TRAIN_CONFIG = TrainConfig(epochs=20, micro_batch_size=4, accumulation_steps=4,
                                           eval_batch_size=8, learning_rate=3e-4, precision=PRECISION,
                                           save_every_steps=250, development_limit=None,
                                           eval_every_steps=EVAL_EVERY_STEPS,
                                           early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                           early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
                                           early_stopping_min_steps=EARLY_STOPPING_MIN_STEPS)
        T5_MODEL_CONFIG = ModelConfig(variant="t5_small", t5_name="google-t5/t5-small")
        T5_TRAIN_CONFIG = replace(SCRATCH_TRAIN_CONFIG, epochs=5, learning_rate=1e-4)
        MODEL_CONFIGS = {variant: T5_MODEL_CONFIG if variant == "t5_small" else
                         replace(SCRATCH_MODEL_CONFIG, variant=variant) for variant in MODEL_VARIANTS}
        TRAIN_CONFIGS = {variant: T5_TRAIN_CONFIG if variant == "t5_small" else
                         SCRATCH_TRAIN_CONFIG for variant in MODEL_VARIANTS}
        BENCHMARK_CONFIG = BenchmarkConfig(examples=200, repeats=3,
                                           throughput_batch_size=4, fixed_output_tokens=32)
        for variant in MODEL_VARIANTS:
            training = TRAIN_CONFIGS[variant]
            print(variant, "epochs:", training.epochs, "lr:", training.learning_rate,
                  "effective batch:", training.micro_batch_size * training.accumulation_steps)
        """),
        md("""
        ## Compatibility and resource pilots

        Each fresh model runs forward/backward, finite-gradient, and cached/uncached
        decoding checks, then a 50-step warmed training pilot. Inspect memory and
        estimated epoch training time for all four before starting full runs.
        Estimates exclude evaluation and checkpointing. Reduce shared microbatch
        size and increase accumulation if memory requires it. These pilot models
        are discarded; the main runs begin fresh or resume their own checkpoints.
        """),
        code("""
        from qa_assignment.runtime import run_pilot
        from qa_assignment.utils import write_json

        PILOT_STEPS = 50
        pilots = {}
        for variant in MODEL_VARIANTS:
            pilots[variant] = run_pilot(bundle, MODEL_CONFIGS[variant],
                                        replace(TRAIN_CONFIGS[variant], seed=SEEDS[0]),
                                        collator, DEVICE, steps=PILOT_STEPS, repo_root=REPO_ROOT)
            print(variant, pilots[variant])
        write_json(OUTPUT_ROOT / "results" / "baseline_pilots.json", pilots)
        """),
        md("""
        ## Optional tiny-subset learning diagnostic

        A successful runtime probe checks execution, not whether a model learns QA.
        Enable this to train each fresh scratch model on four fixed examples and
        inspect its losses and generated answers. Adjust diagnostic steps if needed.
        Diagnostic models and results are separate from the main experiment.
        """),
        code("""
        from qa_assignment.diagnostics import overfit_diagnostic

        RUN_OVERFIT_DIAGNOSTIC = False
        if RUN_OVERFIT_DIAGNOSTIC:
            for variant in MODEL_VARIANTS:
                if variant != "t5_small":
                    print(variant, overfit_diagnostic(bundle, MODEL_CONFIGS[variant],
                                                      collator, DEVICE, steps=300))
        """),
        md("""
        ## Train RNN, LSTM, attention, then fine-tune T5

        Each variant and seed writes its own `checkpoints/<variant>/seed_<seed>/`
        folder with resumable `last.pt`, development-selected `best.pt`, step snapshots,
        configuration, manifest, and dependency versions. Matching runs resume
        automatically. Final official-validation EM/F1 and identical timing workloads
        use the selected checkpoint. T5 also saves the selected `hf_export/`.

        The output root defaults to `/kaggle/working/qa_baselines/` so these runs have
        their own artifacts. Save the notebook outputs to resume a later session.

        Scratch runs have a maximum of 20 epochs; T5 has a maximum of 5. Development
        F1 is checked every 1,000 optimizer steps and at epoch end. After step 5,000,
        five checks without an improvement greater than 0.1 F1 points stop that
        model, then the loop moves to the next model. Training loss is still logged,
        but it is not the stopping signal. The highest observed development F1
        still selects `best.pt`, including improvements smaller than the threshold.
        Official validation is never used for stopping. Patience survives resumption;
        resuming an already-stopped run skips training and evaluates its selected model.

        Start fresh for these updated runs. Leave `RESTORE_FROM=None` and use an
        empty output root. Later resumptions of these runs require the complete
        checkpoint folders, identical configuration, and the same source revision.
        """),
        code("""
        from qa_assignment.workflow import run_experiment, archive_outputs

        RUN_TRAINING = True
        summaries = []
        if RUN_TRAINING:
            try:
                for variant in MODEL_VARIANTS:
                    for seed in SEEDS:
                        summary = run_experiment(
                            bundle, MODEL_CONFIGS[variant], replace(TRAIN_CONFIGS[variant], seed=seed),
                            OUTPUT_ROOT, device=DEVICE, resume="auto",
                            benchmark_config=BENCHMARK_CONFIG, repo_root=REPO_ROOT,
                        )
                        summaries.append(summary)
                        print(summary)
            finally:
                print("Available checkpoints/results archived at:", archive_outputs(OUTPUT_ROOT))
        else:
            print("Training disabled; set RUN_TRAINING=True after inspecting pilots.")
        """),
    ] + ending_cells(baselines=True)
    return cells


def main():
    directory = ROOT
    for name, cells in (("01_attention_mamba_kaggle.ipynb", build_attention_mamba()),
                        ("02_t5_transfer_learning_kaggle.ipynb", build_baselines())):
        notebook = nbf.v4.new_notebook(cells=cells)
        destination = directory / name
        if destination.exists():
            existing = nbf.read(destination, as_version=4)
            migrate_baselines = name.startswith("02_") and existing.metadata.get(
                "qa_assignment", {}).get("workflow") != "seq2seq_baselines"
            if migrate_baselines:
                # New scope: retain the user's clone settings and data controls.
                notebook.cells[1] = existing.cells[1]
                notebook.cells[1].source = notebook.cells[1].source.replace(
                    'OUTPUT_ROOT = WORKSPACE / "qa_assignment"', 'OUTPUT_ROOT = WORKSPACE / "qa_baselines"')
                notebook.cells[5] = existing.cells[5]
            elif len(existing.cells) != len(cells):
                raise ValueError(f"Cell layout changed in {name}; patch it manually to preserve edits.")
            else:
                # On subsequent builds preserve repository, data, and experiment
                # controls; refresh only the shared installation section.
                refresh = {2, 3}
                if name.startswith("02_") and existing.metadata.get("qa_assignment", {}).get("version", 0) < 3:
                    # The stopping upgrade adds the requested controls and 20/5
                    # epoch limits, while retaining clone/data/other cells.
                    refresh.update((7, 12, 13))
                for index in range(len(cells)):
                    if index not in refresh:
                        notebook.cells[index] = existing.cells[index]
                    else:
                        notebook.cells[index].id = existing.cells[index].id
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}}
        if name.startswith("02_"):
            notebook.metadata["qa_assignment"] = {"workflow": "seq2seq_baselines", "version": 3}
        nbf.validate(notebook)
        nbf.write(notebook, destination)
        print(destination.relative_to(ROOT))


if __name__ == "__main__":
    main()
