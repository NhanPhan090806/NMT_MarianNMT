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
        If you previously imported Transformers in this session, restart the session
        after installing and rerun from the top. A successful import is followed by an
        actual forward/backward and generation probe before training.
        """),
        code("""
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(REPO_ROOT / "requirements-kaggle.txt")], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                        "-e", str(REPO_ROOT)], check=True)

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


def ending_cells():
    return [
        md("""
        ## Results and output archive

        Results include normalized EM/F1, predictions, selected checkpoint, resource
        measurements, and warmed timing repeats. Actual QA latency and forced 32-token
        decoding are separate measurements. T5 and the controlled models are plotted
        separately. Import prior `results/` and `checkpoints/` to see runs from the other
        notebook. Matching data fingerprints are required for comparison.
        """),
        code("""
        from qa_assignment.workflow import collect_results, plot_results, archive_outputs

        table = collect_results(OUTPUT_ROOT)
        display(table)
        if not table.empty:
            figures = plot_results(table, OUTPUT_ROOT)
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


def build_t5():
    cells = common_cells("T5-small transfer learning on Kaggle",
                         "Fine-tune pretrained T5 on the same generative QA examples as the scratch models.")
    cells += data_cells() + [
        md("""
        ## Fine-tuning configuration

        T5 is a separate transfer-learning reference. It keeps the same tokenizer,
        input/output limits, article partition, and example IDs, while using pretrained
        weights and its own declared fine-tuning recipe. It does not require Mamba.
        """),
        code("""
        from qa_assignment.config import ModelConfig, TrainConfig, BenchmarkConfig

        MODEL_CONFIG = ModelConfig(variant="t5_small", t5_name="google-t5/t5-small")
        TRAIN_CONFIG = TrainConfig(epochs=3, micro_batch_size=4, accumulation_steps=4,
                                   eval_batch_size=8, learning_rate=1e-4, precision="fp32",
                                   save_every_steps=250, development_limit=None, seed=42)
        BENCHMARK_CONFIG = BenchmarkConfig(examples=200, repeats=3,
                                           throughput_batch_size=4, fixed_output_tokens=32)
        """),
        md("""
        ## Compatibility and memory pilot

        Download T5-small, run forward/backward and cached generation checks, and inspect
        the measured memory/epoch-time estimate. If this does not fit, reduce microbatch
        size and increase accumulation to retain the effective batch size.
        """),
        code("""
        from qa_assignment.runtime import run_pilot
        from qa_assignment.utils import write_json

        pilot = run_pilot(bundle, MODEL_CONFIG, TRAIN_CONFIG, collator, DEVICE,
                          steps=50, repo_root=REPO_ROOT)
        write_json(OUTPUT_ROOT / "results" / "t5_pilot.json", pilot)
        pilot
        """),
        md("""
        ## Fine-tune and evaluate

        `checkpoints/t5_small/seed_42/` contains full resumable `last.pt`, the selected
        `best.pt`, step snapshots, configuration, data manifest, and dependency versions.
        After final evaluation, `hf_export/` contains the selected model and tokenizer in
        Hugging Face format. Select checkpoints using internal development, then score
        official validation and measure the same two inference workloads.
        """),
        code("""
        from qa_assignment.workflow import run_experiment, archive_outputs

        RUN_TRAINING = True
        if RUN_TRAINING:
            try:
                summary = run_experiment(bundle, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_ROOT,
                                         device=DEVICE, resume="auto", benchmark_config=BENCHMARK_CONFIG,
                                         repo_root=REPO_ROOT)
                print(summary)
            finally:
                print("Available checkpoints/results archived at:", archive_outputs(OUTPUT_ROOT))
        else:
            print("Training disabled; set RUN_TRAINING=True after inspecting the pilot.")
        """),
    ] + ending_cells()
    return cells


def main():
    directory = ROOT / "notebooks"
    directory.mkdir(exist_ok=True)
    for name, cells in (("01_attention_mamba_kaggle.ipynb", build_attention_mamba()),
                        ("02_t5_transfer_learning_kaggle.ipynb", build_t5())):
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}}
        nbf.validate(notebook)
        nbf.write(notebook, directory / name)
        print((directory / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
