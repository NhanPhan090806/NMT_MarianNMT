# Generative QA assignment: RNN, LSTM, attention, Mamba, and T5

Python files contain data preparation, model definitions, training, checkpointing,
evaluation, and profiling. Two Kaggle training notebooks and one local testing
notebook provide the interfaces:

- `01_attention_mamba_kaggle.ipynb`: all four scratch encoder/decoder variants.
- `02_t5_transfer_learning_kaggle.ipynb`: **RNN → LSTM → attention → T5**, all
  encoder-decoder models, with no Mamba dependency. The existing filename is retained.
- `03_t5_question_answering_local.ipynb`: load an exported T5 locally and ask questions
  about your own passage; no training, dataset download, or Mamba installation.

`plan.md` records the original design and the revised notebook-2 scope.

## Start on Kaggle

1. Push this folder to your GitHub repository. Dataset files, checkpoints, and outputs
   are ignored by Git; do not push them with the source.
2. Upload either generated notebook to Kaggle. Enable Internet and a GPU accelerator.
3. Set `REPO_URL = ""` to your actual repository URL. `REPO_REF` accepts a branch, tag,
   or commit SHA. Use the same commit in both notebooks. If these files live inside
   a larger repository, set `SOURCE_SUBDIR` to this assignment's relative directory.
4. Run clone/setup, data preparation, configuration, pilots, and the learning diagnostic.
5. Review generated diagnostic answers, then run training. Failed scratch diagnostics
   skip that variant by default; the next variant and T5 can still run.
6. Download the retraining artifact archive or retain the Kaggle notebook outputs.

The notebooks use a single GPU (`cuda:0`). They do not implement distributed training.
Both notebooks run variants sequentially so optimizer states from one
model do not occupy GPU memory during another model's inference benchmark.

## Dataset and experiment controls

Both complete official SQuAD 1.1 files are downloaded from Stanford into `data/raw/`.
Preparation reserves training articles for internal development and excludes examples
whose **entire** formatted input exceeds the cap. Targets are never silently truncated.
No gold answer location selects the evaluation context. All available reference answers
are used for normalized EM/F1. Scores describe the length-filtered official validation
subset, not the full SQuAD leaderboard dataset.

Both notebooks default to **all eligible training examples** (`max_train_examples=None`).
For a shorter run, set the same limit, such as 10,000, in both notebooks. The shared
split seed is independent of model training seeds. Frozen token IDs, split statistics,
raw-file hashes, tokenizer signature, retained IDs, and the prepared-data fingerprint
are recorded. Different fingerprints are rejected when plotting comparisons.

The scratch notebook includes `attn_attn`, `mamba_attn`, `attn_mamba`, and `mamba_mamba`.
For the plan's core scope, choose only the first two. Mamba encoders use independent
forward/backward branches; Mamba decoders are causal. **All variants retain decoder
cross-attention**. Shared residual/normalization/feed-forward shells and positional
embeddings surround the sequence mixers. This is a comparison of these declared
hybrid architectures, not an attention-free encoder-decoder or a long-context scaling study.

Default scratch models use width 128 and two layers per component. Increase dimensions
only after the pilot; small models and no pretraining can yield low QA scores. The
enabled tiny-subset overfit diagnostic checks learning separately from kernel compatibility.
One-seed results are exploratory. T5 uses a separately declared fine-tuning recipe and
is identified as a transfer-learning reference.

### Revised notebook 2: RNN → LSTM → attention → T5

Notebook 2 runs `rnn_rnn`, `lstm_lstm`, `attn_attn`, then `t5_small`. RNN and LSTM use
[PyTorch RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html) and
[PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
layers in both encoder and decoder. The final encoder hidden state (and LSTM cell
state) is projected into the decoder's initial state. Packed sequences exclude right
padding from the encoder state. These are plain recurrent baselines without attention.
The attention encoder-decoder reuses the PyTorch SDPA model, with causal decoder
self-attention and masked cross-attention. T5 uses Hugging Face pretrained T5-small.

All four share the same prepared data, tokenizer, greedy generation, development
selection, EM/F1 implementation, and latency/throughput/memory measurement pipeline.
The three scratch models default to width 128, two layers per component, a maximum of
five epochs, 1% warmup, no early stopping, and learning rate 3e-4. T5 defaults to a maximum of five epochs and learning rate 1e-4. Common
FP32 precision and effective batch size 16 are the initial settings. The notebook
provides a separate resource pilot for every model and an enabled scratch overfit
diagnostic. Change the configurations before starting full runs if the pilots show
the combined budget is excessive.

Scratch runs in both notebooks disable early stopping and reduce the 20-epoch
maximum to five epochs. The shorter 1% warmup is about 248 steps on the current full
dataset, rather than the previous 4,966. T5 retains 5% warmup and independent
development-F1 stopping: patience five checks, min_delta 0.1 F1 points, minimum step
5,000. All models evaluate internal development every 1,000 optimizer steps and
at epoch end. Histories now include teacher-forced loss, token accuracy, first-token
EOS rate, and sample generated answers. Highest development F1 selects `best.pt`;
lower development loss breaks exact F1 ties. This avoids retaining the first
checkpoint solely because every F1 is zero, but does not make a zero-F1 model useful.
The existing checkpoint intervals, full optimizer/RNG resumption, final normalized
EM/F1, timing, memory, and parameter metrics remain available.

Start fresh for the updated notebook: leave `RESTORE_FROM=None` and use an empty
output directory. To resume these new runs later, keep the complete checkpoint
folders, identical settings, and the same source commit. The shared trainer's default
keeps early stopping disabled. Notebook 1 applies the same scratch recipe and
diagnostic policy to all four attention/Mamba variants. Its fresh artifacts go to
`/kaggle/working/qa_assignment_retrain/`; old stopped runs cannot resume with the new
source/settings. Attach only new matching checkpoint folders when resuming these runs.

Notebook 2 artifacts default to `/kaggle/working/qa_baselines_retrain/`, with checkpoint subfolders
`rnn_rnn/seed_42/`, `lstm_lstm/seed_42/`, `attn_attn/seed_42/`, and `t5_small/seed_42/`.
Its ordered `comparison.csv` and combined `seq2seq_baselines_comparison.png` include
all four models, while labeling T5 as pretrained and recording epochs/learning rates.
Parameter counts differ; this comparison measures these declared recipes, and does
not isolate architecture from the benefit of pretraining. Notebook 1 retains the
original Mamba experiments.

Push the source changes and upload the updated training notebook to a fresh Kaggle
session. The filled repository URL is preserved. Notebook 2 invokes no Mamba build.
Its archive is `qa_baselines_retrain_artifacts.zip`; notebook 1 uses
`qa_assignment_retrain_artifacts.zip`. Use the same source revision and settings
when restoring matching retraining artifacts.

### Learning diagnostic and optional generalization pilot

`RUN_OVERFIT_DIAGNOSTIC=True` runs scratch attention first, then other scratch
variants. `DiagnosticConfig` defaults to 16 examples from different training articles,
1,500 maximum steps, checks every 100 steps, constant LR 3e-3, zero weight decay,
disabled dropout, and FP32. The tiny fixed set is accumulated in microbatches of four.
There is no warmup or patience stopping; it ends at its budget or generated F1 ≥95.
These easier memorization settings are recorded separately from the main recipe.

Each variant saves `results/diagnostics/<variant>/seed_42/report.json`, all diagnostic
predictions/references, and a plot. Reports include losses, generated answers, token
accuracy, EOS rate, and pre-clipping gradient norms for embeddings, encoder, decoder,
output weights, and cross-attention where present. They use training examples only;
diagnostic weights are discarded. Main training starts from fresh weights.

`REQUIRE_DIAGNOSTIC_PASS=True` skips main training for a failed scratch variant.
Change the diagnostic settings and rerun to investigate. An explicit
`REQUIRE_DIAGNOSTIC_PASS=False` allows training anyway. T5 bypasses this scratch gate.
If all scratch variants fail, their diagnostic artifacts still survive in the archive.
The diagnostic is a learning check, not a promise of good held-out QA scores.

`RUN_GENERALIZATION_PILOT=True` optionally runs a fresh attention model on 2,000
training examples for three epochs with the main recipe and 256 internal-development
examples. Separate reports/checkpoints go under `results/generalization_diagnostic/`.
It does not evaluate official validation or reuse its weights in the main run.

### Ask questions with your saved T5

Open `03_t5_question_answering_local.ipynb` in VS Code or Jupyter and select
`C:/Users/ADMIN/ai_venv/Scripts/python.exe` as its Python kernel. Run from the project
folder; `PROJECT_DIR` can be set explicitly if the notebook server starts elsewhere.
The notebook imports local helpers without cloning or installing packages.

`MODEL_DIR` defaults to `models/t5_small/`, the single retained best T5 model.
It contains the exported weights, tokenizer, and `qa_config.json` with the saved
input/output limits. Relative paths resolve against `PROJECT_DIR`; absolute Windows
paths work too. Loading is offline and works on CUDA or CPU. Historical runs,
duplicate checkpoints, logs, and analysis outputs have been removed. Model files
are ignored by Git; keep this folder locally when pushing the source to GitHub.

Edit `PASSAGE` and `QUESTION`, then rerun the answer cell. A question list and optional
typed input loop reuse the loaded model. The loop defaults off for automated runs.
Answers can be saved to `outputs/t5_qa_test/answers.json` inside the project. Input formatting and
the saved sequence caps match training; long inputs are rejected rather than silently
truncated. This is passage-based QA, not open-domain chat. SQuAD 1.1 did not train the
model to abstain when the passage lacks the requested answer. Generation uses the
standard [Transformers T5 API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/t5).

## Dependencies and Mamba compatibility

`requirements-kaggle.txt` pins Transformers 4.57.1 and the tokenizer dependency.
The notebook installs these before importing project code. It retains Kaggle's
preinstalled CUDA-enabled PyTorch rather than replacing it. The Mamba setup helper
installs **mamba-ssm 2.2.6.post3** and **causal-conv1d 1.5.3.post1** with build isolation
disabled and a bounded compiler job count. These are release/API pins, not a claim
that every Kaggle runtime has a matching prebuilt binary.

Both setup cells explicitly put the cloned `src/` directory on the running kernel's
import path and print the loaded helper location. An editable installation made by a
pip subprocess writes import-path configuration for interpreter startup; it does not
refresh the already-running notebook kernel. This fixes the logged
`ModuleNotFoundError: No module named 'qa_assignment'` without restarting midway through
a Kaggle batch run. See [Python's site documentation](https://docs.python.org/3/library/site.html).

The Mamba notebook requires the real selective-scan CUDA extension, optimized causal
convolution, and Triton recurrent update. It checks module loading and then executes
forward/backward and cached-generation probes on the actual accelerator. No Python
Mamba substitute is used. Notebook 2 does not install Mamba.

If Mamba setup fails:

- Confirm Internet, Linux GPU runtime, `torch.cuda.is_available()`, and the installed
  PyTorch/CUDA/Triton versions printed by setup.
- A missing binary may trigger compilation; inspect the build log and `nvcc --version`.
  The driver-supported CUDA version is different from the PyTorch runtime and toolkit.
- GPU support depends on the installed Triton release. For example, Triton 3.2 lists
  compute capability 7.0+, whereas current Triton documentation lists 8.0+. A P100 or
  an older GPU/runtime combination may fail; use a compatible accelerator/stack and
  keep it common across controlled runs. Do not silently change the measured backend.
- Restart the notebook runtime if package imports happened before setup. Avoid
  repeatedly reinstalling different binaries into a session with loaded extensions.

Default precision is FP32. FP16 uses autocast, FP32 parameters, and gradient scaling;
enable it only if the same precision passes every model's probe. BF16 is restricted
to Ampere-or-newer GPUs and still requires working kernels. NaNs and nonfinite gradients
stop training, preserving the previously saved optimizer boundary for resumption.

References: [Mamba pinned release](https://pypi.org/project/mamba-ssm/2.2.6.post3/),
[causal-conv1d pinned release](https://pypi.org/project/causal-conv1d/1.5.3.post1/),
[Triton 3.2 compatibility](https://github.com/triton-lang/triton/blob/v3.2.0/README.md#compatibility),
[current Triton compatibility](https://github.com/triton-lang/triton#compatibility),
[T5 API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/t5).

## Outputs and checkpoint resumption

```text
/kaggle/working/qa_assignment_retrain/
  data/
    raw/train-v1.1.json
    raw/dev-v1.1.json
    prepared/<data-signature>/
      manifest.json
      train.jsonl
      development.jsonl
      validation.jsonl
      tokenizer/
  checkpoints/
    rnn_rnn/seed_42/  # notebook 2, under qa_baselines_retrain/
    lstm_lstm/seed_42/ # notebook 2, under qa_baselines_retrain/
    attn_attn/seed_42/
    mamba_attn/seed_42/
    attn_mamba/seed_42/
    mamba_mamba/seed_42/
    t5_small/seed_42/
      last.pt
      best.pt
      step_00000250.pt
      run_config.json
      data_manifest.json
      environment.txt
      history.json
      hf_export/  # selected T5 model/tokenizer after final evaluation
  results/
    <variant>/seed_42/
      summary.json
      validation_predictions.json
      benchmark.json
      training_log.json
    comparison.csv
    controlled_comparison.png
    transfer_learning_comparison.png
    seq2seq_baselines_comparison.png # combined notebook-2 comparison
```

Each variant/seed directory has its own `last.pt`, `best.pt`, and up to two retained
step snapshots. Checkpoints are atomically replaced at optimizer-update boundaries
and include model, optimizer, scheduler, scaler, random state, epoch/batch cursor,
development history, and run/data/source contracts. A partial accumulation window
is replayed from the last saved boundary. Resume restores the exact deterministic
training sample order; changing settings or the source commit is rejected.

Kaggle working storage is **not permanent**. To resume a later session, save the
outputs, attach them as a Kaggle input, and set `RESTORE_FROM` to the directory
containing `checkpoints/`, `results/`, and optionally `data/`. The helper copies them
into writable storage before data preparation. If you downloaded the zip, extract
it before uploading/attaching it. Preserve the complete checkpoint folder, including
both `last.pt` and `best.pt`; the latter preserves development-based model selection.
Use a new `OUTPUT_ROOT` for different hyperparameters or a new source revision.
Only load trusted checkpoint files: resumable checkpoints contain Python/NumPy state.

The checkpoint interval is configurable. Work after the last saved boundary can be
lost if a Kaggle session is abruptly terminated. The training cell also archives the
available checkpoint/result files in a `finally` block for ordinary exceptions or
interruptions; this cannot rescue files from a process killed by the platform.

## Measurements

Training normalizes accumulated gradients by all valid target tokens, including the
final partial batch. Best checkpoints use internal-development F1, with development
token loss breaking exact ties. Official validation
is scored after training/selection; the predictions and all references are saved.

Benchmarks warm up, synchronize CUDA timing boundaries, and repeat three times.
Batch-one latency and common-batch throughput are distinct. Actual QA generation stops
at EOS; forced 32-token decoding measures a fixed workload. Generated lengths, encoding
and decoding time, and allocated/reserved peak memory accompany the timings.
Tokenization, input transfers, downloads, model loading, and checkpoint writes are
excluded from model timing. Decoding includes initial cache/cross-attention preparation.
Optimized SDPA selects the supported attention kernel at runtime; Mamba kernel paths
and environment versions are recorded. Speedups are not guaranteed at 512 tokens.

## Local verification

The source tests require the development dependencies but do not download SQuAD or
pretrained weights. They exercise metrics, article splitting, length filtering,
recurrent/attention causality and padding, incremental caches, a randomly initialized tiny T5,
optimizer-boundary resumption, artifacts, and notebook validation. Mamba integration
shell tests use a clearly labeled test-only causal mixer; real Mamba tests run only
on a Linux CUDA environment with installed extensions.

```powershell
& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe' -m pytest -q
& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe' scripts/build_notebooks.py
```

The notebook generator is optional; all three notebooks are provided as plain
`.ipynb` files with no execution outputs. Full SQuAD training and real Mamba kernel
execution still require running the notebooks on Kaggle.

The generator writes notebooks at the repository root. Its one-time notebook-2
migration preserves repository URL/ref and data controls while replacing the old
T5-only experiment cells with the new scope. Its diagnostic upgrade refreshes the
training/diagnostic cells for this five-epoch scratch retraining workflow while
preserving clone and data settings. Notebook 1 has its own migration. Subsequent builds
preserve experiment controls and refresh only the shared installation section.
