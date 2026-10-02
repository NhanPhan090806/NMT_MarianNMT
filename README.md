# Generative QA assignment: attention, Mamba, and T5

Python files contain data preparation, model definitions, training, checkpointing,
evaluation, and profiling. Two Kaggle notebooks provide the interfaces:

- `notebooks/01_attention_mamba_kaggle.ipynb`: all four scratch encoder/decoder variants.
- `notebooks/02_t5_transfer_learning_kaggle.ipynb`: pretrained T5-small fine-tuning.

The original `train_market1501_kaggle.ipynb` remains unchanged as your sample.
`plan.md` describes the experimental design and limitations.

## Start on Kaggle

1. Push this folder to your GitHub repository. Dataset files, checkpoints, and outputs
   are ignored by Git; do not push them with the source.
2. Upload either generated notebook to Kaggle. Enable Internet and a GPU accelerator.
3. Set `REPO_URL = ""` to your actual repository URL. `REPO_REF` accepts a branch, tag,
   or commit SHA. Use the same commit in both notebooks. If these files live inside
   a larger repository, set `SOURCE_SUBDIR` to this assignment's relative directory.
4. Run clone/setup, data preparation, configuration, and the pilot cells.
5. Inspect the pilot results, then run the training cell. Training is enabled by
   default, matching the supplied sample's workflow.
6. Download `qa_assignment_artifacts.zip` or retain the Kaggle notebook outputs.

The notebooks use a single GPU (`cuda:0`). They do not implement distributed training.
The attention/Mamba notebook runs variants sequentially so optimizer states from one
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
optional tiny-subset overfit diagnostic checks learning separately from kernel compatibility.
One-seed results are exploratory. T5 uses a separately declared fine-tuning recipe and
is kept in a separate results group.

## Dependencies and Mamba compatibility

`requirements-kaggle.txt` pins Transformers 4.57.1 and the tokenizer dependency.
The notebook installs these before importing project code. It retains Kaggle's
preinstalled CUDA-enabled PyTorch rather than replacing it. The Mamba setup helper
installs **mamba-ssm 2.2.6.post3** and **causal-conv1d 1.5.3.post1** with build isolation
disabled and a bounded compiler job count. These are release/API pins, not a claim
that every Kaggle runtime has a matching prebuilt binary.

The Mamba notebook requires the real selective-scan CUDA extension, optimized causal
convolution, and Triton recurrent update. It checks module loading and then executes
forward/backward and cached-generation probes on the actual accelerator. No Python
Mamba substitute is used. The T5 notebook does not install Mamba.

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
/kaggle/working/qa_assignment/
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
final partial batch. Best checkpoints use internal-development F1. Official validation
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
attention causality/padding, incremental caches, a randomly initialized tiny T5,
optimizer-boundary resumption, artifacts, and notebook validation. Mamba integration
shell tests use a clearly labeled test-only causal mixer; real Mamba tests run only
on a Linux CUDA environment with installed extensions.

```powershell
& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe' -m pytest -q
& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe' scripts/build_notebooks.py
```

The notebook generator is optional; both notebooks are already committed as plain
`.ipynb` files with no execution outputs. Full SQuAD training and real Mamba kernel
execution still require running the notebooks on Kaggle.
