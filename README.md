# English → Vietnamese neural machine translation

The assignment now follows **RNN → LSTM with attention → Transformer → pretrained
Marian** on IWSLT 2015. Python helpers implement preparation, models, diagnostics,
training, resumable checkpoints, evaluation and local inference. Notebooks are the
interfaces; no Mamba build is needed.

| Stage | Historical milestone | Default capacity | Purpose |
|---|---|---|---|
| `rnn` | RNN/backprop era, 1986 | 128 units, 1 layer per component | Fixed-state recurrent baseline |
| `lstm_attention` | LSTM 1997; dot attention 2015 | 192 units, 2 layers | Gated memory and access to source tokens |
| `transformer` | Transformer 2017 | 256 units, 2 layers, 4 heads | Parallel source modeling and causal target attention |
| `marian_en_vi` | Pretrained checkpoint 2020 | Upstream 512 units, 6 layers, 8 heads | Pretrained English–Vietnamese reference |

The dates refer to components/checkpoint release. Encoder–decoder NMT appeared in
2014, so the RNN entry is not a claim that modern seq2seq translation existed in
1986. Scratch capacity increases deliberately. Scores and speed need not increase
monotonically: these are different declared recipes, not a parameter-matched
architecture ablation. The attentive LSTM uses unidirectional source recurrence and
Luong-style dot attention without input feeding, not an exact paper reproduction.

## Notebooks

- `01_nmt_scratch_kaggle.ipynb`: resource probes, tiny training-set memorization
  diagnostics, then RNN, attentive LSTM and Transformer training in that order.
- `02_nmt_pretrained_kaggle.ipynb`: pretrained Marian resource probe and fine-tuning,
  followed by the same translation metrics and optional four-stage comparison.
- `03_t5_question_answering_local.ipynb`: the existing local QA demonstration. Its
  custom cells and your `models/t5_small_Qna_SQuAD/` model are preserved. It remains passage-based
  English QA and is not part of the NMT comparison.

The two earlier QA/Mamba training notebooks have been replaced. The retained
`qa_assignment` package supplies tested task-independent cached attention, optimizer
resumption and utility code, as well as the saved QA model's inference helpers.
New translation logic lives in `src/nmt_assignment/`.

## Run on Kaggle

1. Push the source to GitHub and upload the desired NMT notebook to Kaggle.
2. Enable Internet and a GPU. Repository URL/ref/subdirectory settings are carried
   over from the previous notebooks; verify them and preferably pin `REPO_REF` to
   the same commit SHA in both notebooks.
3. Run clone/setup. Installation preserves Kaggle's CUDA PyTorch and explicitly
   exposes `src/` in the running kernel. If Transformers was imported before setup,
   restart the session first.
4. Prepare data. Both notebooks download the entire pinned dataset and default to
   a **25,000-pair training pilot**. Set `max_train_examples=None` in BOTH notebooks
   for the full eligible training pool, using a fresh `OUTPUT_ROOT` for changed
   settings. Default validation reserve is 5,000 sources; evaluation during training
   uses the same fixed first 512 retained validation examples.
5. Run resource/learning checks, inspect their decoded outputs, then train. The
   scratch diagnostic memorizes eight training sentences spread across lengths,
   with at least four alphabetic words per side and at most 48 tokens; chrF ≥90 passes.
   Diagnostic weights are discarded. Failed stages are skipped by default. Set
   `REQUIRE_DIAGNOSTIC_PASS=False` explicitly only after investigating a failure.
6. Download `nmt_artifacts.zip` or save Kaggle outputs. `/kaggle/working` is temporary.

The upstream Marian revision contains `pytorch_model.bin`. With pinned Transformers
5.17.0, loading that format requires **PyTorch ≥2.6**. Notebook 2 checks this and
reports an actionable error on older runtimes; setup does not replace Kaggle's GPU
PyTorch. The scratch helpers also work on PyTorch 2.5. Local tests use your existing
`C:/Users/ADMIN/ai_venv/Scripts/python.exe` and instantiate tiny random Marian models
without downloading upstream weights.

Kaggle setup pins Transformers 5.17.0 and Hub >=1.23,<2. The previous Transformers
4.57.1 pin downgraded Hub to 0.36 and conflicted with Kaggle's Gradio/Diffusers.
Restart the session before running setup if those libraries have already been imported.
Existing completed run artifacts still record their original 4.57.1 environment.

For a local audit of extracted Kaggle outputs, run
`python scripts/analyze_nmt_runs.py --samples 256 --device cuda:0` (or `--device cpu`).
It verifies corpus hashes, split separation, saved test metrics, checkpoint/export
weight agreement and actual layer counts, then evaluates the selected best models
on deterministic, length-stratified training and validation samples. It writes
`reports/nmt_run_audit.json` and a comparison plot without training or downloading.
Only use this audit with your own trusted `.pt` checkpoints. The current interpretation
is in `reports/nmt_run_analysis.md`. The supplied `kaggle_runs/` stays local and is
ignored by Git.

To include the scratch results in notebook 2, attach notebook 1's saved artifacts,
extract the archive if necessary, and set `RESTORE_FROM` to its root before setup.
Use an empty destination `OUTPUT_ROOT`; restoration refuses to overwrite existing
folders. Keep the same source commit and data settings. A different data fingerprint
is rejected when collecting results. Separate Kaggle sessions are supported; they
do not share `/kaggle/working` automatically.

## Data, vocabulary and evaluation

The dataset is the Parquet community mirror
[Angelectronic/IWSLT15_English_Vietnamese](https://huggingface.co/datasets/Angelectronic/IWSLT15_English_Vietnamese),
pinned to `647d179736b0a4f2b62f860ffc962b067505311b` (133,166 training / 1,268 test
pairs before filtering). It provides no validation split. The mirror's provenance
and license documentation are sparse. The original benchmark is described by
[Stanford NLP](https://nlp.stanford.edu/projects/nmt/).

Preparation normalizes HTML escapes, Unicode NFC and whitespace while retaining
case and Vietnamese accents. Empty/long sentences and duplicate source sentences
are removed. All test source sentences are excluded from training; distinct
training sources are shuffled deterministically, validation is reserved, and then
the training pilot is selected. Different references for an identical source do
not put that source in multiple splits. These checks cover exact normalized source
overlap, not every semantic paraphrase or shared TED talk; this mirror lacks talk IDs.

Scratch stages share an 8,000-piece SentencePiece BPE vocabulary trained only on
selected training sources/targets, with distinct PAD/EOS/UNK/BOS tokens. Dynamic
padding and masked losses avoid training on padding. Teacher forcing shifts targets
right and starts from BOS; inference feeds back model predictions until EOS/cap.
Packed recurrent encoders exclude padded timesteps; attention masks exclude source
padding and prevent future target access.

Marian retains its own pretrained source/target SentencePiece tokenization and the
required `>>vie<<` target-language prefix. Eligibility is checked with BOTH
tokenizers using 96-token source/target caps, plus a preliminary 60-word cap.
This gives all stages identical retained example IDs. No text is silently truncated.
The manifest reports raw counts, duplicate/length exclusions, selected counts,
tokenizer/file SHA-256 hashes, and the data fingerprint. Cached assets are checked
before reuse. Report scores as **filtered IWSLT test scores**, not full benchmark
leaderboard scores.

Checkpoint selection uses validation **chrF**; lower validation loss breaks exact
chrF ties. Scratch training defaults to eight maximum epochs, LR 0.001, effective
batch 32, and early stopping after three eligible evaluations without improvement
greater than 0.2 chrF points; counting starts at epoch three. Marian fine-tuning
defaults to three epochs, LR 0.00002, effective batch 32, patience two, counting from
epoch one. Evaluations occur at epoch end by default. Patience counts evaluations,
not training-log lines. Set `early_stopping_patience=0` to disable stopping; keep an
explicit small epoch budget.

Every stage records detokenized **SacreBLEU (13a)** and **chrF2**, their reproducibility
signatures, loss/token accuracy, empty/EOS rates, translation examples, parameter
counts, training time, CUDA peak memory, generation latency and throughput. All
comparison generation is greedy. Benchmarks warm up and synchronize CUDA, repeat
three times and separate batch-one latency from batch-eight throughput. They time
model generation, excluding tokenization/transfer. Decode length varies with EOS.
Teacher-forced losses across scratch and Marian vocabularies are not directly
comparable. Marian's external data, larger capacity, pretrained tokenizer and
possible prior IWSLT exposure must be disclosed; it is a reference, while the three
scratch systems are the assignment's independently trained models.

## Checkpoints and portable models

```text
/kaggle/working/nmt/
  data/prepared/<signature>/       # shared splits, tokenizers and manifest
  checkpoints/<stage>/seed_42/
    last.pt                       # optimizer, scheduler, scaler, RNG, cursor, contracts
    best.pt                       # highest validation chrF
    step_*.pt                     # at most two recent step snapshots
    run_config.json
    data_manifest.json
    environment.json
    history.json
  models/<stage>/seed_42/          # portable selected model + tokenizer + settings
  results/
    diagnostics/<stage>/report.json
    pilots/<stage>.json
    <stage>/seed_42/
      summary.json
      test_predictions.json
      training_log.json
      history.json
    comparison.csv
    historical_comparison.png
```

Checkpoints are atomically saved at optimizer boundaries every 250 updates and
around validation. An interruption during an unsaved accumulation window replays
from the saved boundary. Resume restores deterministic sample order and optimizer,
scheduler, scaler, RNG, early stopping and selected-best state; changed source/data/
configuration contracts are rejected. Preserve the full checkpoint folder including
`best.pt`. Only load trusted resumable `.pt` artifacts. Save the archive before the
platform terminates the session; an ordinary interruption triggers notebook archival
but an abrupt process/platform kill cannot.

The archive includes checkpoints, results, portable models and prepared data. It
omits downloaded HF caches. All training/model artifacts are ignored by Git.
Exports use safe tensors and include `translation_config.json`; they can be moved
to short directories such as `models/translator/` independently of their old run.

```python
# Run locally after selecting the ai_venv kernel and exposing the project's src/.
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from nmt_assignment.inference import Translator

translator = Translator.from_export("models/translator")
print(translator.translate("Scientists are studying how the climate is changing."))
```

## Development and regeneration

```powershell
& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe' -X utf8 scripts/build_notebooks.py
& 'C:\Users\ADMIN\ai_venv\Scripts\python.exe' -X utf8 -m pytest
```

The generator preserves tagged repository/data/configuration cells and user-added
cells on subsequent builds, refreshes workflow helpers, clears training notebook
outputs and never edits the local T5 QA notebook. When changing the workflow itself,
update both templates and generated notebooks consistently. Regression tests cover
causality/padding, common split scope, token limits, metrics, optimizer resumption,
exports, stopping and actual notebook orchestration. They do not establish trained
translation quality; the Kaggle pilot and held-out evaluation do that.
