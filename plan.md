# Assignment plan: English → Vietnamese NMT

## Objective

Train and explain increasingly capable encoder–decoder translation models from
scratch on IWSLT 2015. Use a pretrained translation model as the final reference.
The main submission evidence is the scratch models, their learning behavior and
held-out translations. Preserve the prior best T5 QA model as a separate local demo.

## Historical progression

1. **RNN**: tanh recurrence, one 128-unit layer per component, final source state
   initializes the decoder. The recurrent/backprop milestone is 1986; seq2seq NMT
   itself dates to 2014.
2. **LSTM + attention**: two 192-unit LSTM layers per component; a projected dot
   attention reads source outputs at each target position. LSTM is 1997, the
   implemented Luong-style attention is 2015. No input feeding is used.
3. **Transformer**: two 256-unit layers per component, four heads, 1,024-unit feed
   forward layers. Encoder self-attention, causal target self-attention and decoder
   cross-attention use PyTorch SDPA with cached generation. Milestone: 2017.
4. **Pretrained Marian EN→VI**: `Helsinki-NLP/opus-mt-en-vi`, checkpoint trained
   2020-06-17. Keep its original vocabulary and `>>vie<<` language token, then fine-tune.

Capacity increases deliberately; this is a recipe progression, not a matched-size
architecture ablation. Accuracy and speed ordering are hypotheses to measure, not
guaranteed outcomes. The LSTM stage changes both recurrence and source conditioning.

## Data and experiment contracts

- Use the modern Parquet mirror `Angelectronic/IWSLT15_English_Vietnamese` at
  commit `647d179736b0a4f2b62f860ffc962b067505311b`. Raw counts: 133,166 train,
  1,268 test. The mirror lacks a validation split and detailed licensing/provenance.
- Normalize HTML/Unicode/whitespace, preserve Vietnamese accents and case. Remove
  empty or >60-word sentence pairs and duplicate normalized sources. Exclude every
  test source from training. Exact source checks do not eliminate semantic overlap
  or talk-level leakage; talk IDs are unavailable.
- Reserve 5,000 distinct validation sources with split seed 2026. Default pilot:
  25,000 selected training pairs; expand to all eligible data after the pilot works.
- Fit joint 8,000-piece BPE only on selected training sources/targets. Scratch
  PAD=0, EOS=1, UNK=2, BOS=3. Marian uses its original source and target tokenizer.
- Filter using both vocabularies' 96-token caps, preserving identical eligible
  example IDs for all models. Never silently truncate targets or evaluation inputs.
- Freeze splits/token IDs, tokenizer hashes, dataset/model revisions, data fingerprint
  and source commit. Match them in both notebooks and reject incompatible resumption
  or mixed-data comparisons. Changed recipes/data use a new output root.

## Execution and diagnostics

Use `01_nmt_scratch_kaggle.ipynb` for stages 1–3 and
`02_nmt_pretrained_kaggle.ipynb` for stage 4. Python implementations live in
`src/nmt_assignment/`, reusing tested generic utilities in `qa_assignment`.
Clone GitHub on Kaggle, expose `src/` in the already-running kernel, and preserve
Kaggle's CUDA PyTorch. No custom CUDA extensions or Mamba build are required.
The pinned Marian `.bin` weights require PyTorch ≥2.6 with Transformers 5.17.0;
check the runtime before loading. Scratch models also support PyTorch 2.5.

Before full training, probe forward/backward/generation and record a resource
estimate. For scratch models, run a train-only memorization check for at most
1,000 updates, FP32, dropout zero, LR 0.003. Use eight length-spread training
sentences with at least four alphabetic words per side and at most 48 tokens.
Target chrF ≥90. Discard diagnostic weights
and start main training from fresh weights. Missing the target is advisory by default;
nonfinite losses/gradients still stop execution. An opt-in strict gate raises an
explicit error before any main training, rather than omitting a model from the results.
Inspect decoded predictions rather than relying on decreasing loss alone.

## Training and stopping

- Scratch: at most eight epochs, AdamW LR 0.001, weight decay 0.01, 2% warmup,
  effective batch 32, gradient clip 1, FP32 by default.
- Marian: at most three fine-tuning epochs, LR 0.00002, 5% warmup, effective batch 32.
- Normalize accumulated losses by actual non-padding target-token counts, including
  the final incomplete batch. Teacher forcing shifts labels right; generation uses
  model predictions. Source and causal masks must pass regression tests.
- Validate at epoch end on the same fixed 512-example development subset. Set the
  limit to `None` in both notebooks for full development evaluation.
- Choose the highest validation chrF checkpoint; lower development loss breaks
  exact chrF ties. Test data never selects epochs/checkpoints.
- Scratch stopping: three evaluations without >0.2 chrF-point improvement, start
  counting at epoch three. Marian: patience two, start at epoch one. Patience zero
  disables stopping. Fluctuating training losses do not directly trigger it.

## Measurement and interpretation

Report SacreBLEU with 13a tokenization, chrF2, metric signatures, filtered test scope,
loss/token accuracy, EOS/empty rates, decoded examples, parameter counts, elapsed
training time, single-sentence latency, batch-eight throughput and CUDA peak memory.
Greedy decoding is common to every stage; synchronize timing and repeat three times.
Do not compare losses as though different vocabularies defined the same objective.
Plot results chronologically, preserving measured scores even if the progression
is not monotonic. Include failure diagnostics in the report.

Marian has a larger model, external training data and pretrained vocabulary.
Prior exposure to IWSLT is not ruled out. Clearly label its results as a pretrained
reference; do not attribute its gains solely to architecture.

## Artifacts and local use

Use the short Kaggle root `/kaggle/working/nmt/`, with separate
`checkpoints/<stage>/seed_42/`, `results/<stage>/seed_42/` and portable
`models/<stage>/seed_42/`. Save `last.pt`, `best.pt` and up to two periodic snapshots,
including optimizer/scheduler/scaler/RNG/cursor/stopping/data/source contracts.
Restore the complete folder into writable storage. Archive checkpoints/results/
portable exports/prepared data and omit downloaded caches.

Portable exports contain safe tensors, tokenizer and `translation_config.json`;
move a selected model to `models/translator/` for local offline inference through
`nmt_assignment.inference.Translator`. Reject overlong inputs instead of truncating.
Keep the user's `models/t5_small_Qna_SQuAD/` and local QA notebook intact.

## Validation and interview preparation

Verify offline synthetic pipelines, causal/padding invariance, common data scope,
token limits, metric conventions, optimizer-boundary resumption, early stopping,
portable model loading and notebook orchestration. Then run the actual Kaggle pilot
and expand only after decoded validation translations improve.
Be ready to explain source/target tokens, teacher forcing versus autoregressive
decoding, fixed-state compression, LSTM gates, attention masking, pretraining,
data leakage, checkpoint selection and the limits of BLEU/chrF.

References: [Stanford IWSLT data](https://nlp.stanford.edu/projects/nmt/),
[dataset mirror](https://huggingface.co/datasets/Angelectronic/IWSLT15_English_Vietnamese),
[LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf),
[Luong attention](https://arxiv.org/abs/1508.04025),
[Transformer](https://arxiv.org/abs/1706.03762),
[Marian model card](https://huggingface.co/Helsinki-NLP/opus-mt-en-vi),
[SacreBLEU](https://github.com/mjpost/sacrebleu).
