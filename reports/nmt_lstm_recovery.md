# Missing LSTM in the second NMT run

The new outputs contain completed RNN, Transformer and Marian runs. LSTM's
resource pilot and memorization diagnostic ran, but its main training never started.
There is no LSTM main checkpoint, result summary or model export to recover.

The scratch console log, line 301, records:

> Skipping lstm_attention because its learning diagnostic did not pass.

The old notebook required `overfit_demonstrated=True` for every stage and skipped
stages that failed the short diagnostic. LSTM's report at
`kaggle_runs/rnn_lstm_trsf/results/diagnostics/lstm_attention/report.json` contains:

| Diagnostic result at step 1,000 | Value |
|---|---:|
| Required chrF | 90.00 |
| Actual chrF | 73.98 |
| Teacher-forced cross-entropy | 0.1021 |
| Teacher-forced token accuracy | 98.34% |
| Predictions missing EOS | 25% (two of eight) |

Loss declined throughout the probe, from 3.5578 at step 100 to 0.1021 at step
1,000. This is evidence of learning, but generated translations had not reached
the memorization target. High teacher-forced accuracy does not guarantee correct
free-running generation: one wrong predicted token can change later predictions.
The probe did reveal generation errors; changing the gate does not turn that
result into a pass.

The earlier local probe passed at step 700 on PyTorch 2.5.1 / a GTX 1650; Kaggle
used PyTorch 2.11 / a Tesla T4. The local pass was insufficient justification for
a hard 1,000-step gate on Kaggle. PyTorch documents that reproducibility is not
guaranteed across releases, platforms or CPU/GPU execution, even with fixed seeds:
[official reproducibility notes](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
The exact numerical cause of the different convergence was not isolated.

## Fix

- `REQUIRE_DIAGNOSTIC_PASS=False` is now the default. Missing the short diagnostic's
  target produces an explicit advisory; main training remains enabled.
- Numerical failures remain fatal. Existing nonfinite-loss/gradient checks still
  raise, and the preflight also rejects nonfinite diagnostic metrics.
- Strict gating remains available with `REQUIRE_DIAGNOSTIC_PASS=True`. It now
  raises an explicit error before any selected model's main training starts,
  instead of silently leaving a stage out of the final comparison.
- `TRAIN_STAGES` selects which scratch stages run pilots, diagnostics and training.
  Its default is all three, in the historical order. Set it to
  `("lstm_attention",)` to train only the missing stage.

The diagnostic budget, model architectures, epoch budgets, optimizer settings,
early stopping, dataset and evaluation remain the same. Notebook 2 has no scratch
diagnostic gate and requires no corresponding change. Both updated repository
URLs were preserved. Existing uploaded outputs were left intact.

## Recovery

For the separate temporary interface, use `temp_lstm_only_kaggle.ipynb`. Its model
configuration matches notebook 1, its optimizer/data configuration is frozen from
this run's checkpoint contract, and it verifies the exact prepared-data fingerprint
and original LSTM parameter count before training. Run all cells on Kaggle and
extract its final `lstm_attention_seed_42_merge_ready.zip` into
`kaggle_runs/rnn_lstm_trsf/`. It trains only LSTM and requires no attached old outputs.

The following instructions remain an alternative using the main notebook:

1. Push the updated Python helpers and upload the revised notebook 1 to Kaggle.
2. Start a fresh session. Optionally attach the existing extracted scratch output
   as an input and set `RESTORE_FROM` to its root before setup, with an initially
   empty destination `OUTPUT_ROOT`.
3. After the configuration cell, use `TRAIN_STAGES = ("lstm_attention",)` and
   `REQUIRE_DIAGNOSTIC_PASS = False`. Keep the original data and training recipe.
4. Run diagnostics, training and final comparison. Only LSTM trains. Restored
   RNN/Transformer summaries remain available for the comparison; their old
   checkpoints are not resumed against the new source revision.

This recovers the missing stage. It does not address the previously measured
Transformer generalization gap or increase the training dataset.

## Recovery notebook import failure (2026-10-05)

The additional root log `kaggle_runs/nmt-rnn-lstm-trsf-marian.log` shows that
Kaggle fetched commit `a1465b4d2bbb2a955fce4a498e49f4a8da6ca275` and failed in
configuration, before preparation or training, on the import of
`archive_stage_outputs`. That checkout has the core training/export helpers but
lacks this new merge utility. The main three-stage notebook similarly depended
on the newer `review_learning_diagnostics` utility.

Both notebooks now bundle the respective small utility, generated directly from
its authoritative Python function in `src/nmt_assignment/workflow.py`. They use
the cloned implementation when available and the bundled function when absent,
with an explicit console message. This keeps the interfaces compatible with that
older checkout without replacing model, data, training, evaluation or checkpoint
code. Source revision reporting remains the actual GitHub checkout. The frozen
LSTM/data recipe, fingerprint checks, advisory diagnostic and stage-only merge
archive are retained.

Regression tests remove both helpers from the imported workflow to reproduce the
logged API mismatch. They execute the main training controls and the temporary
LSTM's actual training, export, archive and reload on a small offline corpus, in
addition to checking the current-checkout path. Full Kaggle training was not
rerun locally.

Validation: 33 targeted NMT/notebook tests passed. All 12 workflow functions
present in `a1465b4` also match the local versions structurally; the two missing
utilities are the only added functions in that module.
