# Text Mining / NLP Assignment — Question Answering with Attention and Mamba

## 1. Project Overview

This project investigates **generative Question Answering (QA)** using **SQuAD 1.1**.

**Planning status (2026-10-02):** technically feasible at small scale, subject to the environment and resource checks in Sections 19 and 22. The local GPU is a GTX 1650 with 4 GB VRAM. The minimum research scope is Attn-Attn versus Mamba-Attn, plus a T5-small transfer-learning reference. Attn-Mamba and Mamba-Mamba extend this to the full architecture comparison after the core pipeline works. This scope is provisional until checked against the lecturer's rubric; the rubric and submission deadline have not been provided.

The main goal is to compare conventional **attention-based encoder-decoder architectures** with **Mamba-based State Space Model (SSM) architectures**, focusing on the trade-off between:

* QA quality
* Training efficiency
* Inference efficiency
* Computational/resource requirements

A pretrained **T5 model** will additionally be fine-tuned on the same task as a practical transfer-learning reference.

The project is intentionally divided into two parts:

1. **Controlled architecture experiment**

   * Attention vs. Mamba
   * Models trained under comparable conditions
2. **Transfer-learning reference**

   * Fine-tuned T5
   * Demonstrates the performance achievable using a pretrained modern encoder-decoder model

---

# 2. Research Question

### Main Research Question

> **Under a fixed data and training budget, how does replacing encoder self-attention with bidirectional Mamba affect generative QA quality and computational efficiency, while retaining decoder cross-attention?**

If the full experiment is completed, also investigate decoder self-attention replacement and the interaction between encoder and decoder replacements.

### Secondary Question

> **What changes in QA quality, latency, and memory usage are observed, and do any variants offer a useful trade-off under the measured conditions?**

The project should **not assume beforehand that Mamba will perform worse than attention**.

Instead, the expected hypothesis is:

> Mamba-based architectures may provide computational or inference-efficiency advantages while potentially sacrificing some QA performance compared with an attention-based baseline.

The experiment will determine whether this hypothesis is supported by the results.

---

# 3. Task Definition

The project will formulate SQuAD as a **generative Question Answering task**.

Given:

```text
Context + Question
        ↓
      Model
        ↓
Generated Answer
```

Example:

```text
Context:
"William Shakespeare wrote Hamlet sometime around 1600."

Question:
"Who wrote Hamlet?"

Target:
"William Shakespeare"
```

Unlike extractive QA, the model will generate the answer as a sequence of tokens.

This formulation is particularly suitable for comparing encoder-decoder architectures.

---

# 4. Dataset

## SQuAD

Use **Stanford Question Answering Dataset version 1.1 (SQuAD 1.1)**. Version 2.0 and its unanswerable questions are outside the core scope. See the [official dataset description](https://rajpurkar.github.io/SQuAD-explorer/).

Each sample contains:

* Context
* Question
* Ground-truth answer
* Answer span information

The original SQuAD task is primarily extractive, but this project will reformulate the task into generative QA.

### Dataset processing

For each example:

```text
Input:
question + context

Target:
answer text
```

Use the provided answer text as the target; retain span offsets for coverage checks. Use one deterministic reference for training and retain all reference answers for evaluation.

### Dataset split

Use the official SQuAD training and validation splits. Reserve approximately 10% of training articles as an internal development split, with a fixed seed. Keep all questions from an article in the same partition.

Use the internal development split for hyperparameters, early stopping, and checkpoint selection. Use official validation only for final reporting. Call these results **official validation results**, rather than claiming access to the hidden official test set.

Do not randomly mix the official validation examples into training.

---

# 5. Main Experimental Constraint

## Fixed Input Length

The main experiment will use **one fixed maximum input sequence length** for all architectures.

Example:

```text
MAX_INPUT_LENGTH = 512
```

The exact value should be selected according to available GPU memory and the chosen tokenizer/model configuration.

The important requirement is:

> **All main experimental models must use the same maximum input length.**

This prevents sequence length from becoming an additional uncontrolled variable.

### Padding / truncation

* Use the frozen `google-t5/t5-small` tokenizer for all models. Its vocabulary is reused, but embeddings in the scratch models are randomly initialized.
* Use the same input format: `question: <question> context: <context>`.
* Initial input cap: 512 tokens, including prefixes and special tokens. If the resource pilot cannot fit, choose 256 before the main runs and reprocess every model consistently. Do not change the cap for individual models.
* Construct inputs without using gold answer offsets to choose the visible context. The core filtering policy below requires no input truncation. If truncation is introduced in an extension, protect the question and truncate only the context, then audit answer coverage.
* For the core experiment, retain examples whose **entire question and context fit the chosen cap**, determined without consulting the answer. This avoids removing answer evidence and avoids gold-guided evaluation cropping. Apply the same rule to every split and model.
* Report original and retained counts, length distributions, and answer-span coverage. Results describe this length-filtered subset and cannot be presented as full SQuAD scores.
* Initial output cap: 64 tokens including the end token. Audit target lengths before freezing it; exclude oversized training targets with counts reported. Never silently truncate a training answer. Keep final validation references intact and report any output-cap limitation.
* Right-pad to the fixed input cap. Mask encoder padding in attention and decoder cross-attention; ignore target padding in the loss. Reverse only valid tokens in the backward Mamba encoder branch, then realign and pad the outputs.
* Sliding windows and full-dataset evaluation are optional extensions; they require a prediction aggregation policy without gold answer locations.

---

# 6. Baseline Architecture — Attention-Attention

The primary baseline will be an encoder-decoder architecture in which both encoder and decoder use attention-based sequence modeling.

```text
                 ┌────────────────────┐
Question +       │ Attention Encoder   │
Context ────────►│                    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Attention Decoder  │
                 └─────────┬──────────┘
                           │
                           ▼
                     Generated Answer
```

This model will be referred to as:

> **Attn-Attn**

The baseline establishes the reference point against which the Mamba variants will be evaluated.

---

# 7. Mamba Experiments

The project will investigate replacing **self-attention sequence mixing** with **Mamba-1**. Every controlled model retains decoder cross-attention to the encoder. Encoder Mamba is bidirectional; decoder Mamba is causal. These labels describe the sequence mixers, not the absence of all attention.

## Experiment A — Mamba Encoder / Attention Decoder

```text
Question + Context
        │
        ▼
   Mamba Encoder
        │
        ▼
 Attention Decoder
        │
        ▼
   Generated Answer
```

Name:

> **Mamba-Attn**

Purpose:

* Determine the effect of replacing the encoder's attention mechanism with Mamba.
* Keep the decoder architecture unchanged.

---

## Experiment B — Attention Encoder / Mamba Decoder

```text
Question + Context
        │
        ▼
 Attention Encoder
        │
        ▼
   Mamba Decoder
        │
        ▼
   Generated Answer
```

Name:

> **Attn-Mamba**

Purpose:

* Determine the effect of replacing the decoder's attention-based sequence modeling with Mamba.
* Keep the encoder architecture unchanged.

---

## Experiment C — Mamba Encoder / Mamba Decoder

```text
Question + Context
        │
        ▼
   Mamba Encoder
        │
        ▼
   Mamba Decoder
        │
        ▼
   Generated Answer
```

Name:

> **Mamba-Mamba**

Purpose:

* Investigate Mamba sequence mixing in both encoder and decoder, with cross-attention retained.
* Determine whether replacing both components produces a useful quality-efficiency trade-off.

---

# 8. Important Architecture Consideration

The Mamba decoder cannot simply be substituted into an encoder-decoder Transformer without considering how encoder information reaches the decoder.

The implementation must explicitly define the mechanism through which the decoder conditions on the encoder representation.

The architecture should therefore be documented clearly, including:

* Encoder output representation
* Decoder conditioning mechanism
* Autoregressive generation mechanism
* Token embedding
* Positional/sequence representation where applicable
* Output vocabulary projection

### Fixed architecture interface

All controlled models expose encoder outputs of shape `(batch, input_length, d_model)` plus a padding mask. All decoders use the same cross-attention module structure to query those outputs.

* Attention encoder: bidirectional self-attention.
* Mamba encoder: independent forward and backward Mamba-1 branches, realigned and merged by a learned projection. Both directions see valid tokens only. Count both branches and the merge projection in parameters, time, and memory.
* Attention decoder: causal self-attention, followed by cross-attention.
* Mamba decoder: causal Mamba-1 mixing, followed by the same cross-attention structure.
* Keep normalization, residual connections, feed-forward layers, embedding tying, and dropout policies common where possible. Define the replacement unit explicitly; do not silently compare a whole Mamba block against only the attention operator.
* Use one shared positional-embedding policy for the scratch models, including Mamba variants, and record its cost. This is an experimental design choice, not a claim that Mamba inherently requires positional embeddings.
* Train with shifted answer tokens and masked token cross-entropy. Generate greedily from the start token until the end token or output cap.
* For timed decoding, retain Transformer self-attention KV caches, Mamba convolution/SSM states, and encoder/cross-attention projections. Check that cached and uncached generation agree before benchmarking.

**Mamba-Mamba is a hybrid with cross-attention, not an attention-free model.** Cross-attention still costs work proportional to input length for each generated token. The experiment cannot claim the entire architecture has the scaling properties of a pure Mamba language model. Designing attention-free encoder-decoder conditioning is future work.

The standard Mamba block and inference-state interface are documented in the [official implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py).

---

# 9. Transfer-Learning Reference — T5

A pretrained **`google-t5/t5-small`** model will be fine-tuned on the same examples, input format, tokenizer, and length caps. Use its conditional-generation interface, rather than an extractive start/end-span head. Larger T5 checkpoints are outside the core scope. See the [official model card](https://huggingface.co/google-t5/t5-small).

Example:

```text
Input:
question: Who wrote Hamlet?
context: William Shakespeare wrote Hamlet...

Target:
William Shakespeare
```

T5 will serve as a **transfer-learning reference model**, rather than being treated as another controlled from-scratch architecture.

This distinction is important because T5 has access to large-scale pretraining that the experimental models do not.

### Purpose

T5 provides:

* A strong practical reference
* A modern pretrained encoder-decoder baseline
* A practical test of transfer learning even if the experimental architectures trained from scratch do not achieve strong absolute QA scores; useful QA performance still needs to be demonstrated

T5 results should therefore be discussed separately from the controlled Attn/Mamba comparison.

---

# 10. Experimental Groups

The target full experiment set will be:

| Model       | Encoder    | Decoder    | Training          |
| ----------- | ---------- | ---------- | ----------------- |
| Attn-Attn   | Attention  | Attention  | From scratch      |
| Mamba-Attn  | Mamba      | Attention  | From scratch      |
| Attn-Mamba  | Attention  | Mamba      | From scratch      |
| Mamba-Mamba | Mamba      | Mamba      | From scratch      |
| T5          | T5 Encoder | T5 Decoder | Transfer learning |

The first four models form the **controlled architecture experiment**, with bidirectional encoders, causal decoders, and cross-attention retained throughout. The core comparison uses the first two; the remaining two are extensions gated by available time and GPU resources (Section 22).

T5 forms the **transfer-learning reference**.

---

# 11. Evaluation Metrics

The evaluation will measure both **QA quality** and **computational efficiency**.

## 11.1 QA Quality

### Exact Match (EM)

Measures whether the generated answer exactly matches the reference answer after standard normalization.

Example:

```text
Reference:
William Shakespeare

Prediction:
William Shakespeare
```

→ Exact Match = 1

Whereas:

```text
Reference:
William Shakespeare

Prediction:
Shakespeare
```

→ Exact Match = 0

---

### Token-level F1

Measures token overlap between the generated answer and the reference answer.

F1 captures partial normalized word overlap; it does **not** measure semantic equivalence. Use official SQuAD normalization (lowercase, punctuation and article removal, whitespace normalization), and score against all reference answers using the best score separately for EM and F1. Report dataset-level metrics on a 0-100 scale. See the [official evaluation script reproduced in AllenAI's QA repository](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py).

The primary QA metrics will therefore be:

* Exact Match
* F1

---

# 12. Efficiency Metrics

## Training

Measure:

* Total training time
* Time per epoch
* Number of training steps
* Peak GPU memory usage
* Number of parameters

Where possible, training should be performed under the same hardware and comparable training conditions.

---

## Inference

Measure:

* Total inference time
* Average latency per sample
* Throughput (samples/second)
* Peak GPU memory during inference

The inference benchmark must use the same saved example IDs, hardware, precision policy, batching, and generation settings for every model. Start with 200 fixed examples; expand to 1,000 if timing permits. Use batch size 1 for latency and a common batch size that fits every model for throughput.

Measure two workloads:

1. **Actual QA generation:** greedy generation with the common end-token rule and output cap. Report output-length distribution, empty-answer rate, average and median latency, and samples/second. A model that stops early is not automatically more efficient at equivalent work.
2. **Fixed-workload decoding:** use the same encoder inputs and force 32 generated tokens with early stopping disabled. Report encoding time, decoding time, and generated tokens/second. These forced outputs are for timing only and must not be used for QA scores.

Perform warm-up runs and at least three timed repeats. Synchronize at timing boundaries or use CUDA events; GPU launches are asynchronous. Record whether tokenization and host/device transfer are included, and keep model loading, downloads, and checkpoint writes outside model-inference timing. Reset peak-memory counters per measurement, record allocated and reserved peaks separately, and keep only the measured model on the GPU. Use the [PyTorch CUDA timing guidance](https://docs.pytorch.org/docs/stable/notes/cuda#asynchronous-execution).

Record the actual attention and Mamba kernel paths. Use supported PyTorch optimized attention where available. Do not compare optimized Mamba against an intentionally naive attention implementation or present a slow Mamba fallback as representative of its optimized implementation. Short inputs and short answers may show no Mamba advantage; that is a valid result.

---

## Testing / Evaluation Time

Record:

* Total validation/test evaluation time
* Number of evaluated examples
* Average evaluation time per example

Avoid reporting only raw total time because models must be compared over the same number of examples.

---

# 13. Experimental Fairness

The controlled Attn/Mamba experiments should keep the following as consistent as possible:

* Dataset
* Train/validation split
* Input maximum length
* Output maximum length
* Identical frozen tokenizer and serialized token IDs
* Common effective batch size through gradient accumulation
* Common examples seen and optimizer-update budget
* Optimizer
* Learning-rate policy
* Evaluation dataset
* Generation settings
* Random seeds

If a constraint cannot be kept identical because of architectural differences, document the difference rather than hiding it.

Start with one shared training recipe, an effective batch size of 16, and a common microbatch of 1 or 2 after the resource pilot. Choose the largest common microbatch that fits all included models. Keep sample order and loss normalization over valid target tokens consistent. Use the same optimizer family and schedule, while preserving Mamba-specific initialization and necessary parameter-group exclusions; document these architectural requirements.

Set a common maximum epoch/update budget after measuring pilot throughput. Select each model's best checkpoint using internal-development F1. Report both training consumed by the selected checkpoint and total compute spent on training and selection. If early stopping is used, specify the same policy and report realized examples and steps. Separate quality at an equal training budget from time to a selected checkpoint.

Identical hyperparameters test behavior under a shared recipe, not the best possible performance of each architecture. Any later tuning receives an equal declared trial budget per controlled model. T5 may use a separate fine-tuning recipe and is labeled accordingly.

---

# 14. Model Capacity

Parameter count should be recorded for every model.

The controlled architectures should attempt to maintain reasonably comparable model capacity where practical.

The objective is not necessarily to force identical parameter counts, but to prevent an unfair comparison such as:

```text
Tiny Mamba
vs.
Huge Transformer
```

without acknowledging the difference.

Parameter count should therefore be included in the final results table.

For the 4 GB local pilot, start at `d_model = 128`, two encoder and two decoder layers, four attention heads, and a common feed-forward width of 512. Start Mamba-1 at state size 16, convolution width 4, and expansion factor 2. These are provisional small configurations, not verified memory-fit claims.

Keep common widths and depth for the primary replacement experiment. Report total trainable parameters **and parameters excluding token embeddings/output projections**, since a shared vocabulary can hide substantial differences in mixer capacity. Aim for total counts within approximately 20%, but report exact differences and do not silently alter depth just to meet a tolerance. A separately labeled capacity-matched comparison is optional. Train one model at a time.

---

# 15. Main Comparison

The primary comparison will examine:

### Quality

```text
Exact Match
F1
```

versus:

### Efficiency

```text
Training time
Inference latency
Throughput
GPU memory
Parameter count
```

The project will investigate whether Mamba provides measurable efficiency advantages and how those advantages relate to QA performance.

---

# 16. Quality-Efficiency Trade-off Analysis

The most important analysis will not simply ask:

> "Which model has the highest F1?"

Instead, investigate:

> **How much QA performance is gained or lost for a given computational cost?**

For example:

```text
Model A:
F1 = 85
Inference = 100 ms/sample

Model B:
F1 = 83
Inference = 60 ms/sample
```

The analysis would ask whether the reduction in F1 is accompanied by a sufficiently large efficiency improvement to make the architecture attractive under certain deployment constraints.

A useful visualization is a **quality-efficiency/Pareto plot**, for example:

```text
F1
↑
│          ● Attn-Attn
│
│      ● Mamba-Attn
│
│                ● Mamba-Mamba
│
└────────────────────────────→
       Inference efficiency
```

The actual conclusions must be based on measured results rather than assuming beforehand that Mamba is superior or inferior.

---

# 17. T5 Reference Analysis

T5 should be discussed separately because it uses transfer learning.

Compare:

* EM
* F1
* Inference time
* Parameter count
* GPU memory
* Training/fine-tuning time

The purpose is to answer a practical question:

> **How does a pretrained modern encoder-decoder model compare with the architectures developed from scratch in this project?**

T5 is not required to "beat" every model in every metric.

Its role is to provide a strong practical reference and demonstrate the benefit of transfer learning.

---

# 18. Optional Experiment — Sequence Length

Changing sequence length is **not part of the main experiment**.

If time and computational resources permit, an additional experiment may investigate different maximum input lengths.

For example:

```text
Main experiment:
512 tokens
```

Optional:

```text
256
512
1024
2048
```

This experiment would investigate whether the quality-efficiency relationship changes as the input sequence becomes longer.

This should be treated as a **future/optional extension**, not a requirement for project completion.

---

# 19. Environment Compatibility and Reproducibility

## 19.1 Observed local environment

Read-only checks on 2026-10-02 found:

| Component | Observed state |
| --- | --- |
| Host | Windows |
| GPU | NVIDIA GeForce GTX 1650, 4,096 MiB VRAM |
| NVIDIA driver | 591.59 |
| Existing Python | 3.12.0 at `C:\Users\ADMIN\ai_venv\Scripts\python.exe` |
| PyTorch | 2.5.1+cu121; CUDA available; GPU recognized |
| Transformers | 5.17.0; top-level import succeeded |
| Datasets | 5.0.1 installed; dataset loading not checked |
| Mamba packages | `mamba-ssm`, `causal-conv1d`, and `triton` not installed in this Windows venv |
| WSL | Ubuntu-22.04 registered as WSL2; stopped when inspected |

Windows CUDA availability confirms the existing PyTorch installation can detect the GPU. It does not establish Mamba compatibility, T5 forward/backward compatibility, Linux CUDA availability, or model memory fit. No packages were installed and no WSL training environment was inspected during this plan revision.

## 19.2 Execution environments

* Use the existing Windows venv for local Python work and data/metric checks. Do not upgrade or replace this shared environment as an incidental part of the assignment.
* The candidate local Mamba environment is Ubuntu 22.04 under WSL2. The official Mamba repository lists Linux and additional CUDA requirements for its GPU extension. A Windows venv cannot be reused as a Linux venv. A separate Linux interpreter/environment is a future setup requirement, to be agreed before executing Python outside the user-specified Windows venv.
* Treat GTX 1650 Mamba-kernel compatibility as **unverified** until forward, backward, and cached generation execute successfully on that GPU. Do not assume a prebuilt wheel or its GPU architecture coverage is suitable.
* If local compatibility or throughput fails, use one Linux NVIDIA GPU environment for the entire controlled comparison. Target at least 12-16 GB VRAM as a planning preference, not a guaranteed requirement or memory-fit claim. Availability, cost, and any purchase are unresolved; no cloud resource is provisioned by this plan.
* If no compatible GPU environment is available, an educational PyTorch reference implementation can support correctness demonstrations, but cannot answer the optimized-Mamba efficiency question. Report that reduced outcome honestly rather than substituting its timing into the main results.

WSL GPU support uses the Windows NVIDIA driver. Future setup must follow NVIDIA's WSL instructions and avoid installing a separate Linux display driver. Consult [NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## 19.3 Dependency and precision policy

Use one verified stack for all controlled experiments. Python 3.11 is a candidate for a new Linux environment, not a mandatory downgrade of the existing Python 3.12 environment. Choose PyTorch, its CUDA runtime, Mamba-1, causal-conv1d, Transformers, and any required Triton versions as a compatible set; freeze exact versions and source revisions only after the compatibility checks. Do not invent an untested version matrix or install every latest release independently.

Distinguish the CUDA version supported by the driver, the runtime bundled with PyTorch, and the local CUDA toolkit/compiler used to build extensions. Building a missing wheel may require a matching toolkit and compiler. Some current Mamba installation modes omit the selective-scan CUDA extension; package import alone is insufficient. Follow the installation instructions for the selected pinned release and verify the kernels actually used. See the [official Mamba installation and precision notes](https://github.com/state-spaces/mamba#installation).

Use FP32 for initial correctness checks on the GTX 1650. Evaluate FP16 autocast with FP32 master parameters and gradient scaling only after stable forward/backward checks. Do not select BF16 solely from a capability flag; native acceleration and every required kernel must be verified. Apply the same verified precision policy to all controlled models and disclose any T5 difference. Preserve Mamba's specialized initialization and sensitive recurrent parameters.

## 19.4 Compatibility checks before main training

These are future checks, not checks already passed:

1. Confirm GPU visibility from the selected Linux environment and record GPU architecture, available memory, driver, runtime, and toolkit versions.
2. Run tiny Mamba-1 forward/backward operations with finite outputs, losses, and gradients; verify the optimized selective-scan and convolution paths intended for measurement.
3. Verify the forward/backward encoder branches align valid tokens correctly and padding does not contaminate encoder outputs or cross-attention.
4. Verify causal decoder training, shifted labels, end-token handling, and cached versus uncached generation.
5. Load T5-small and its tokenizer; run a training step and generation. A top-level Transformers import is not a substitute for this check.
6. Overfit a tiny fixed QA subset to demonstrate the pipeline learns; keep this check separate from final validation.
7. Measure peak memory and steady-state step/evaluation time at the proposed input/output caps for every model included in the chosen scope.

If any check fails, resolve it or revise the scope before long runs. Set aside a bounded setup period (initially one focused work session); avoid spending the assignment schedule on unsupported kernel builds.

## 19.5 Reproducibility records

Record:

* Random seeds
* Hardware
* GPU model
* CPU
* RAM
* Python version
* PyTorch version
* Relevant library versions
* Dataset version/source
* Model configuration
* Hyperparameters

For the controlled experiments, use multiple random seeds if computational resources permit.

If multiple seeds are used, report:

```text
mean ± standard deviation
```

rather than relying on one lucky run.

---

# 20. Expected Deliverables

The project should produce the following for the completed scope in Section 22:

### Code

```text
data/
models/
training/
evaluation/
configs/
notebooks/
results/
```

### Experiments

* Attn-Attn
* Mamba-Attn
* T5-small fine-tuning
* Attn-Mamba and Mamba-Mamba if the full comparison is completed

### Results

A final table similar to:

Include only completed models or mark uncompleted rows explicitly. Add training steps/examples seen, parameter counts excluding embeddings, generated answer length, precision, and kernel backend. Separate training and inference peak memory, and present fixed-workload decoding timing separately from actual QA latency. Identify the retained dataset size in the caption. T5 is a separate reference group.

| Model       | EM | F1 | Train Time | Inference ms/sample | Throughput | GPU Memory | Params |
| ----------- | -: | -: | ---------: | ------------------: | ---------: | ---------: | -----: |
| Attn-Attn   |    |    |            |                     |            |            |        |
| Mamba-Attn  |    |    |            |                     |            |            |        |
| Attn-Mamba  |    |    |            |                     |            |            |        |
| Mamba-Mamba |    |    |            |                     |            |            |        |
| T5          |    |    |            |                     |            |            |        |

### Visualizations

For the completed models, combine plots into panels where helpful:

1. F1 comparison
2. Exact Match comparison
3. Training time comparison
4. Inference latency comparison
5. GPU memory comparison
6. Quality vs. efficiency/Pareto plot

---

# 21. Recommended Report Structure

Use the structure below for the full comparison. For the core scope, omit results subsections for unimplemented variants and narrow the discussion to encoder replacement. Document environment compatibility checks and resource-budget decisions in Experimental Setup.

```text
1. Introduction
   1.1 Background
   1.2 Problem Statement
   1.3 Research Question
   1.4 Contributions

2. Background
   2.1 Sequence Models
   2.2 Attention
   2.3 Encoder-Decoder Architecture
   2.4 Transformer
   2.5 State Space Models
   2.6 Mamba
   2.7 Transfer Learning

3. Dataset and Task
   3.1 SQuAD
   3.2 Generative QA Formulation
   3.3 Preprocessing

4. Methodology
   4.1 Attn-Attn Baseline
   4.2 Mamba-Attn
   4.3 Attn-Mamba
   4.4 Mamba-Mamba
   4.5 T5 Transfer Learning

5. Experimental Setup
   5.1 Hardware
   5.2 Hyperparameters
   5.3 Fixed Sequence Length
   5.4 Evaluation Metrics

6. Results
   6.1 QA Performance
   6.2 Training Efficiency
   6.3 Inference Efficiency
   6.4 Resource Usage
   6.5 T5 Reference

7. Quality-Efficiency Trade-off
   7.1 Performance vs. Speed
   7.2 Performance vs. Memory
   7.3 Pareto Analysis

8. Discussion
   8.1 Effect of Replacing the Encoder
   8.2 Effect of Replacing the Decoder
   8.3 Mamba in Both Components with Cross-Attention
   8.4 Transfer Learning
   8.5 Limitations

9. Conclusion

10. Future Work
```

---

# 22. Scope Control

The project should prioritize completing the following in order:

### Tier 1 — Core research scope

1. Verify assignment rubric, deadline, training environment, and resource budget.
2. Freeze SQuAD 1.1 partitions, tokenizer, retained example IDs, and length caps.
3. Establish evaluation correctness and a tiny-subset learning check.
4. Train and evaluate Attn-Attn.
5. Train and evaluate bidirectional Mamba-Attn with the same decoder interface.
6. Fine-tune T5-small on the same data as a separate transfer-learning reference.
7. Measure quality, training cost, inference cost, and peak GPU memory.
8. Produce the final comparison, quality-versus-latency plot, and limitations.

This core supports conclusions about **encoder replacement only**. It cannot establish the effect of replacing the decoder or both components. If the rubric requires every original variant, Tier 2 becomes required and the budget must cover it; the revised plan does not establish rubric compliance.

### Resource and schedule decision

For local resource-limited pilots, use up to 10,000 deterministic training examples after filtering and partitioning. The implemented Kaggle notebooks default to all eligible training examples (`max_train_examples=None`) and download both official files in full. Reduce the same training-example limit in both notebooks if the measured Kaggle budget requires it. Select the same IDs for every scratch model and T5, independent of training seed. Use the entire eligible internal-development and official-validation subsets for quality assessment.

Measure at least 50 steady-state pilot training steps and a representative generation batch before choosing the run budget. Start with a maximum of 5 epochs and adjust the common budget before main training if projected completion time is excessive. Project training time from observed steps/second, dataset size, microbatch/accumulation, and epoch count; add measured development/final evaluation time, all seeds/trials, setup, and a contingency allowance. Do not promise a runtime from GPU memory capacity alone.

Start with one seed for the core. The three-model core requires three training jobs before pilots or tuning. The full design requires five jobs for one seed, or twelve scratch jobs plus at least one T5 job with three scratch seeds. Single-seed findings are exploratory; repeated inference timing does not measure training variability.

If memory does not fit, first reduce the common microbatch and consider common activation checkpointing. If needed, reduce common scratch-model width/depth or input cap before any main run and repeat the pilots. Reducing dataset size reduces runtime but does not fix per-step memory usage. If T5-small remains infeasible locally, use the compatible GPU environment or report the incomplete reference rather than claiming success.

### Tier 2 — Full comparison and stronger evidence

* Attn-Mamba and Mamba-Mamba, completing the encoder/decoder factorial comparison
* Multiple random seeds, preferably three per controlled model
* Better hyperparameter tuning
* Full eligible training data if the measured budget permits
* Detailed GPU memory profiling

### Tier 3 — Optional

* Different sequence lengths
* Mamba-2 or later Mamba variants, each requiring a fresh compatibility assessment
* Larger model sizes
* Additional datasets
* Additional pretrained models

**Do not sacrifice completion of the core experiment for Tier 3 experiments.**

---

# 23. Success Criteria

The core research scope is complete if it can:

1. Train a working generative QA model on SQuAD.
2. Establish Attn-Attn as a reproducible baseline.
3. Implement and evaluate bidirectional Mamba-Attn with retained cross-attention.
4. Measure both QA quality and computational efficiency.
5. Compare the models under the same main fixed sequence length.
6. Fine-tune T5-small as a transfer-learning reference.
7. Analyze the quality-efficiency trade-off using measured results.
8. Clearly explain the limitations of the experiment.

A low absolute F1 score in the from-scratch models should **not automatically invalidate the experiment**, provided the models are functioning correctly, the evaluation is valid, and the comparison is scientifically controlled.

Support that judgment with falling training loss, a successful tiny-subset overfit check, answer examples, empty-answer statistics, and evidence of context use (for example, a small context-shuffling diagnostic). If both scratch models remain near a trivial baseline or ignore the context, report training/learning limitations and avoid claiming their score difference establishes architecture superiority. T5 success alone does not validate the scratch comparison.

Full-scope completion additionally requires Attn-Mamba and Mamba-Mamba. The report must identify the actual completed scope and any rubric requirements still unmet. A failed Mamba compatibility check yields a partial implementation study, not a completed Mamba efficiency comparison.

The T5 reference provides an additional practical benchmark for assessing how much performance can be obtained through transfer learning.

---

# 24. Final Project Goal

The final project should answer the following:

> **On the declared SQuAD 1.1 subset, hardware, and training budget, what happens to QA quality and efficiency when encoder self-attention is replaced by bidirectional Mamba while decoder cross-attention is retained? If the full design is completed, how do decoder and combined replacements change those results?**

At one fixed input cap, conclusions apply to that measured operating point. Long-context scaling, attention-free conditioning, and best-possible architecture performance remain outside the core claims.

The objective is therefore not simply to find the model with the highest F1.

The objective is to understand the **quality–efficiency trade-off between attention and Mamba**, while using a pretrained T5 model as a practical transfer-learning reference.

---

# 25. Implemented Kaggle Workflow

The chosen execution platform is now **Kaggle**, as requested. The Windows environment in Section 19 remains the local editing/testing environment; WSL setup is not part of the current implementation. Kaggle's notebook interpreter is used for notebook execution. No local environment upgrade or cloud purchase is required by this workflow.

* `01_attention_mamba_kaggle.ipynb` clones the GitHub source and runs all four controlled variants by default. Selecting only Attn-Attn and Mamba-Attn gives the core scope.
* `02_t5_transfer_learning_kaggle.ipynb` now runs RNN, LSTM, attention encoder-decoders, and pretrained T5-small. It does not install Mamba; Section 26 describes the revised scope.
* `REPO_URL` is intentionally an empty string for the user to fill in after pushing. Use the same Git commit and data configuration in both notebooks.
* Python helpers live in `src/qa_assignment/`; long model, data, training, evaluation, and artifact logic stays out of notebook cells.
* Each variant and seed writes separate `checkpoints/<variant>/seed_<seed>/` and `results/<variant>/seed_<seed>/` folders under `/kaggle/working/qa_assignment/`.
* Checkpoints contain model/optimizer/scheduler/scaler/RNG state and an optimizer-boundary training cursor. `last.pt` resumes training; `best.pt` preserves internal-development selection. Saved outputs must be attached/copied into a later Kaggle session to resume across sessions.
* Runtime probes check real Mamba CUDA kernels, finite forward/backward operations, and cached/uncached decoder agreement. Successful notebook generation or local attention/T5 tests do not establish Kaggle Mamba compatibility.
* Both notebooks now enable development-F1 early stopping: evaluate every 1,000 optimizer steps and at epoch end; patience five checks, minimum improvement 0.1 F1 points, failure counting from step 5,000. Scratch models have a maximum of 20 epochs, and T5 has a maximum of five. Each variant/seed retains its own stopping state and highest-F1 checkpoint. Shared pilot configuration must be adjusted before long runs when needed.
* See `README.md` for setup, checkpoint layout, measured timing scope, and compatibility troubleshooting.

---

# 26. Revised Notebook 2 Scope: RNN → LSTM → Attention → T5

After prolonged Mamba extension builds on Kaggle, the requested practical comparison
is now four generative QA encoder-decoder models in notebook 2. Notebook 1 retains
the original Mamba design; Sections 1–24 document that design and are not completion
requirements for this alternative comparison.

1. **RNN-RNN:** built-in PyTorch tanh RNN encoder and decoder, connected through a
   learned projection of the final encoder hidden state; no cross-attention.
2. **LSTM-LSTM:** built-in PyTorch LSTM encoder and decoder, with projected encoder
   hidden/cell states initializing the decoder; no cross-attention.
3. **Attention-Attention:** the existing PyTorch SDPA encoder-decoder, with
   bidirectional encoder self-attention, causal decoder self-attention, and masked
   cross-attention to the encoder output.
4. **T5-small:** Hugging Face pretrained model fine-tuned as the transfer-learning
   reference.

RNN/LSTM inputs use packed sequences so padding does not replace the final encoder
state. All models use the same SQuAD 1.1 retained examples, article-level split,
tokenizer, caps, teacher forcing, target-token loss normalization, greedy generation,
development F1 checkpoint selection, and official-validation EM/F1. The recurrent
decoders cache their hidden/cell states; attention/T5 cache decoder keys/values.

Record training loss/time, validation predictions and EM/F1, generated lengths,
parameters, allocated/reserved peak GPU memory, batch-one latency, common-batch
throughput, and fixed 32-token encoding/decoding workloads with the existing metric
helpers. All four appear in one ordered comparison, with T5 labeled pretrained and
its training recipe recorded. Equal width/depth does not give equal parameter counts.
Differences involving T5 combine architecture, pretraining, and fine-tuning settings;
do not claim this comparison isolates those causes.

Notebook 2 uses Kaggle's installed PyTorch and pinned Transformers/tokenizer packages;
it invokes no Mamba/causal-conv1d compilation. Artifacts live under
`/kaggle/working/qa_baselines/`, with separate per-variant/seed checkpoint and result
folders. Each model runs a resource/compatibility pilot before sequential full
training. Scratch limits are now twenty epochs at 3e-4; T5 has a five-epoch limit at
1e-4. All four models use development-F1 early stopping: evaluate every 1,000 optimizer
steps and at epoch end, with five unsuccessful checks and a 0.1-point minimum F1
improvement. Failure counting begins at step 5,000. Retain the highest-F1 checkpoint
independently of the minimum stopping improvement; never use official validation to
make stopping decisions. Persist each model's patience and stop decision in checkpoints.
Use the measured pilot times to choose a feasible combined budget before long
runs. Full training remains subject to Kaggle session and GPU limits.

The revised question is: **Under the declared shared QA data and measured training
recipes, how do RNN, LSTM, attention encoder-decoders, and pretrained T5 differ in
answer quality, training cost, inference latency/throughput, and memory?**

## Retraining and diagnostic revision (2026-10-03)

The completed runs showed zero QA F1 for all scratch variants, including attention
in both notebooks. The previous 20-epoch schedule stopped after 1.81 epochs and used
approximately one epoch for warmup. The following settings supersede the scratch
schedule above; the architecture definitions and shared full-data contract stay the same.

- Both training notebooks now run enabled memorization diagnostics on 16 training
  examples from different articles, up to 1,500 steps, checking every 100 steps.
  Diagnostic settings are explicitly easier: constant LR 3e-3, FP32, no weight decay,
  no dropout, no warmup, and no patience stopping. Generated F1 ≥95 passes the check.
  Report predictions, token loss/accuracy, immediate EOS, gradient paths, and curves.
  Discard diagnostic weights. Failed variants skip full training unless overridden.
- Scratch retraining uses five epochs, LR 3e-4, 1% warmup, and disabled early stopping.
  T5 retains five epochs, LR 1e-4, 5% warmup, and its previous development-F1 stopping.
  Keep all existing metrics and full resumable checkpoint folders per variant/seed.
- Development evaluation additionally records teacher-forced loss, token accuracy,
  first-token EOS rate, and generated samples. Best development F1 selects a checkpoint;
  lower development token loss breaks exact F1 ties. Official validation never selects it.
- An optional 2,000-example attention pilot uses 256 internal-development examples
  for a separate small generalization check, without official validation or benchmarks.
- Fresh output roots are `qa_assignment_retrain/` and `qa_baselines_retrain/` under
  `/kaggle/working`. Old stopped runs are incompatible with the new settings/source.
- A third notebook, `03_t5_question_answering_local.ipynb`, loads the selected T5
  `hf_export/` locally and answers editable questions about a supplied passage. The
  previous successful T5 export can be used immediately, with no retraining required.
  It runs locally with `C:/Users/ADMIN/ai_venv/Scripts/python.exe`, imports this project's
  helpers directly, and saves answers under `outputs/t5_qa_test/`.

Memorization is a prerequisite for diagnosing the common recipe, not evidence of
held-out QA performance. Reassess scratch learning and generalization before making
architecture claims; original T5 has a large parameter and prior-training advantage.
