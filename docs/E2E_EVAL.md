# Director-AI End-to-End Guardrail Benchmark

> **Module**: `benchmarks/e2e_eval.py` | **Version**: 3.11.1 | **License**: GNU AGPL v3
>
> © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
> © Code 2020–2026 Miroslav Šotek. All rights reserved.

---

## Overview

The end-to-end benchmark evaluates Director-AI as a deployed guardrail —
not individual components in isolation, but the complete scoring pipeline:
NLI scorer + threshold gating + evidence retrieval + soft warning zone +
fallback recovery. This is the stack a real user deploys.

Unlike component-level NLI benchmarks (`aggrefact_eval`, `mnli_eval`) that
measure model accuracy on NLI tasks, this benchmark answers the operational
question: **when a hallucinated response reaches Director-AI, does it get
caught? And when a correct response reaches it, does it pass through?**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       e2e_eval.py                                │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  HaluEval Dataset                        │    │
│  │  ┌────────┐  ┌───────────────┐  ┌──────────────┐        │    │
│  │  │   QA   │  │ Summarisation │  │   Dialogue   │        │    │
│  │  └────┬───┘  └───────┬───────┘  └──────┬───────┘        │    │
│  └───────┼──────────────┼─────────────────┼────────────────┘    │
│          └──────────────┼─────────────────┘                      │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │              Per-sample Evaluation Loop                   │    │
│  │                                                          │    │
│  │  1. Ingest context → VectorGroundTruthStore              │    │
│  │  2. scorer.review(context, response)                     │    │
│  │  3. Record: approved, score, warning, evidence, latency  │    │
│  │  4. Compare against ground-truth is_hallucinated label   │    │
│  └──────────────────────────────────┬───────────────────────┘    │
│                                     ▼                            │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │                  E2EMetrics Aggregation                   │    │
│  │                                                          │    │
│  │  TP/FP/TN/FN → catch_rate, FPR, precision, F1           │    │
│  │  warning_rate, fallback_rate, evidence_coverage          │    │
│  │  avg_latency_ms, p95_latency_ms                          │    │
│  │  per_task breakdown (QA, summarisation, dialogue)        │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Metrics

### Primary Metrics

| Metric               | Formula                    | Interpretation                          |
|----------------------|----------------------------|-----------------------------------------|
| **Catch rate**       | TP / (TP + FN)             | % of hallucinations detected (recall)   |
| **False positive rate** | FP / (FP + TN)          | % of correct outputs wrongly halted     |
| **Precision**        | TP / (TP + FP)             | Of flagged outputs, % actually hallucinated |
| **F1**               | 2 × P × R / (P + R)       | Harmonic mean of precision and recall   |
| **Accuracy**         | (TP + TN) / total          | Overall classification accuracy         |

### Operational Metrics

| Metric                | Definition                                           |
|-----------------------|------------------------------------------------------|
| **Warning rate**      | % of approved outputs in the soft warning zone       |
| **Fallback rate**     | % of halted outputs recovered by fallback mode       |
| **Evidence coverage** | % of rejections that include evidence chunks         |
| **Avg latency**       | Mean per-sample wall-clock time (ms)                 |
| **p95 latency**       | 95th percentile per-sample latency (ms)              |

### Confusion Matrix Semantics

| Prediction \ Ground Truth | Hallucinated         | Correct              |
|---------------------------|----------------------|----------------------|
| **Rejected** (not approved) | True Positive (TP) | False Positive (FP)  |
| **Approved**              | False Negative (FN)  | True Negative (TN)   |

A higher catch rate means fewer hallucinations reach the user. A lower FPR
means fewer correct outputs are unnecessarily blocked.

---

## Data Classes

### `E2ESample`

Represents a single evaluation sample with ground truth and predictions.

| Field             | Type   | Description                              |
|-------------------|--------|------------------------------------------|
| `task`            | str    | HaluEval task (qa, summarization, dialogue) |
| `context`         | str    | Source context (truncated to 200 chars)  |
| `response`        | str    | Model response (truncated to 200 chars)  |
| `is_hallucinated` | bool   | Ground-truth label from HaluEval        |
| `coherence_score`  | float  | Director-AI coherence score (0.0–1.0)  |
| `approved`        | bool   | Whether the guardrail approved the output |
| `warning`         | bool   | Whether a soft warning was raised        |
| `fallback_used`   | bool   | Whether fallback recovery was attempted  |
| `has_evidence`    | bool   | Whether evidence chunks were retrieved   |
| `evidence_chunks` | int    | Number of evidence chunks found          |
| `latency_ms`      | float  | Per-sample processing time (ms)          |

### `E2EMetrics`

Aggregates all samples and computes metrics via properties.

**Constructor parameters**:

| Parameter       | Type           | Default | Description                |
|-----------------|----------------|---------|----------------------------|
| `samples`       | list[E2ESample] | []     | Accumulated samples        |
| `threshold`     | float          | 0.5     | Coherence threshold used   |
| `soft_limit`    | float          | 0.6     | Soft warning zone bound    |
| `fallback_mode` | str or None    | None    | Fallback strategy label    |

**Properties**: `total`, `tp`, `fp`, `tn`, `fn`, `catch_rate`,
`false_positive_rate`, `precision`, `f1`, `accuracy`, `warning_rate`,
`fallback_rate`, `evidence_coverage`, `avg_latency_ms`, `p95_latency_ms`.

**Methods**:

- `per_task() -> dict[str, dict]`: breakdown of TP/FP/TN/FN, catch rate,
  FPR, precision, F1, and avg latency per HaluEval task.
- `to_dict() -> dict`: serialise all metrics to a JSON-safe dictionary.

---

## Core Functions

### `run_e2e_benchmark(...) -> E2EMetrics`

The main benchmark function. Instantiates a `CoherenceScorer` with the
specified configuration and runs it against HaluEval samples.

**Parameters**:

| Parameter            | Type          | Default      | Description                     |
|----------------------|---------------|--------------|---------------------------------|
| `tasks`              | list[str]     | all three    | HaluEval tasks to benchmark     |
| `max_samples_per_task` | int or None | None (all)   | Limit per task for quick runs   |
| `threshold`          | float         | 0.5          | Hard limit for gating           |
| `soft_limit`         | float         | 0.6          | Soft warning zone upper bound   |
| `use_nli`            | bool          | False        | Enable NLI model                |
| `nli_model`          | str or None   | None         | HuggingFace model ID for NLI    |
| `fallback`           | str or None   | None         | "retrieval" or "disclaimer"     |
| `scorer_backend`     | str           | "deberta"    | Backend: deberta/hybrid/onnx/lite |
| `llm_judge_provider` | str or None   | None         | "openai"/"anthropic"/"local"    |
| `llm_judge_model`    | str or None   | None         | Model name for LLM judge        |
| `nli_torch_dtype`    | str or None   | None         | "float16"/"bfloat16" for FP16   |

**Key implementation details**:

1. **Single scorer instance**: the scorer is created once and reused across
   all samples to avoid reloading models per sample (line 253).
2. **Fresh store per sample**: `VectorGroundTruthStore` is replaced per
   sample (line 288) to prevent context leakage between samples.
3. **Context ingestion**: the ground-truth context is ingested into the
   store before scoring (line 290), simulating a RAG pipeline.
4. **Latency measurement**: uses `time.perf_counter()` for sub-millisecond
   precision (line 292).

### `sweep_thresholds(...) -> list[dict]`

Pre-scores all samples once, then sweeps the threshold from 30% to 80%
in 5% increments. Returns a list of metrics at each threshold point,
enabling ROC-like curve plotting.

This is more efficient than running `run_e2e_benchmark()` eleven times,
because the NLI scoring (the expensive part) happens only once.

**Output per threshold**:
```json
{
  "threshold": 0.50,
  "catch_rate": 0.7234,
  "false_positive_rate": 0.0891,
  "precision": 0.8901,
  "f1": 0.7983,
  "tp": 723, "fp": 89, "tn": 911, "fn": 277
}
```

### `run_baseline(...) -> E2EMetrics`

No guardrail — every sample is approved with a coherence score of 1.0.
This measures the raw hallucination prevalence in HaluEval:

- **Catch rate**: 0.0% (nothing is caught)
- **False positive rate**: 0.0% (nothing is halted)

The baseline is used as a comparison reference to quantify the guardrail's
impact.

### `print_comparison(baseline, guarded, label) -> None`

Prints a side-by-side delta table comparing two E2EMetrics objects:

```
========================================================================
  Director-AI: Baseline vs Guarded Comparison
========================================================================
  Metric                     Baseline      Guarded        Delta
  ----------------------------------------------------------
  Catch rate                    0.0%        72.3%       +72.3%
  False positive                0.0%         8.9%        +8.9%
  Precision                     0.0%        89.0%       +89.0%
  F1                            0.0%        79.8%       +79.8%
  Accuracy                     50.0%        81.7%       +31.7%
========================================================================
```

### `print_e2e_results(m, baseline=None) -> None`

Pretty-prints the full metric report including per-task breakdown:

```
========================================================================
  Director-AI End-to-End Guardrail Benchmark
========================================================================
  Samples:           2000
  Threshold:         0.5
  Soft limit:        0.6
  Fallback:          none

  Catch rate:        72.3% (723/1000 hallucinations caught)
  False positive:    8.9% (89/1000 correct halted)
  Precision:         89.0%
  F1:                79.8%
  Accuracy:          81.7%

  Warning rate:      12.4%
  Fallback rate:     0.0%
  Evidence coverage: 94.2%

  Latency avg:       45.3 ms
  Latency p95:       112.7 ms

  Task              N Catch     FPR    Prec      F1      Lat
  ------------------------------------------------------------
  qa              667  75.2%    7.3%  91.2%   82.4%    42.1ms
  summarization   667  68.4%   10.1%  87.1%   76.6%    48.7ms
  dialogue        666  73.1%    9.2%  88.8%   80.2%    45.0ms
========================================================================
```

---

## CLI Usage

### Basic benchmark (heuristic scorer)

```bash
python -m benchmarks.e2e_eval
```

### Quick run with NLI (200 samples per task)

```bash
python -m benchmarks.e2e_eval --nli --max-samples 200
```

### Full benchmark with NLI + FP16

```bash
python -m benchmarks.e2e_eval --nli --scorer-backend deberta
```

### Hybrid scorer (NLI + local judge)

```bash
python -m benchmarks.e2e_eval --nli --scorer-backend hybrid \
    --llm-judge-provider local \
    --llm-judge-model training/output/deberta-v3-base-judge
```

### Threshold sweep

```bash
python -m benchmarks.e2e_eval --nli --sweep-thresholds --max-samples 500
```

### Baseline vs guarded comparison

```bash
python -m benchmarks.e2e_eval --nli --compare --max-samples 500 \
    --output-json results/e2e_comparison.json
```

### CLI Flags

| Flag                    | Default    | Description                           |
|-------------------------|------------|---------------------------------------|
| `--max-samples`         | None (all) | Max samples per task                  |
| `--threshold`           | 0.5        | Coherence threshold                   |
| `--soft-limit`          | 0.6        | Soft warning zone upper bound         |
| `--nli`                 | off        | Enable NLI model                      |
| `--nli-model`           | None       | HuggingFace model ID                  |
| `--fallback`            | None       | "retrieval" or "disclaimer"           |
| `--sweep-thresholds`    | off        | Sweep 30%-80% and print table         |
| `--baseline`            | off        | No guardrail (raw halluc. prevalence) |
| `--compare`             | off        | Side-by-side baseline vs guarded      |
| `--output-json`         | None       | Save comparison JSON                  |
| `--scorer-backend`      | deberta    | deberta/hybrid/onnx/lite              |
| `--llm-judge-provider`  | None       | openai/anthropic/local                |
| `--llm-judge-model`     | None       | Model name for LLM judge              |

---

## FP16 Pipeline Wiring

The `nli_torch_dtype` parameter flows through four layers:

1. **CLI** (`benchmarks/run_judge_benchmark.py`): `--fp16` → `nli_dtype="float16"`
2. **Benchmark** (`e2e_eval.py`): `nli_torch_dtype=nli_dtype`
3. **Scorer** (`CoherenceScorer`): `nli_torch_dtype=nli_torch_dtype`
4. **NLI model** (`NLIScorer`): loads model with `torch_dtype=torch.float16`

FP16 reduces VRAM usage by ~50%, enabling the FactCG-DeBERTa-v3-Large
model (304M parameters, ~1.2 GB FP32) to fit on 8 GB GPUs alongside the
judge model (184M parameters, ~0.7 GB FP32).

---

## Context Leakage Prevention

Each sample gets a fresh `VectorGroundTruthStore` (line 288). Without this,
context from previous samples would accumulate in the store, allowing the
scorer to find spurious evidence matches:

1. Sample A context: "The sky is blue" → ingested
2. Sample B response: "The sky is blue" → matches Sample A context (leak)

By replacing the store per sample, each evaluation is isolated.

---

## Output Files

Results are saved to `benchmarks/results/` via `_common.save_results()`:

| Mode              | Output file                        |
|-------------------|------------------------------------|
| Standard          | `e2e_guardrail.json`               |
| Baseline          | `e2e_baseline.json`                |
| Comparison        | `e2e_comparison.json` (or custom)  |
| Threshold sweep   | `e2e_threshold_sweep.json`         |

---

## Performance

Benchmark runtime depends on the scorer backend and sample count:

| Configuration          | Samples | Runtime    | Hardware          |
|------------------------|---------|------------|-------------------|
| Heuristic (no NLI)     | 2000    | ~30s       | Any CPU           |
| NLI (DeBERTa, FP32)    | 2000    | ~45 min    | RX 6600 XT 8GB    |
| NLI (DeBERTa, FP16)    | 2000    | ~35 min    | RX 6600 XT 8GB    |
| NLI (ONNX)             | 2000    | ~20 min    | CPU (16 threads)  |
| Hybrid (NLI + judge)   | 2000    | ~60 min    | RX 6600 XT 8GB    |

Per-sample latency:
- Heuristic: ~15 ms
- NLI (FP32): ~1.3s
- NLI (FP16): ~1.0s
- Hybrid: ~1.8s

---

## Dependencies

| Package       | Version | Purpose                               |
|---------------|---------|---------------------------------------|
| `numpy`       | ≥1.24   | Percentile calculation, mean          |
| `datasets`    | ≥2.14   | HaluEval download (via halueval_eval) |
| `torch`       | ≥2.0    | NLI model inference (optional)        |
| `transformers`| ≥4.30   | NLI model loading (optional)          |

---

## Testing

Covered by `tests/test_e2e_eval.py` (24 tests):

- `E2EMetrics` property correctness (TP/FP/TN/FN, rates, F1)
- `E2ESample` field validation
- `accuracy` computation with edge cases (empty, all-correct, all-halluc)
- Latency aggregation (avg, p95)
- Per-task breakdown correctness
- `to_dict()` serialisation completeness
- FP16 wiring verification (parameter propagation)

Run:

```bash
pytest tests/test_e2e_eval.py -v
```

---

## File Reference

| Item                    | Path                              |
|-------------------------|-----------------------------------|
| Benchmark module        | `benchmarks/e2e_eval.py`          |
| HaluEval loader         | `benchmarks/halueval_eval.py`     |
| Common utilities        | `benchmarks/_common.py`           |
| Results directory       | `benchmarks/results/`             |
| Tests                   | `tests/test_e2e_eval.py`          |
| Upstream: CoherenceScorer | `src/director_ai/core/scoring/scorer.py` |
| Upstream: NLIScorer     | `src/director_ai/core/scoring/nli.py` |
| Related: judge benchmark | `benchmarks/run_judge_benchmark.py` |
