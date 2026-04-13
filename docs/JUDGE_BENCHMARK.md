# Director-AI Local Judge Benchmark

> **Module**: `benchmarks/run_judge_benchmark.py` | **Version**: 3.14.0 | **License**: GNU AGPL v3
>
> © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
> © Code 2020–2026 Miroslav Šotek. All rights reserved.

---

## Overview

The judge benchmark compares two scorer configurations head-to-head on
HaluEval to quantify the value of the local DeBERTa-v3-base judge model:

1. **NLI-only baseline**: FactCG-DeBERTa-v3-Large NLI scorer without
   judge escalation.
2. **NLI + local judge**: the same NLI scorer with a local DeBERTa-v3-base
   binary classifier that re-evaluates borderline cases.

The benchmark answers: **does the local judge improve catch rate and
precision over NLI alone? What is the latency cost?**

Additionally, it measures pure judge inference latency in isolation
(no NLI overhead) to characterise the judge model's computational
footprint.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  run_judge_benchmark.py                       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │            1. Judge Latency Benchmark             │        │
│  │                                                    │        │
│  │  Load judge model → warmup (10 iters) →           │        │
│  │  measure 200 iterations → percentile stats        │        │
│  └──────────────────────────────┬─────────────────────┘        │
│                                 │                              │
│  ┌──────────────────────────────▼─────────────────────┐        │
│  │            2. NLI-Only Baseline                     │        │
│  │                                                    │        │
│  │  run_e2e_benchmark(scorer_backend="deberta")       │        │
│  │  → catch_rate, FPR, precision, F1, latency         │        │
│  └──────────────────────────────┬─────────────────────┘        │
│                                 │                              │
│  ┌──────────────────────────────▼─────────────────────┐        │
│  │            3. NLI + Local Judge                     │        │
│  │                                                    │        │
│  │  run_e2e_benchmark(scorer_backend="hybrid",        │        │
│  │                    llm_judge_provider="local")      │        │
│  │  → catch_rate, FPR, precision, F1, latency         │        │
│  └──────────────────────────────┬─────────────────────┘        │
│                                 │                              │
│  ┌──────────────────────────────▼─────────────────────┐        │
│  │            4. Side-by-Side Comparison               │        │
│  │                                                    │        │
│  │  print_comparison(nli_only, local_judge)            │        │
│  │  → delta table with per-task F1 breakdown          │        │
│  └────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
```

---

## Core Functions

### `run_nli_only(max_samples, nli_torch_dtype=None) -> dict`

Runs the E2E benchmark with NLI-only scoring (no judge escalation).
Delegates to `run_e2e_benchmark()` with `scorer_backend="deberta"`.

Returns a dictionary with all E2EMetrics fields plus:

| Field              | Type   | Description                          |
|--------------------|--------|--------------------------------------|
| `benchmark`        | str    | Always `"E2E-NLI-Only"`             |
| `samples_per_task` | int    | Number of samples per HaluEval task  |
| `elapsed_s`        | float  | Total wall-clock time (seconds)      |
| `hw`               | dict   | GPU info (name, VRAM, torch version) |

Saves results to `benchmarks/results/judge_bench_nli_only_{N}.json`.

### `run_local_judge(max_samples, nli_torch_dtype=None) -> dict`

Runs the E2E benchmark with NLI + local judge (hybrid scorer).

The function:
1. Verifies the judge model exists at `training/output/deberta-v3-base-judge/`
2. Runs `run_e2e_benchmark()` with `scorer_backend="hybrid"`,
   `llm_judge_provider="local"`, `llm_judge_model=<judge_path>`
3. Returns metrics in the same format as `run_nli_only()`

Saves results to `benchmarks/results/judge_bench_local_judge_{N}.json`.

**Error case**: raises `FileNotFoundError` if the judge model directory
does not exist. This typically means `training/train_judge.py` has not
been run yet.

### `run_judge_latency(n_iters=200) -> dict`

Measures pure judge inference latency without NLI overhead.

**Protocol**:
1. Load the judge model and tokeniser from `training/output/deberta-v3-base-judge/`
2. Move model to GPU (if available) and set to eval mode
3. Tokenise a fixed test input (representative borderline case)
4. Warmup: 10 forward passes (not timed)
5. If CUDA: `torch.cuda.synchronize()` before and after each measurement
6. Measure `n_iters` forward passes with `time.perf_counter()`
7. Compute percentile statistics

**Test input** (line 155):
```
NLI divergence: 0.45
Context: The Earth orbits the Sun at an average distance of 150 million km.
Response: The Earth orbits the Sun at roughly 150 million kilometers.
```

This input represents a typical borderline case with a divergence of 0.45
(mid-zone), a factual context, and a paraphrased response.

**CUDA synchronisation**: critical for accurate GPU timing. Without
`torch.cuda.synchronize()`, CUDA kernel launches return immediately
(asynchronous execution), and `perf_counter()` measures only the launch
overhead, not the actual computation time.

**Return values**:

| Field       | Type   | Description                          |
|-------------|--------|--------------------------------------|
| `benchmark` | str    | Always `"Judge-Latency"`             |
| `device`    | str    | "cuda" or "cpu"                      |
| `n_iters`   | int    | Number of iterations measured        |
| `median_ms` | float  | Median inference time                |
| `mean_ms`   | float  | Mean inference time                  |
| `p5_ms`     | float  | 5th percentile (best-case)           |
| `p95_ms`    | float  | 95th percentile (worst-case)         |
| `min_ms`    | float  | Absolute minimum                     |
| `max_ms`    | float  | Absolute maximum                     |
| `hw`        | dict   | GPU info                             |

Saves results to `benchmarks/results/judge_bench_latency.json`.

### `print_comparison(nli_only, local_judge) -> None`

Prints a formatted side-by-side comparison:

```
============================================================================
  Local Judge vs NLI-Only Comparison
============================================================================
  Metric                        NLI-Only   + Local Judge        Delta
  ---------------------------------------------------------------
  Catch rate                       72.3%          78.1%        +5.8%
  False positive rate               8.9%           6.2%        -2.7%
  Precision                        89.0%          92.6%        +3.6%
  F1                               79.8%          84.7%        +4.9%
  Accuracy                         81.7%          85.9%        +4.2%

  Latency avg (ms)                   45            68
  Runtime (s)                       271           389

  Per-task F1:
    qa                             82.4%          87.1%        +4.7%
    summarization                  76.6%          81.3%        +4.7%
    dialogue                       80.2%          85.6%        +5.4%
============================================================================
```

### `_gpu_info() -> dict`

Detects GPU hardware for result provenance:

| Field      | Type   | Example                          |
|------------|--------|----------------------------------|
| `gpu`      | str    | "AMD Radeon RX 6600 XT"          |
| `vram_gb`  | float  | 8.0                              |
| `cuda`     | bool   | True                             |
| `torch`    | str    | "2.5.1+rocm6.2"                  |

Returns `{"gpu": "none", "cuda": False}` if no GPU is available.

### `_save(data, name) -> None`

Writes JSON to `benchmarks/results/{name}` with indented formatting.
Logs the file path and size.

---

## CLI Usage

### Default (500 samples per task)

```bash
python benchmarks/run_judge_benchmark.py
```

### Large-scale benchmark

```bash
python benchmarks/run_judge_benchmark.py --samples 10000
```

### FP16 for VRAM-constrained GPUs

```bash
python benchmarks/run_judge_benchmark.py --fp16 --samples 500
```

### Skip NLI-only baseline (reuse previous results)

```bash
python benchmarks/run_judge_benchmark.py --skip-nli-only --samples 500
```

### Skip latency measurement

```bash
python benchmarks/run_judge_benchmark.py --skip-latency --samples 500
```

### CLI Flags

| Flag              | Default | Description                              |
|-------------------|---------|------------------------------------------|
| `--samples`       | 500     | Max samples per HaluEval task            |
| `--latency-iters` | 200     | Iterations for latency measurement       |
| `--skip-latency`  | off     | Skip pure latency benchmark              |
| `--skip-nli-only` | off     | Skip NLI-only baseline                   |
| `--fp16`          | off     | Use FP16 for NLI model (~50% VRAM saved) |

---

## FP16 Pipeline

When `--fp16` is specified:

1. `main()` converts it to `nli_dtype = "float16"` (line 281)
2. Passed to `run_nli_only(nli_torch_dtype=nli_dtype)` (line 303)
3. Passed to `run_local_judge(nli_torch_dtype=nli_dtype)` (line 312)
4. Each function forwards it to `run_e2e_benchmark(nli_torch_dtype=...)`
5. The E2E benchmark passes it to `CoherenceScorer(nli_torch_dtype=...)`
6. The scorer passes it to `NLIScorer`, which loads the model with
   `torch_dtype=torch.float16`

This reduces the FactCG-DeBERTa-v3-Large model from ~1.2 GB (FP32) to
~0.6 GB (FP16), enabling it to coexist with the judge model on 8 GB GPUs.

---

## Judge Model

The local judge is a DeBERTa-v3-base model (184M parameters) fine-tuned
as a binary classifier (approve/reject). It receives formatted input with
the NLI divergence score prepended:

```
NLI divergence: 0.45
Context: <truncated context>
Response: <truncated response>
```

The judge adds value in the borderline zone (NLI divergence 0.2–0.8) where
the NLI scorer alone is uncertain. By learning patterns in this zone from
446K labelled examples, the judge can make more accurate decisions than
raw thresholding.

**Model path**: `training/output/deberta-v3-base-judge/`

**Input format**: tokenised text, max 384 tokens, truncation enabled.

**Output**: 2-class logits → softmax → approve probability.

---

## Output Files

All results saved to `benchmarks/results/`:

| File                                    | Content                        |
|-----------------------------------------|--------------------------------|
| `judge_bench_nli_only_{N}.json`         | NLI-only metrics               |
| `judge_bench_local_judge_{N}.json`      | NLI + judge metrics            |
| `judge_bench_latency.json`              | Pure judge latency stats       |
| `judge_bench_summary_{N}.json`          | Combined status + HW info      |

The summary file records the status (ok/failed) of each benchmark stage
along with total elapsed time and hardware information.

---

## Expected Results

### Judge Latency (DeBERTa-v3-base, 184M parameters)

| Hardware         | Median   | p95      | Notes                    |
|------------------|----------|----------|--------------------------|
| NVIDIA L40S 48GB | ~1.2 ms  | ~1.5 ms  | UpCloud GPU instance     |
| AMD RX 6600 XT   | ~3.5 ms  | ~5.0 ms  | Consumer, ROCm 6.2      |
| CPU (i7-12700K)  | ~25 ms   | ~35 ms   | No GPU acceleration      |

### E2E Comparison (500 samples/task, FP32)

| Metric         | NLI-only | + Judge | Delta  |
|----------------|----------|---------|--------|
| Catch rate     | ~72%     | ~78%    | +6%    |
| FPR            | ~9%      | ~6%     | -3%    |
| Precision      | ~89%     | ~93%    | +4%    |
| F1             | ~80%     | ~85%    | +5%    |

The judge's primary contribution is **reducing false positives** (correct
outputs wrongly halted) while slightly improving catch rate. This is because
the judge learns to approve borderline cases that the NLI scorer would
reject due to high divergence from stylistic variation, not factual error.

---

## Error Handling

The benchmark wraps each stage in try/except (lines 293, 302, 312):

- If latency benchmark fails (e.g. judge model not found), the error is
  logged and recorded as `{"status": "failed", "error": "..."}`.
- If NLI-only fails, the benchmark continues to local judge.
- If local judge fails, the comparison is skipped.
- The summary file always records the final status.

The function logs `exc_info=True` for full tracebacks.

---

## Hardware Detection

`_gpu_info()` uses PyTorch's CUDA API:

- `torch.cuda.is_available()`: checks if any GPU is visible
- `torch.cuda.get_device_name(0)`: GPU name string
- `torch.cuda.get_device_properties(0).total_memory`: VRAM in bytes
- `torch.__version__`: PyTorch version (includes ROCm/CUDA suffix)

If `torch` is not installed, returns `{"gpu": "unavailable", "cuda": False}`.

---

## UpCloud Deployment

The benchmark was designed to run on UpCloud's GPU instances (NVIDIA L40S
48 GB). The module header includes deployment instructions:

```bash
source /opt/director-bench/bin/activate
cd /opt/director-bench/work/director-ai
python benchmarks/run_judge_benchmark.py --samples 500 \
    2>&1 | tee /tmp/judge_bench.log
```

For the full 10K benchmark (~3 hours on L40S):

```bash
python benchmarks/run_judge_benchmark.py --samples 10000 \
    2>&1 | tee /tmp/judge_bench_10k.log
```

---

## Dependencies

| Package         | Version | Purpose                           |
|-----------------|---------|-----------------------------------|
| `torch`         | ≥2.0    | Model loading, GPU inference      |
| `transformers`  | ≥4.30   | AutoModel, AutoTokenizer          |
| `numpy`         | ≥1.24   | Percentile statistics             |
| `datasets`      | ≥2.14   | HaluEval download (indirect)      |

---

## Testing

Covered by `tests/test_run_judge_benchmark.py` (18 tests):

- Function signature verification (parameters, defaults)
- `_gpu_info()` return schema
- `_save()` file writing and JSON validity
- `print_comparison()` output format
- FP16 flag propagation (`nli_torch_dtype` parameter presence)

Run:

```bash
pytest tests/test_run_judge_benchmark.py -v
```

---

## File Reference

| Item                    | Path                                     |
|-------------------------|------------------------------------------|
| Benchmark module        | `benchmarks/run_judge_benchmark.py`      |
| Judge model             | `training/output/deberta-v3-base-judge/` |
| Results directory       | `benchmarks/results/`                    |
| Tests                   | `tests/test_run_judge_benchmark.py`      |
| Upstream: E2E eval      | `benchmarks/e2e_eval.py`                 |
| Upstream: judge trainer | `training/train_judge.py`                |
| Upstream: NLI scorer    | `src/director_ai/core/scoring/nli.py`    |
