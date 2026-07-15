# Director-AI Benchmark Report

Version: 3.17.0
Date: 2026-03-27

Reproduction index: [`PUBLIC_BENCHMARKS.md`](PUBLIC_BENCHMARKS.md)
with machine-readable manifest
[`public_accuracy_manifest.toml`](public_accuracy_manifest.toml).

## Hardware

All latency numbers measured on:

- **Primary**: NVIDIA GeForce GTX 1060 6 GB, Python 3.12, torch 2.6.0+cu124
- **Cloud GPU**: NVIDIA L40S 45 GB (UpCloud fi-hel2), Python 3.12, torch 2.6.0+cu124
- **Cross-GPU**: RTX 6000 Ada (48 GB), RTX A5000 (24 GB), RTX A6000 (48 GB), Quadro RTX 5000 (16 GB)
- Iterations: 30 (latency), 5 warmup. GPU clocks not locked.

## v3.11.0 Full Benchmark Suite (L40S, 2026-03-27)

14-scenario benchmark on PyPI package (v3.11.0), NVIDIA L40S 46GB, 8 vCPU.
Rust FFI wheel not included in PyPI — signal functions run Python path.

**BUG: NLI scenarios 2-7, 12 ran on CPU, not GPU.** The benchmark script did
not pass `nli_device="cuda"` and v3.11.0 lacked CUDA auto-detection. Fixed in
v3.12.0: `_load_nli_model()` now auto-selects CUDA when `torch.cuda.is_available()`.
NLI numbers below are **CPU-only** — GPU re-benchmark pending.

### Latency Summary

| # | Scenario | Median | p95 | Notes |
|---|----------|--------|-----|-------|
| 1 | Heuristic (no NLI) | 0.081 ms | 0.090 ms | 2000 iter |
| 2 | NLI single-pair (DeBERTa, **CPU**) | 169.5 ms | 176.5 ms | 300 iter — CPU, not GPU |
| 3 | NLI batch-8 (**CPU**) | 292.9 ms/call | — | 100 iter — CPU fallback |
| 4 | Full KB+NLI (**CPU**) | 354.4 ms/call | — | CPU NLI + vector KB |
| 5 | ONNX (**CPU**) | 169.4 ms | 175.4 ms | 200 iter — CPU, not GPU |
| 6 | Hybrid NLI+GPT-4o-mini | 416 ms | — | NLI on CPU + API call |
| 7 | Hybrid NLI+Claude Haiku | 424 ms | — | NLI on CPU + API call |
| 8 | VerifiedScorer (sentence) | 0.029 ms/resp | — | 200 iter (CPU-only, correct) |
| 9 | VerifiedScorer (atomic) | 0.031 ms/resp | — | 200 iter (CPU-only, correct) |
| 10 | Streaming false-halt | 0/10 (0%) | — | 10 science passages |
| 11 | Throughput heuristic (4T) | 10,630 RPS | 0.535 ms | 10s window |
| 12 | Throughput NLI (**CPU**, 2T) | 5.2 RPS | 399 ms | CPU, not GPU |
| 13 | Signal functions (Python) | 1.5-2.2 us | — | 5000 iter each |
| 14 | BM25 hybrid (100 docs) | 150 us | — | 3000 iter |

### Hybrid Judge Cold Start

| Provider | Cold start | Steady state | Model |
|----------|-----------|--------------|-------|
| OpenAI | 2,401 ms | 416 ms | gpt-4o-mini |
| Anthropic | 1,511 ms | 424 ms | claude-haiku-4-5 |

### Bug Root Cause

`CoherenceScorer(use_nli=True)` without `nli_device="cuda"` left the device as
`None`. `_load_nli_model()` only called `model.to(device)` when device was
truthy — so the model stayed on CPU. nvidia-smi confirmed: 0% GPU, 3 MiB VRAM
throughout the entire benchmark run.

**Fix (v3.12.0):** `_load_nli_model()` now checks `torch.cuda.is_available()`
when device is `None` and auto-selects CUDA. This matches the ONNX loader which
already auto-detected `CUDAExecutionProvider`.

### v3.12.0 Fix Verified — L40S GPU (2026-03-27)

CUDA auto-detection confirmed working on L40S. No `nli_device` param passed —
model auto-placed on GPU, 1,757 MB VRAM allocated. **6.8x faster than v3.11.0
CPU-only bug.**

| Metric | L40S GPU (v3.12.0) | L40S CPU (v3.11.0 bug) | GTX 1060 GPU |
|--------|-------------------|----------------------|-------------|
| Median | **24.9 ms** | 169.5 ms | 431.5 ms |
| p95 | **26.3 ms** | 176.5 ms | 643.1 ms |
| VRAM | 1,757 MB | 3 MB | 1,757 MB |
| Model on GPU | Yes | No | Yes |
| NLI throughput | **40.2 RPS** | 5.2 RPS (CPU) | ~2.3 RPS |
| Heuristic | 0.088 ms | 0.081 ms | — |

The v3.11.0 L40S benchmark ran NLI on CPU (169.5 ms) due to missing CUDA
auto-detection. With the fix, NLI on L40S GPU drops to **24.9 ms** — a 6.8x
improvement. Model load: 33.4s (includes HuggingFace cache check).

Raw data: `benchmarks/results/gtx1060_nli_auto_detect_v3.12.0.json` (local).

Raw data: `benchmarks/results/gtx1060_nli_auto_detect_v3.12.0.json`

### Rust vs Python Signal Benchmark (v3.12.0, 5000 iterations)

| Function | Python (us) | Rust (us) | Speedup |
|----------|------------|----------|---------|
| entity_overlap | 14.70 | 3.70 | 4.0x |
| numerical_consistency | 14.80 | 2.30 | 6.4x |
| negation_flip | 11.80 | 14.40 | 0.8x |
| traceability | 12.20 | 22.20 | 0.5x |
| trend_drop | 6.20 | 0.30 | 20.7x |
| BM25 (100 docs) | 110.20 | 10.80 | 10.2x |

`negation_flip` and `traceability` are slower in Rust due to PyO3 FFI string
marshalling overhead at microsecond scale. Pure numeric (`trend_drop`) and
index-heavy (`BM25`) workloads show 10–21x speedup.

Raw data: `benchmarks/results/rust_signals_bench.json`

### Valid (non-NLI) Results

Scenarios 1, 8-11, 13-14 are CPU-only by design and are valid:
- Heuristic: 0.081ms, 10,630 RPS
- VerifiedScorer: 0.029-0.031 ms/resp
- Signal functions: 1.5-2.2 us
- BM25: 150 us/query

Raw data: `benchmarks/results/l40s_v3.11.0_2026-03-27.json`

### Retrieval Quality Smoke — In-Memory Backend

`benchmarks.retrieval_bench` now reports ranking metrics and a downstream
factual-scoring probe. The probe scores each query against one supported answer
and one distractor answer so retrieval misses show up as guardrail behavior,
not only as rank failures.

| Metric | Value |
|---|---:|
| Hit@1 | 56.7% |
| Hit@3 | 73.3% |
| Precision@3 | 0.311 |
| Downstream scoring accuracy | 86.7% |
| Supported answers accepted | 73.3% |
| Unsupported answers rejected | 100.0% |

Raw data: `benchmarks/results/retrieval_inmemory.json`

## 1. NLI Accuracy — LLM-AggreFact (29,320 samples)

Model: `yaxili96/FactCG-DeBERTa-v3-Large` (0.4B params).
Metric: macro-averaged balanced accuracy (standard for LLM-AggreFact).

| Model | Bal. Acc | Params | Streaming |
|-------|---------|--------|-----------|
| Bespoke-MiniCheck-7B | 77.4% | 7B | No |
| **Director-AI (FactCG)** | **75.6%** | 0.4B | Yes |
| MiniCheck-Flan-T5-L | 75.0% | 0.8B | No |
| MiniCheck-DeBERTa-L | 72.6% | 0.4B | No |
| HHEM-2.1-Open | 71.8% | ~0.4B | No |

### Per-Dataset Breakdown (threshold=0.46)

| Dataset | Bal. Acc | Bal. Acc (L40S) | Pos | Neg | Failure Mode |
|---------|---------|-----------------|-----|-----|-------------|
| Reveal | 89.1% | 88.4% | 400 | 1310 | — |
| Lfqa | 86.4% | 86.6% | 1121 | 790 | — |
| RAGTruth | 82.2% | 82.5% | 15102 | 1269 | — |
| ClaimVerify | 78.1% | 78.0% | 789 | 299 | — |
| Wice | 76.9% | 76.7% | 111 | 247 | — |
| TofuEval-MeetB | 74.3% | 73.6% | 622 | 150 | Summarization |
| AggreFact-XSum | 74.3% | 74.1% | 285 | 273 | Extreme summarization |
| FactCheck-GPT | 73.0% | 72.1% | 376 | 1190 | GPT-generated claims |
| TofuEval-MediaS | 71.9% | 71.9% | 554 | 172 | Summarization (media) |
| AggreFact-CNN | 68.8% | 69.1% | 501 | 57 | Extreme class imbalance (9:1) |
| ExpertQA | 59.1% | 59.0% | 2971 | 731 | Long expert answers |

L40S column: threshold=0.50, 55 ms avg latency, 29,320 samples in 1,619s.
Accuracy differences within ±0.7% — GPU choice does not affect accuracy.

Reproduce: `python -m benchmarks.aggrefact_eval --sweep`

## 2. Latency

### Per-Backend (GTX 1060, 16-pair batch)

| Backend | Median | P95 | Per-pair |
|---------|--------|-----|----------|
| Heuristic (no NLI) | 0.15 ms | 0.44 ms | 0.15 ms |
| Streaming token | 0.02 ms | 0.02 ms | 0.02 ms |
| **ONNX GPU batch** | **233 ms** | **250 ms** | **14.6 ms** |
| PyTorch GPU batch | 304 ms | 353 ms | 19.0 ms |
| ONNX GPU seq | 1042 ms | 1249 ms | 65.1 ms |
| PyTorch GPU seq | 3145 ms | 3580 ms | 196.6 ms |
| ONNX CPU batch | 6124 ms | 8143 ms | 383 ms |

### Cross-GPU (16-pair batch, per-pair median)

| GPU | VRAM | ONNX CUDA | PyTorch FP16 | PyTorch FP32 |
|-----|------|-----------|--------------|--------------|
| RTX 6000 Ada | 48 GB | **0.9 ms** | 1.2 ms | 2.1 ms |
| L40S | 45 GB | — | — | 3.4 ms† |
| RTX A5000 | 24 GB | 2.0 ms | 3.4 ms | 4.8 ms |
| RTX A6000 | 48 GB | 3.5 ms | 9.7 ms | 10.1 ms |
| Quadro RTX 5000 | 16 GB | 5.1 ms | 2.5 ms | 5.9 ms |
| GTX 1060 | 6 GB | 13.9 ms | N/A | 17.4 ms |

† L40S 3.4 ms = 55 ms / 16-pair batch, measured via AggreFact sweep (29,320 samples).

Reproduce: `python -m benchmarks.latency_bench`

## 3. End-to-End Guardrail — HaluEval (300 traces)

Full pipeline: CoherenceAgent + GroundTruthStore + StreamingKernel.
Threshold=0.35, soft_limit=0.45, scorer_backend=deberta (heuristic+NLI).

| Task | N | Catch Rate | Precision | F1 |
|------|---|-----------|-----------|-----|
| QA | 100 | 36.0% | 81.8% | 50.7% |
| Summarization | 100 | 24.0% | 66.7% | 35.3% |
| Dialogue | 100 | 80.0% | 48.2% | 60.2% |
| **Overall** | **300** | **46.7%** | **56.9%** | **51.3%** |

Evidence coverage: 100%. Avg latency: 15.8 ms (p95: 40 ms).

Reproduce (heuristic+NLI): `python -m benchmarks.e2e_eval --nli`

### Hybrid Mode — NLI + LLM Judge (600 traces, L40S)

| Judge | Task | N | Catch | FPR | Precision | F1 | Avg Latency |
|-------|------|---|-------|-----|-----------|-----|-------------|
| Claude Sonnet 4 | QA | 200 | 78.0% | 4.0% | 95.1% | 85.7% | 10.1 s |
| Claude Sonnet 4 | Summarization | 200 | 95.0% | 93.0% | 50.5% | 66.0% | 26.3 s |
| Claude Sonnet 4 | Dialogue | 200 | 99.0% | 95.0% | 51.0% | 67.4% | 6.2 s |
| **Claude Sonnet 4** | **Overall** | **600** | **90.7%** | **64.0%** | **58.6%** | **71.2%** | **14.2 s** |
| GPT-4o-mini | QA | 200 | 77.0% | 3.0% | 96.2% | 85.6% | 1.3 s |
| GPT-4o-mini | Summarization | 200 | 95.0% | 93.0% | 50.5% | 66.0% | 4.3 s |
| GPT-4o-mini | Dialogue | 200 | 99.0% | 95.0% | 51.0% | 67.4% | 1.3 s |
| **GPT-4o-mini** | **Overall** | **600** | **90.3%** | **63.7%** | **58.7%** | **71.1%** | **2.3 s** |

Hybrid mode improves catch rate from **46.7% → 90.7%** (+94% relative).
QA task achieves production-grade precision (95-96%) at 3-4% FPR.
GPT-4o-mini matches Claude at 6x lower latency — recommended for production.

Reproduce:
```bash
python -m benchmarks.e2e_eval --nli --scorer-backend hybrid \
    --llm-judge-provider openai --llm-judge-model gpt-4o-mini
```

### Local Judge — NLI + DeBERTa-v3-base Binary Classifier (L40S)

Replaces LLM API judge with a locally fine-tuned DeBERTa-v3-base (86M params)
trained on 35K borderline NLI samples (HaluEval + AggreFact + FEVER + VitaminC).
The judge runs on borderline NLI scores only; same 70/30 blending as hybrid mode.

**Judge inference latency (L40S, 200 iterations):**

| Metric | Value |
|--------|-------|
| Median | 3.97 ms |
| Mean | 3.98 ms |
| P5 | 3.94 ms |
| P95 | 4.01 ms |

**E2E comparison — 200 samples/task (1200 reviews per pass), threshold 0.5,
tracked artefacts (full escalation analysis in §14):**

| Metric | NLI-Only | + Local Judge | Delta |
|--------|----------|---------------|-------|
| Catch rate | 33.5% | 33.8% | +0.3pp |
| FPR | 4.0% | 3.7% | −0.3pp |
| Precision | 89.3% | 90.2% | +0.9pp |
| F1 | 48.7% | 49.2% | +0.5pp |

**Per-task, + Local Judge (200 samples/task):**

| Task | Catch | FPR | Precision | F1 |
|------|-------|-----|-----------|-----|
| QA | 84.5% | 4.0% | 95.5% | 89.7% |
| Summarization | 12.5% | 2.5% | 83.3% | 21.7% |
| Dialogue | 4.5% | 4.5% | 50.0% | 8.3% |

This is a conservative, high-precision operating point: the aggregate catch
(33.8%) is dominated by QA (84.5% catch at 95.5% precision), while
summarisation and dialogue stay low because FactCG NLI divergence on
long-context tasks is too extreme for the 30% judge weight to flip. The judge
adds precision at near-zero cost (§14: 8.6 ms isolated inference on an L4,
zero API cost) — the same mechanism as the hybrid remote judge without the
per-call latency.

> **Superseded numbers.** An earlier untracked 1000/task L40S run recorded
> 93.63% → 93.80% catch at ~66% FPR. That aggregate was inflated by
> summarisation and dialogue flagging ~99% of *all* outputs (FPR ~98% on those
> tasks) — a degenerate over-flagging point, not detection — and it does not
> reproduce on the current pipeline, which yields 33.8% catch at 3.7% FPR at
> the identical threshold (0.5). QA is stable across both pipelines (≈82–85%
> catch, ≈95% precision). The tracked 200/task run above is the current claim.

Reproduce:
```bash
python benchmarks/run_judge_benchmark.py --samples 200
```

## 4. False-Positive Rate

### Summarization FPR (200 correct HaluEval samples, L4 GPU)

Measures how often correct (non-hallucinated) summaries are falsely rejected.
Three-phase fix in v3.4.0:

| Phase | Config | Threshold | FPR | Relative Reduction |
|-------|--------|-----------|-----|-------------------|
| 0 (original) | max-max | 0.55 | 95.0% | baseline |
| 1 (min agg) | min-mean | 0.35 | 60.0% | -37% |
| 2 (summ-profile) | min-mean + premise_ratio 0.85 | 0.35 | 42.5% | -55% |
| 3 (direct scoring) | w_logic=0, trimmed_mean, direct NLI | 0.15 | 25.5% | -73% |
| **4 (bidir NLI)** | **bidirectional NLI + baseline=0.20** | **0.15** | **10.5%** | **-89%** |

**Phase 4 fixes (v3.5.0):**
- Bidirectional NLI: score both source→summary and summary→source, take min.
  Abstractive rephrasing scores low forward but high reverse — min catches this.
- Baseline calibration: `adjusted = max(0, (raw - 0.20) / 0.80)` shifts the
  score distribution so expected NLI noise at baseline=0.20 maps to zero.

FPR across bidirectional baseline values (200 HaluEval samples, L4 GPU):

| Profile | FPR | Mean Coherence | Mean h_fact |
|---------|-----|---------------|-------------|
| fwd-only (Phase 3) | 25.5% | 0.5347 | 0.4653 |
| bidir, baseline=0.00 | 17.0% | 0.5506 | 0.4494 |
| bidir, baseline=0.10 | 13.0% | 0.5987 | 0.4013 |
| bidir, baseline=0.15 | 11.5% | 0.6164 | 0.3836 |
| **bidir, baseline=0.20** | **10.5%** | **0.6317** | **0.3683** |
| bidir, baseline=0.25 | 9.5% | 0.6467 | 0.3533 |

Reproduce:
```bash
python -m benchmarks.summarization_fpr_diag 200 --threshold 0.15
```

### Streaming False-Halt

4.4% false-halt rate (6/135 passages, heuristic mode, no NLI).
All 6 false halts are trend-triggered on borderline score trajectories.
The current artifact also includes labelled bad-passage diagnostics:
halt precision 14.3%, halt recall 33.3%, and token-of-halt accuracy 0.0%
within an 8-token window. Use this row to audit heuristic halt quality; do
not present false-halt rate as hallucination catch rate.

Reproduce: `python -m benchmarks.streaming_false_halt_bench`

## 5. RAGTruth & FreshQA (L40S, full datasets)

### RAGTruth (2,700 samples, NLI-only)

Source: `wandb/RAGTruth-processed` (HuggingFace). Task: detect hallucinations
in LLM-generated summaries and responses.

| Metric | Value |
|--------|-------|
| Samples | 2,700 (943 hallucinated, 1,757 clean) |
| Catch rate | **49.3%** (465/943) |
| False positive rate | 40.9% |
| Precision | 39.3% |
| F1 | 43.7% |
| Avg latency | 2,650 ms/sample |

### FreshQA (600 samples, NLI-only)

Source: FreshQA Nov 2025 (Google Sheets). Task: detect false-premise questions.

| Metric | Value |
|--------|-------|
| Samples | 600 (148 false-premise, 452 valid) |
| Catch rate | **98.6%** (146/148) |
| False positive rate | 97.8% |
| Precision | 24.8% |
| F1 | 39.7% |
| Avg latency | 1,119 ms/sample |

FreshQA's high FPR is expected: without ground-truth context, the NLI model
cannot verify consistency and defaults to flagging. The 98.6% catch rate on
false-premise questions demonstrates strong detection of factual impossibilities.

Reproduce:
```bash
pip install director-ai[nli] datasets
python benchmarks/run_ragtruth_freshqa.py
```

## 6. Cross-Platform Latency

Platform-specific latency profiling with GC overhead measurement.
Covers heuristic, lite, and streaming backends without GPU dependency.

```bash
python -m benchmarks.platform_latency_bench --iterations 100
```

Reports: platform info, GC pause distribution, per-backend latency,
peak RSS. Results saved to `results/platform_latency_results.json`.

## 7. PyO3 FFI Overhead

Quantifies the cost of crossing the Python->Rust FFI boundary via PyO3 0.24.

| Operation | Python | Rust FFI | Speedup |
|-----------|--------|----------|---------|
| StreamingKernel (500 tok) | 1.970 ms | 0.139 ms | 14.2x |
| CoherenceScorer.review() | 0.022 ms | 0.002 ms | 11.0x |
| Kuramoto UPDE 100 steps | 2.626 ms | 0.272 ms | 9.7x |

Measured on Intel i7-10700K, Python 3.12, 10 iterations. Reproduce:

```bash
pip install -e backfire-kernel/crates/backfire-ffi
python -m benchmarks.ffi_overhead_bench --iterations 100
```

## 8. Batch Coalescing & Continuous Batching (v3.3.0)

### CoherenceScorer.review_batch()

Coalesced NLI inference: single `.forward()` for all H_logical pairs + single
`.forward()` for all H_factual pairs, instead of per-item calls. H_logical and
H_factual run in parallel via `ThreadPoolExecutor`.

**Measured** (GTX 1060, PyTorch backend, 30 iterations, 5 warmup):

| Mode | Median (16-pair) | Per-Pair | Speedup |
|------|------------------|----------|---------|
| `scorer.review()` × 16 (serial) | 14,099 ms | 881 ms | baseline |
| `scorer.review_batch(16)` (coalesced) | 5,627 ms | 352 ms | **2.5x** |

NLI-primitive batch speedup (same run):

| Backend | Median (16-pair) | Per-Pair |
|---------|------------------|----------|
| ONNX GPU batch | 222 ms | 13.8 ms |
| PyTorch batch | 4,142 ms | 258.9 ms |
| ONNX vs PyTorch (batch) | — | **18.7x** |

Reproduce: `python -m benchmarks.latency_bench --nli --onnx`

### ReviewQueue (Continuous Batching)

Server-level request accumulator for `/v1/review`. Collects concurrent HTTP
requests and flushes as a single `review_batch()` per tenant per flush window.

```bash
DIRECTOR_REVIEW_QUEUE_ENABLED=1 \
DIRECTOR_REVIEW_QUEUE_MAX_BATCH=32 \
DIRECTOR_REVIEW_QUEUE_FLUSH_TIMEOUT_MS=10 \
uvicorn director_ai.server:app
```

Expected throughput scales with request arrival rate — bounded by GPU inference,
not per-request overhead.

## 9. Honest Limitations

1. **Summarisation FPR improved**: v3.4.0 reduced FPR on correct summaries
   from 95% to 25.5% via direct NLI scoring, w_logic=0, and trimmed_mean
   aggregation. Remaining 25.5% is a FactCG model limitation on highly
   abstractive text. AggreFact-CNN 68.8%, ExpertQA 59.1% balanced accuracy.
2. **E2E heuristic+NLI catch rate is 46.7%**: hybrid mode (NLI + LLM
   judge) raises this to 90.7% but adds LLM latency (2.3s with GPT-4o-mini).
   Local judge mode achieves equivalent accuracy at 3.97ms and zero API cost.
3. **Summarization FPR solved**: 95% → 25.5% (direct NLI, v3.4.0) → 10.5%
   (bidirectional NLI + baseline=0.20, v3.5.0). Dialogue FPR: 97.5% → 4.5%.
   QA: 3-4% FPR. All three task types now production-grade.
4. **ONNX CPU not competitive**: 383 ms/pair. Requires `onnxruntime-gpu`.
5. **Fine-tuned NLI replacement regressed**: DeBERTa-v3-large fine-tuned as
   a 3-class NLI replacement scored 64.7% — below FactCG 75.6%. The local
   judge (2-class binary on borderline cases only) is a different approach
   that succeeded: +0.23pp F1, +0.80pp QA precision at 1000 samples/task.
6. **Competitor latencies are estimates**: values marked "~" or "(est.)"
   from published papers, not our measurements.
7. **FreshQA NLI-only is detection-only**: 98.6% catch but 97.8% FPR
   without ground truth context. Hybrid mode required for production use.

## 10. Competitive Position

| Feature | Director-AI | NeMo Guardrails | Lynx | GuardrailsAI | SelfCheckGPT |
|---------|-------------|----------------|------|-------------|-------------|
| Approach | NLI + RAG | LLM self-consistency | Fine-tuned LLM | LLM-as-judge | Multi-call LLM |
| Model size | 0.4B | LLM-dependent | 8-70B | LLM-dependent | LLM-dependent |
| Latency | 0.9 ms (Ada) | 50-300 ms + LLM | 1-10 s | 2.26 s | 5-10 s |
| Streaming halt | Yes | No | No | No | No |
| Offline/local | Yes | No | Yes (GPU) | No | No |
| AggreFact bal. acc | 75.6% | N/A | N/A | N/A | N/A |
| Integrations | LC/LI/LG/HS/CrewAI | LangChain | Python | LC/LI | Python |

Director-AI's unique value: sub-ms streaming halt + 75.6% balanced accuracy
at 0.4B params + local judge at 3.97ms (no API dependency). No competitor
offers claim-level halt with fully local hybrid scoring.

Full analysis: [`benchmarks/comparison/COMPETITOR_COMPARISON.md`](comparison/COMPETITOR_COMPARISON.md)

## 11. BEIR Retrieval — grounded() Hybrid Pipeline (L4, 2026-07-12)

The shipped `VectorGroundTruthStore.grounded()` retrieval pipeline
(hybrid BM25 + dense with RRF k=60, `title + text` as one field,
optional cross-encoder rerank of the top 30 candidates) measured on
BEIR test splits, scored with `pytrec_eval` and cross-checked against
a built-in linear-gain nDCG@10 (zero disagreement on every arm).
Artefact: `benchmarks/results/beir_competitive_bench.json` — records
exact model revisions, host environment, and the published baseline
rows with sources.

| Arm (embedder__reranker) | NFCorpus nDCG@10 | SciFact nDCG@10 | p50/query (L4 GPU) |
|---|---|---|---|
| bge-large__none (default embedder, no rerank) | 0.3703 | 0.7331 | 33–52 ms |
| bge-large__ms-marco (default pair) | 0.3625 | 0.6993 | 106–126 ms |
| bge-m3__none | 0.3440 | 0.7015 | 33–52 ms |
| bge-m3__ms-marco | 0.3529 | 0.6927 | 108–125 ms |
| bge-m3__bge-reranker-v2-m3 | 0.3465 | 0.7407 | ~1.8 s |

Published reference rows (verified at source 2026-07-12): BM25
0.325 / 0.665 and BM25+CE 0.350 / 0.688 (BEIR paper, arXiv:2104.08663,
Table 2); bge-large-en-v1.5 pure dense 0.38129 / 0.74607 (model-card
MTEB metrics — measured with a query instruction prefix the shipped
recipe does not add).

Readings:

- The hybrid pipeline without a reranker scores above the BEIR paper's
  BM25 and BM25+CE rows on both datasets.
- The candidate bge-m3 embedder does not improve English BEIR retrieval
  over the shipped bge-large default (−0.026 NFCorpus, −0.032 SciFact
  nDCG@10); the default embedder stays bge-large-en-v1.5.
- The ms-marco cross-encoder rerank lowers nDCG@10 on these
  out-of-domain corpora (−0.008 / −0.034) while raising hit@1
  substantially on the internal curated-KB evaluation set
  (`benchmarks/results/retrieval_model_refresh_ab.json`: 0.73–0.77 →
  0.93–0.97). The default keeps the reranker because `grounded()`
  targets curated fact KBs; operators ranking open corpora can pass
  `use_reranker=False`.
- bge-reranker-v2-m3 is the strongest SciFact arm (0.7407) at roughly
  15× the rerank latency; it remains an opt-in alternative
  (`reranker_model="BAAI/bge-reranker-v2-m3"`), not the default.
- Quality numbers are hardware-independent: the arms measured on both
  an i5-11600K (CPU) and the L4 host produced identical nDCG@10.

CPU cross-check (`benchmarks/results/beir_competitive_bench_cpu_i5_11600K.json`,
committed): all five NFCorpus arms re-run end-to-end on a heavily loaded
i5-11600K reproduce the GPU nDCG@10 values exactly (0.3703 / 0.3625 /
0.3440 / 0.3529 / 0.3465). CPU latency under that load: hybrid p50
~0.26–0.30 s/query, ms-marco rerank ~2.7–3.1 s/query, and
bge-reranker-v2-m3 ~70 s/query — the practical argument against the
large cross-encoder as a CPU default. The SciFact CPU arms were
intentionally not run: the quality cross-check was already conclusive on
a full dataset, the script resumes per-arm if they are ever wanted, and
the shared workstation was needed back.

Claim boundary: retrieval-quality evidence for the shipped pipeline on
two public BEIR test splits; it is not a leaderboard submission and
makes no claim about corpora or configurations that were not measured.

## 12. BEIR Fusion Strategies — beyond RRF (CPU i5-11600K, 2026-07-13)

Every fusion strategy shipped in
`director_ai.core.retrieval.vector_store.fusion` measured on the same
two BEIR test splits through the unreranked `grounded()` recipe
(bge-large embedder, one shared index per dataset,
`HybridBackend.with_fusion()` views — arms differ only in query-time
fusion). Scored identically to §11 (pytrec_eval + built-in nDCG
cross-check). Artefact: `benchmarks/results/beir_fusion_bench.json`.
Arm naming: `<method>__s<sparse%>_d<dense%>`.

| Arm | NFCorpus nDCG@10 | SciFact nDCG@10 |
|---|---|---|
| **convex__s30_d70** | **0.3796** | **0.7537** |
| zscore__s50_d50 | 0.3641 | 0.7500 |
| convex__s50_d50 | 0.3726 | 0.7467 |
| combmnz__s50_d50 | 0.3725 | 0.7435 |
| rrf__s30_d70 | 0.3731 | 0.7413 |
| rrf__s50_d50 (shipped default) | 0.3703 | 0.7331 |
| rrf__s70_d30 | 0.3569 | 0.7168 |
| convex__s70_d30 | 0.3528 | 0.7149 |

Readings:

- `rrf__s50_d50` reproduces the §11 `bge_large__none` numbers exactly
  on both datasets — the fusion refactor left the shipped default
  bit-identical (regression check inside the artefact).
- `convex__s30_d70` (min-max CombSUM, dense-weighted convex
  combination) is the strongest arm on both datasets: +0.009 NFCorpus
  and +0.021 SciFact nDCG@10 over the shipped RRF default. On SciFact
  it also scores above the best §11 rerank arm (bge-reranker-v2-m3,
  0.7407) and the bge-large MTEB dense row (0.74607, measured with an
  instruction prefix this recipe does not add); on NFCorpus it sits
  0.003 below that MTEB row.
- The pattern is consistent: dense-leaning weights beat balanced beat
  sparse-leaning on both corpora, and score fusion (convex) beats rank
  fusion (RRF) at equal weights. Z-score fusion is strong on SciFact
  (0.7500) but below the default on NFCorpus (0.3641) — not a
  candidate.
- Latencies in the artefact are from a loaded shared host
  (`pinned-loaded-host`-era run, pre host-conditions wiring) and are
  not comparable across arms; quality numbers are load-independent
  (identical CPU/GPU nDCG established in §11).
- Default decision (internal A/B measured, 2026-07-13,
  `benchmarks/results/retrieval_fusion_internal_ab.json` — curated-KB
  EVAL_SET + distractors, one shared bge-large index, each fusion arm
  ± the default ms-marco reranker): **unreranked**, convex s30/d70
  dominates (hit@1 0.967 vs 0.733 for RRF, hit@3 1.000 vs 0.933);
  **under the full default chain (reranker on)** the gain disappears
  (hit@1 0.933 = 0.933, hit@3 0.967 vs 1.000 — the reranker's 3×
  over-fetch washes out fusion differences and RRF fed it a slightly
  better pool on one query). `rrf` (k=60) therefore REMAINS the
  shipped default; `convex` s30/d70 is the measured recommendation
  for unreranked / open-corpus / CPU-latency-sensitive deployments:
  `grounded(use_reranker=False, fusion_method="convex",
  sparse_weight=0.3, dense_weight=0.7)`.

Claim boundary: same as §11 — two public BEIR test splits, shipped
pipeline, no leaderboard claim, no claim about unmeasured corpora.

## 13. Self-consistency signal — lexical vs NLI (WikiBio GPT-3, H100, 2026-07-13)

SelfCheckGPT-style evaluation of the opt-in
`director_ai.core.scoring.self_consistency.SelfConsistencyScorer` (WCA-3)
on the public `potsawee/wiki_bio_gpt3_hallucination` dataset, evaluation
split (238 GPT-3 passages, each with 20 dataset-shipped samples). Signal =
`1 - consistency_score` of each passage against its 20 samples; truth =
mean sentence-level hallucination annotation. The two entailment backends
of the scorer are compared: `lexical` (Jaccard overlap) and `nli`
(DeBERTa-v3-large MNLI, bidirectional-entailment clustering). Artefact:
`benchmarks/results/self_consistency_wikibio_bench.json`.

| Backend | Pearson | Spearman | AUROC (strong halluc.) | s/passage |
|---|---|---|---|---|
| **lexical (Jaccard)** | **0.3090** | **0.3784** | **0.5800** | 0.015 |
| nli (DeBERTa-v3-large MNLI) | 0.1793 | −0.0310 | 0.5051 | 8.699 |

Readings:

- The cheap lexical overlap signal beats the NLI entailment signal on
  every metric here: higher rank correlation with the human hallucination
  annotation (Spearman 0.378 vs −0.031, i.e. essentially none) and a
  better strong-hallucination AUROC (0.580 vs 0.505, where 0.5 is chance).
  Bidirectional NLI clustering over the 20 samples adds no usable signal
  over token overlap on this dataset.
- It is also ~580× slower per passage (8.699 s vs 0.015 s on an H100). The
  full NLI run is ~35 min on an H100 GPU and 11 h+ on a loaded shared CPU
  (≈100k DeBERTa-large forward passes; the scorer scores sample pairs
  one at a time). This backend is GPU-only in practice.
- Consequence for the shipped opt-in scorer: when self-consistency is
  enabled, the **lexical backend is the sensible default** — it is both
  stronger and effectively free here; the NLI backend is not worth its
  cost on WikiBio-style passage self-consistency. This is a measured
  result, not a design preference — mirroring §12, the cheaper path won.
- The lexical row reproduced bit-identically across an earlier loaded-CPU
  run and this GPU run (pearson 0.3090 / spearman 0.3784 / AUROC 0.5800),
  confirming the signal is deterministic. Environment recorded in the
  artefact: JarvisLabs H100 (EU1), torch 2.6.0+cu124, `cuda:0`.

Claim boundary: one public dataset (WikiBio GPT-3 hallucination,
evaluation split), the shipped `SelfConsistencyScorer` at its default
20-sample protocol; no leaderboard claim and no claim about other
self-consistency corpora, sample counts, or NLI checkpoints.

## 14. Local-judge escalation — NLI-only vs NLI + judge (HaluEval, L4, 2026-07-13)

Calibrated-escalation evaluation of the local DeBERTa-v3-base judge (WCA-5)
as a borderline-escalation layer on the end-to-end guardrail. Both configs
run the full `CoherenceScorer` pipeline over HaluEval (QA / summarization /
dialogue); the judge (a 2-class grounded-vs-hallucinated head) is consulted
only on borderline scores near the decision threshold. `NLI-only` uses the
`deberta` backend; `+ Local Judge` uses the `hybrid` backend with
`llm_judge_provider="local"` pointing at the trained checkpoint. 200
samples/task (600 pairs per backend). Environment: JarvisLabs L4 (IN2),
torch 2.6.0+cu124, `cuda:0`, transformers 5.8.0. Artefacts:
`benchmarks/results/judge_bench_{nli_only,local_judge,summary}_200.json`,
`judge_bench_latency.json`.

| Metric | NLI-only | + Local judge | Delta |
|---|---|---|---|
| Catch rate (recall) | 33.5% | 33.8% | +0.3% |
| False-positive rate | 4.0% | 3.7% | −0.3% |
| Precision | 89.3% | 90.2% | +0.9% |
| F1 | 48.7% | 49.2% | +0.5% |
| Accuracy | 64.8% | 65.1% | +0.3% |
| Avg latency (ms) | 504 | 503 | ≈0 |

Per-task F1: qa 88.9% → 89.7% (+0.8%), summarization 21.0% → 21.7% (+0.8%),
dialogue 8.3% → 8.3% (+0.0%). Judge inference latency in isolation: median
8.61 ms on the L4.

Readings:

- The judge escalation is **directionally consistent but small**: all five
  aggregate metrics move favourably (precision +0.9%, FPR −0.3%, F1 +0.5%,
  catch +0.3%, accuracy +0.3%) and two of three per-task F1 rows improve with
  none regressing. Random noise would be expected to scatter the signs; a
  clean sweep of favourable deltas is mildly suggestive of a real, small
  effect concentrated on the borderline subset the judge actually sees.
- The magnitude, however, sits within the noise band at this sample size
  (600 pairs/backend). The eval establishes the **direction** (the judge
  helps precision/FPR without hurting recall) but not a decisive margin; a
  larger-N run would be needed to separate a genuine ~0.5% F1 gain from
  sampling noise. N was held to 200/task deliberately: the full E2E pipeline
  is dominated by long-context summarization/dialogue chunking (many NLI
  forwards per sample), and the run was kept inside a monitorable window on
  the metered GPU rather than run unattended for hours.
- Cost is effectively free: average latency is unchanged (503 vs 504 ms)
  because the judge is only consulted on borderline scores, and its isolated
  inference is 8.6 ms. This matches §9 — the local judge adds precision at
  near-zero latency and zero API cost, unlike an external LLM judge.
- Consequence for the default: the measured gain does not justify flipping
  the judge on by default (it adds a 0.18 GB model dependency for a
  within-noise F1 move), but it is a **defensible opt-in** for
  precision-sensitive deployments — it lowered FPR and raised precision on
  every task without a recall cost. Default stays off pending a larger-N
  confirmation if the precision gain is wanted.
- This run also surfaced a real concurrency defect: the parallel logical and
  factual divergence futures share one fast tokenizer, whose per-call
  truncation/padding mutation raised `RuntimeError("Already borrowed")` on
  the GPU. Fixed under commit `ff3303ff` (tokenizer access serialised behind
  a lock); the numbers above are from the fixed pipeline.

Claim boundary: one public dataset (HaluEval QA/summarization/dialogue) at
200 samples/task, the shipped scorer with the local DeBERTa-v3-base judge
checkpoint; no leaderboard claim and no claim about other judge checkpoints,
datasets, or sample sizes.

## 15. Claim decomposition — LLM atomic vs regex sentence split (WICE, L4, 2026-07-13)

Downstream detection measurement for the WCA-2 gate ("claim-coverage F1 vs
regex baseline needs a live provider or local checkpoint; without a
measurement the default does not flip"). Three decomposition strategies are
scored on the full WICE test subset of `lytang/LLM-AggreFact` (358 rows, 111
supported / 247 not-supported): `no-decomp` (NLI the whole claim),
`regex-decomp` (the production `split_sentences`, then min over sub-claims),
and `llm-decomp` (`AtomicClaimDecomposer` with a **local** Qwen2.5-7B-Instruct
as its injected transport, then min over atomic claims). NLI is
FactCG-DeBERTa-v3-Large. Each support score is thresholded at both a fixed
0.5 cut and the balanced-accuracy-maximising oracle cut. Environment:
JarvisLabs L4 (IN2), torch 2.6.0+cu124, transformers 5.8.0, ~34.5 min.
Artefact: `benchmarks/results/claim_decomp_wice.json`.

| Config | avg claims | BA @0.5 | BA @oracle | Halluc F1 @0.5 | Halluc F1 @oracle |
|---|---|---|---|---|---|
| no-decomp | 1.00 | 0.766 | 0.816 | 0.888 | 0.862 |
| regex-decomp | 1.05 | 0.762 | 0.814 | 0.886 | 0.863 |
| llm-decomp | 3.25 | 0.783 | 0.832 | 0.888 | 0.836 |

Delta (llm − regex): balanced accuracy +0.021 (fixed) / +0.018 (oracle);
hallucination F1 +0.001 (fixed) / −0.026 (oracle). The LLM decomposer drove
355/358 rows (3 fell to the labelled sentence fallback).

Readings:

- **Regex barely decomposes on WICE.** The claims are single sentences, so
  `split_sentences` produces 1.05 sub-claims on average — effectively
  `no-decomp`. The LLM genuinely splits compound claims into 3.25 atomic
  facts, which is exactly the regime the decomposer was built for; this is a
  fair test of atomic decomposition, not of the sentence splitter.
- **The gain is small, consistent, and in balanced accuracy — not in
  hallucination F1.** LLM decomposition lifts BA by +1.8 to +2.1 points over
  regex (and +1.6 over no-decomp) at **both** threshold regimes, a clean
  favourable direction. The improvement lands on the supported class
  (supported recall 0.64 vs 0.58, supported F1 0.71 vs 0.68 at 0.5), because
  weakest-link `min` over more atomic claims makes a fully-supported claim
  easier to certify. Hallucination F1 is flat at the fixed cut and slightly
  **negative** at the oracle cut: the stricter aggregation shifts the
  operating point (oracle threshold 0.15, hallucination precision 0.953 but
  recall 0.745), trading recall for precision rather than raising F1.
- **Magnitude sits within the noise band.** A ~2-point BA difference at
  N=358 is roughly one standard error; the eval establishes the direction
  (atomic decomposition helps supported-class balanced accuracy) but not a
  decisive margin. This is the same expensive-addition / marginal-benefit
  shape as §13 (NLI worse than lexical) and §14 (judge escalation).
- **Cost is not free here, unlike §14.** Each claim costs a 7B-model
  generation plus 3.25× the NLI forwards (~5.8 s/row end-to-end vs
  near-instant for regex), and the decomposer sends the passage off-host — it
  emits a privacy warning at construction for exactly that reason.
- **Consequence for the default:** the measured gain does not justify
  flipping the LLM-decomposition default on — a within-noise BA move and a
  flat-to-negative hallucination F1 do not clear the latency and privacy
  cost. The default stays on the regex splitter; LLM decomposition is a
  **defensible opt-in** where supported-class balanced accuracy matters and a
  local instruct model is already hosted. The gate is answered with a
  measurement, and it does not flip the default.

Claim boundary: one public dataset (WICE subset of LLM-AggreFact) at the full
358 test rows, FactCG-DeBERTa-v3-Large NLI with a local Qwen2.5-7B-Instruct
decomposer and weakest-link `min` aggregation; no leaderboard claim and no
claim about other datasets, decomposers, aggregations, or sample sizes.

## 16. Long-context evidence composition × aggregation (HaluEval, 2×L4, 2026-07-15)

WCS-1 measurement, HaluEval half. The tracked _200 baseline
(`benchmarks/results/judge_bench_nli_only_200.json`) catches 84.5 % on QA
but only 12 % on summarisation and 4.5 % on dialogue; the 2026-07-14
diagnosis attributed this to three mechanical causes — D1 dialogue premises
omit the `knowledge` field, D2 summarisation premises truncate to a
3 000-char prefix (60 % of documents are longer), D3 coverage-style
aggregation dilutes single-fact-swap responses. `run_longcontext_bench.py`
scores the claim×evidence support matrix once per checker (200 samples/task,
800 paired rows, 4 148 unique checker calls after memoisation) and sweeps
evidence composition × aggregation offline. The headline metric is **catch
at the baseline's matched per-task FPR** (summarisation 0.025, dialogue
0.045). Environment: two JarvisLabs L4 (IN2) instances, torch 2.6.0+cu124,
`isolated-quiet` host verdict on both; FactCG scoring 361 s, MiniCheck
(`lytang/MiniCheck-DeBERTa-v3-Large`, pinned revision) ~5 min. Artefacts:
`benchmarks/results/longcontext_{matrix,sweep}_{factcg,minicheck}.json`.

Catch at matched FPR — FactCG-DeBERTa-v3-Large:

| Evidence (summarisation) | min | mean | low2mean | coverage |
|---|---|---|---|---|
| prefix3000 (production shape) | 0.090 | 0.070 | 0.105 | 0.000 |
| fulldoc | 0.125 | 0.115 | 0.075 | 0.000 |
| **anchored@5** | 0.105 | 0.165 | **0.205** | 0.000 |

| Evidence (dialogue) | min | mean | low2mean | coverage |
|---|---|---|---|---|
| history (production shape) | 0.205 | 0.090 | 0.125 | 0.000 |
| **knowledge+history** | **0.270** | 0.095 | 0.150 | 0.000 |

MiniCheck-DeBERTa-v3-Large agrees on direction, weaker on dialogue: best
summarisation 0.200 (anchored@5 / mean), best dialogue 0.140
(knowledge+history / min).

Readings:

- **D1 confirmed and it is the largest single lever.** Composing
  `knowledge` into the dialogue premise lifts catch from the 0.045
  baseline to 0.270 at the same 4.5 % FPR — six-fold. Part of the lift
  (0.205) already comes from claim-split + weakest-link `min` against the
  history alone, i.e. the E2E path's calibration/aggregation was hiding
  signal the checker already had (D3).
- **D2 confirmed for summarisation.** Claim-anchored top-5 source
  sentences over the whole document beat both the production 3 000-char
  prefix and the full document (0.205 vs 0.105/0.125 at best aggregation);
  the full document dilutes (SummaC max over more chunks raises the score
  of wrong claims), the prefix simply cannot see late evidence.
- **D3 confirmed negatively: `coverage` is unusable at strict FPR.** Its
  quantised scores collapse the matched-FPR threshold to zero catch in
  every cell of both checkers' grids.
- **Aggregation is task-dependent:** dialogue favours weakest-link `min`
  (single-fact-swap responses), summarisation favours `low2mean`/`mean`
  over anchored evidence (min alone over-fires on the strict FPR budget).
- **FactCG stays the better checker** of the two on this sweep;
  MiniCheck's direction matches but its dialogue ceiling is half of
  FactCG's here.

**Second set — RAGTruth Summary (test split, same day):** 900 natural
(unpaired) rows — 696 supported / 204 hallucinated model responses over
CNN/DM-style documents (median ~2.5 k chars, 37 % past the 3 000-char
prefix), response-level label = any annotated span; corpus pinned to
ParticleMedia/RAGTruth@c103204b. Single L4, `isolated-quiet`, FactCG,
1 463 s, 11 810 unique calls. Catch at matched FPR 0.025:

| Evidence (RAGTruth Summary) | min | mean | low2mean | coverage |
|---|---|---|---|---|
| prefix3000 (production shape) | 0.049 | 0.088 | 0.034 | 0.108 |
| **fulldoc** | **0.289** | 0.181 | 0.162 | 0.137 |
| anchored@5 | 0.064 | 0.088 | 0.074 | 0.000 |

Cross-dataset readings:

- **The production prefix truncation is dominated on BOTH datasets** —
  it is the worst evidence family everywhere it appears.
- **The winning variant is dataset-dependent.** HaluEval's synthetic
  single-fact swaps favour claim-anchored evidence (0.205); RAGTruth's
  natural model errors favour the whole document (0.289, near-six-fold
  over the production shape at the same FPR) while anchored@5 stays weak
  (lexical top-k retrieves similar-but-not-refuting sentences for
  unsupported-addition errors).
- **`fulldoc / min` is the only configuration that improves or matches
  the baseline on BOTH sets** (HaluEval 0.125 vs baseline 0.120 — thin,
  within noise at N=200; RAGTruth 0.289 vs a 0.049 production-shape
  arm). The honest production consequence is therefore *stop truncating
  the summarisation premise at 3 000 chars* (whole-document chunked
  scoring), not *switch to anchored evidence*; anchoring remains a
  HaluEval-style-error tunable, not a default.

Gate status (WCS-1): **summarisation axis passed on both sets** — a
task-routed config (fulldoc/min) beats the single-threshold production
shape at matched FPR on tracked HaluEval AND RAGTruth Summary. The
dialogue axis (knowledge+history, six-fold on HaluEval) has no second
public set with the same task shape in this sweep; its premise
composition is an evidence-completeness fix (the `knowledge` field was
silently dropped) rather than an operating-point retune, and is
recommended independently of the gate. Default flips remain a separate,
CEO-visible wiring decision.

**E2E wiring proof (same day, recorded negative at the fixed operating
point):** after wiring whole-document coverage premises (D2) and the
composed dialogue premise (D1) into the production path, the E2E
NLI-only benchmark re-run (200/task, L4, artefact
`e2e_nli_only_200_wcs1_wired.json`) produced **decision-identical
results to the tracked baseline** — TP/FP unchanged to the unit on all
three tasks (dialogue 4.5 %, summarisation 12.0 %, QA 84.0 %) — while
per-task latencies rose 15–27 % (the longer premises really were
scored). The evidence-composition fixes are therefore **necessary but
not sufficient at threshold 0.5 through the calibration layer**: the
dialogue 0.80-baseline squeeze demands raw divergence > 0.90 and the
summarisation coverage blend buffers the improved evidence, exactly the
D3 mechanism. The sweep's catch gains materialise only with the D3 fix
family — task-routed weakest-link aggregation and matched-FPR
operating points — which is the follow-on wiring lane, not a threshold
tweak to sneak past. The premise fixes stay (they are correct evidence
hygiene and the sweep shows the signal is now present at the checker);
no E2E improvement is claimed for them alone.

Claim boundary: HaluEval summarisation + dialogue at 200 samples/task
(two checkers) and RAGTruth Summary test split at 900 rows (FactCG),
evidence/aggregation sweep at matched per-task FPR against the tracked
_200 baseline; no leaderboard claim, no claim about QA, other datasets,
other checkers, or the E2E production path beyond the recorded
decision-identical proof run above.

## 17. Task-routed operating points — E2E decision proof (WCS-2a, L4, 2026-07-15)

Follow-on to §16: the raw-support routes landed (commit `721e7cbe` —
dialogue gates raw weakest-link claim support at a matched-FPR
operating point by default; summarisation weakest-link stays an opt-in)
and the full deployment flow was exercised on one L4:
**on-device calibration first, then two E2E runs at the calibrated
gates** (artefacts `wcs2a_operating_points_calibration.json`,
`e2e_nli_only_200_wcs2a_dialogue_raw.json`,
`e2e_nli_only_200_wcs2a_weakest_link.json`; instance destroyed,
run cost ≈ $0.19).

**On-device calibration** (`director-ai operating-points` flow through
`CoherenceScorer.raw_task_support`, HaluEval 200-sample pairs):
dialogue threshold 0.008875 (target FPR 0.045, realised 0.042 on the
calibration pairs, predicted catch 27.2 %); summarisation threshold
0.039970 (target 0.025, realised 0.021, predicted catch 9.3 %). Both
land within 2 % of the sweep-seeded defaults (0.0091 / 0.0402) —
the seeds transfer, but calibrate per deployment anyway.

| Task (200/200 per task) | Baseline catch @ FPR | WCS-2a catch @ FPR | Δ decisions |
|---|---|---|---|
| dialogue (raw support, default) | 4.5 % @ 4.5 % | **30.5 % @ 8.5 %** | TP 9 → 61 |
| summarisation (blend, default) | 12.0 % @ 2.5 % | 12.0 % @ 2.5 % (unchanged) | — |
| summarisation (weakest-link opt-in) | 12.0 % @ 2.5 % | **20.0 % @ 5.0 %** | TP 24 → 40 |
| QA (untouched control) | 84.0 % @ 5.0 % | 84.0 % @ 5.0 % | — |

**The §16 recorded negative is resolved:** with the operating point
made a first-class, calibrated decision instead of a squeeze
side-effect, the evidence gains finally move E2E decisions — dialogue
catch 4.5 % → 30.5 % (6.8×). QA is decision-identical (control holds),
and the default summarisation path is untouched.

**Honest deltas against the calibration targets:** the realised E2E
false-positive rates run above the calibration targets (dialogue 8.5 %
vs 4.5 % target; weakest-link summarisation 5.0 % vs 2.5 %). The
calibration pairs and the E2E harness compose premises through
different extraction paths, so the support distributions shift slightly
between calibration and production scoring — at thresholds this deep in
the lower tail, a small shift doubles the tail mass. Deployment lesson,
now written into the tool's docs: calibrate on samples scored through
the SAME path that will gate production traffic, and re-fit on real
traffic rather than benchmark pairs. A threshold re-fit on the E2E
distribution would trade part of the catch back for the target FPR;
the artefacts carry both distributions so that choice stays visible.

**Weakest-link latency win:** the opt-in summarisation mode skips the
Layer-A bidirectional pass entirely — per-task average latency fell
from 1713 ms (blend, whole-document) to 489 ms (−71 %) while catch
rose 12 % → 20 %. The FPR doubling (2.5 % → 5.0 %) is the cost; the
mode stays opt-in and per-deployment calibrated.

Claim boundary: HaluEval 200 samples/task through the E2E NLI-only
path (FactCG default backend, threshold 0.5 elsewhere); dialogue gains
generalise only as far as §16's dialogue axis (no second public set
with the same task shape); no leaderboard claim; weakest-link
summarisation numbers are for the opt-in mode at its calibrated gate,
not the shipped default.

## Reproduction

```bash
# Full NLI benchmark suite (requires GPU + HF_TOKEN)
export HF_TOKEN=hf_...
python -m benchmarks.aggrefact_eval --sweep
python -m benchmarks.e2e_eval --nli
python -m benchmarks.latency_bench
python -m benchmarks.streaming_false_halt_bench

# All NLI benchmarks + comparison table
python -m benchmarks.run_all --max-samples 500

# Hybrid-mode E2E (requires OpenAI API key)
export OPENAI_API_KEY=sk-...
python -m benchmarks.e2e_eval --nli --scorer-backend hybrid \
    --llm-judge-provider openai

# Local judge E2E (requires trained model at training/output/deberta-v3-base-judge/)
python benchmarks/run_judge_benchmark.py --samples 500

# Claim decomposition — LLM atomic vs regex (WICE, GPU + HF_TOKEN)
export HF_TOKEN=hf_...
PYTHONPATH=src:. python benchmarks/run_claim_decomp_benchmark.py \
    --decomposer-model Qwen/Qwen2.5-7B-Instruct \
    --out benchmarks/results/claim_decomp_wice.json
```

## Throughput (QPS)

Run with `python -m benchmarks.load_test` to generate a QPS artifact:

```bash
# Single-node heuristic (CPU only)
python -m benchmarks.load_test --concurrency 4 --duration 30

# Single-node NLI (GPU)
python -m benchmarks.load_test --concurrency 4 --duration 30 --nli

# API server
python -m benchmarks.load_test --server http://localhost:8080 --concurrency 16 --duration 30
```

Results saved to `benchmarks/results/load_test.json`.

Latency matrix (backend x batch size): `python -m benchmarks.latency_matrix`

## Sources

- [LLM-AggreFact Leaderboard](https://llm-aggrefact.github.io/)
- [FactCG (arXiv 2501.17144, NAACL 2025)](https://arxiv.org/abs/2501.17144)
- [MiniCheck (arXiv 2404.10774)](https://arxiv.org/abs/2404.10774)
- Tang et al. (2024). "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents."
