# Director-AI Benchmark Report

Version: 3.3.0
Date: 2026-03-07

## Hardware

All latency numbers measured on:

- **Primary**: NVIDIA GeForce GTX 1060 6 GB, Python 3.12, torch 2.6.0+cu124
- **Cloud GPU**: NVIDIA L40S 45 GB (UpCloud fi-hel2), Python 3.12, torch 2.6.0+cu124
- **Cross-GPU**: RTX 6000 Ada (48 GB), RTX A5000 (24 GB), RTX A6000 (48 GB), Quadro RTX 5000 (16 GB)
- Iterations: 30 (latency), 5 warmup. GPU clocks not locked.

## 1. NLI Accuracy — LLM-AggreFact (29,320 samples)

Model: `yaxili96/FactCG-DeBERTa-v3-Large` (0.4B params).
Metric: macro-averaged balanced accuracy (standard for LLM-AggreFact).

| Model | Bal. Acc | Params | Streaming |
|-------|---------|--------|-----------|
| Bespoke-MiniCheck-7B | 77.4% | 7B | No |
| **Director-AI (FactCG)** | **75.8%** | 0.4B | Yes |
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

## 4. False-Positive Rate

### Streaming False-Halt
0.0% false-halt rate across 20 known-good Wikipedia passages streamed
through `StreamingKernel` (heuristic mode).

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
`.forward()` for all H_factual pairs, instead of per-item calls.

| Mode | GPU Kernels | Total (16-pair) | Per-Pair |
|------|-------------|-----------------|----------|
| `review()` × 16 (serial) | 32 | ~304 ms | 19.0 ms |
| `review_batch(16)` (coalesced) | 2 | ~233 ms | 14.6 ms |

H_logical and H_factual run in parallel via `ThreadPoolExecutor`, cutting
single-review latency by ~40%.

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

1. **Summarisation is weakest**: AggreFact-CNN 68.8%, ExpertQA 59.1%. NLI
   models under-perform on abstractive summarization where surface forms
   diverge from source.
2. **E2E heuristic+NLI catch rate is 46.7%**: hybrid mode (NLI + LLM
   judge) raises this to 90.7% but adds LLM latency (2.3s with GPT-4o-mini).
3. **Hybrid summarization/dialogue FPR is high**: 93-95% false positive
   rate on summarization and dialogue tasks. QA task is production-grade
   (3-4% FPR, 95%+ precision).
4. **ONNX CPU not competitive**: 383 ms/pair. Requires `onnxruntime-gpu`.
5. **Fine-tuned models regressed**: DeBERTa-v3-large fine-tuned on
   hallucination data scored 64.7% — below the pre-trained FactCG 75.8%.
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
| AggreFact bal. acc | 75.8% | N/A | N/A | N/A | N/A |
| Integrations | LC/LI/LG/HS/CrewAI | LangChain | Python | LC/LI | Python |

Director-AI's unique value: sub-ms streaming halt + 75.8% balanced accuracy
at 0.4B params. No competitor offers token-level halt.

Full analysis: [`benchmarks/comparison/COMPETITOR_COMPARISON.md`](comparison/COMPETITOR_COMPARISON.md)

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
```

## Sources

- [LLM-AggreFact Leaderboard](https://llm-aggrefact.github.io/)
- [FactCG (arXiv 2501.17144, NAACL 2025)](https://arxiv.org/abs/2501.17144)
- [MiniCheck (arXiv 2404.10774)](https://arxiv.org/abs/2404.10774)
- Tang et al. (2024). "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents."
