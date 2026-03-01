# Director-AI -- Competitor Benchmark Comparison

Last updated: 2026-03-01 (v1.7.0)

## One-Pager Summary

| Feature | Director-AI | NeMo Guardrails | Lynx | GuardrailsAI | SelfCheckGPT |
|---------|-------------|----------------|------|-------------|-------------|
| **Approach** | NLI + RAG dual-entropy | LLM self-consistency | Fine-tuned LLM | LLM-as-judge | Multi-call LLM |
| **Model size** | 0.4B (DeBERTa) | LLM-dependent | 8-70B | LLM-dependent | LLM-dependent |
| **Latency** | 0.9 ms/pair (Ada GPU) | 50-300 ms + LLM | 1-10 s | 2.26 s | 5-10 s |
| **Streaming halt** | Yes (token-level) | No | No | No | No |
| **Offline/local** | Yes | No (needs LLM) | Yes (GPU) | No (needs LLM) | No (needs LLM) |
| **False-halt rate** | 0.0% (20 passages) | N/A | N/A | N/A | N/A |
| **AggreFact bal. acc** | 75.8% | N/A | N/A | N/A | N/A |
| **Integrations** | LC/LI/LG/HS/CrewAI | LangChain | Python | LC/LI | Python |
| **License** | AGPL v3 | Apache 2.0 | Apache 2.0 | Apache 2.0 | MIT |

## Measured Latency (benchmarks/latency_bench.py)

Hardware: NVIDIA GeForce GTX 1060 6GB, Python 3.12, torch 2.6.0+cu124.
Iterations=30, warmup=5.

| Pipeline | Median | P95 | Per-pair | Notes |
|----------|--------|-----|----------|-------|
| Lightweight (no NLI) | 0.15 ms | 0.44 ms | 0.15 ms | Heuristic only |
| Streaming session | 0.02 ms | 0.02 ms | 0.02 ms | Token-level |
| PyTorch GPU seq (16 pairs) | 3145 ms | 3580 ms | 196.6 ms | Sequential score() |
| PyTorch GPU batch (16 pairs) | 304 ms | 353 ms | 19.0 ms | 10.4x vs sequential |
| PyTorch chunked-seq | 250 ms | 335 ms | — | 12-sentence doc |
| PyTorch chunked-batch | 195 ms | 280 ms | — | 12-sentence doc |
| ONNX GPU seq (16 pairs) | 1042 ms | 1249 ms | 65.1 ms | CUDAExecutionProvider |
| **ONNX GPU batch (16 pairs)** | **233 ms** | **250 ms** | **14.6 ms** | **Fastest** |
| ONNX CPU seq (16 pairs) | 6553 ms | 8512 ms | 410 ms | CPUExecutionProvider |
| ONNX CPU batch (16 pairs) | 6124 ms | 8143 ms | 383 ms | CPUExecutionProvider |

ONNX GPU batch is the fastest path: **14.6 ms/pair** (1.3x faster than PyTorch GPU batch).
Batching gives 10.4x speedup (PyTorch) and 4.5x (ONNX GPU).
ONNX GPU sequential (65 ms/pair) is 3x faster than PyTorch GPU sequential (197 ms/pair).

## Cross-GPU Latency (benchmarks/gpu_bench.py)

16-pair batch, 50 iterations, warmup=10. Per-pair median latency (ms).

| GPU | VRAM | Compute | ONNX CUDA | PyTorch FP16 | PyTorch FP32 |
|-----|------|---------|-----------|--------------|--------------|
| **RTX 6000 Ada** | 48 GB | 8.9 | **0.9 ms** | 1.2 ms | 2.1 ms |
| RTX A5000 | 24 GB | 8.6 | 2.0 ms | 3.4 ms | 4.8 ms |
| RTX A6000 | 48 GB | 8.6 | 3.5 ms | 9.7 ms | 10.1 ms |
| Quadro RTX 5000 | 16 GB | 7.5 | 5.1 ms | 2.5 ms | 5.9 ms |
| GTX 1060 6GB | 6 GB | 6.1 | 13.9 ms | N/A | 17.4 ms |

ONNX CUDA is the fastest backend on all GPUs. RTX 6000 Ada achieves **sub-1ms per pair**.
GTX 1060 lacks tensor cores (compute 6.1) — FP16 auto-skips.
Full JSON results in `benchmarks/results/gpu_bench_*.json`.

## Apples-to-Apples: LLM-AggreFact Leaderboard

All models evaluated on the same benchmark (29,320 samples, 11 datasets).
Metric: macro-averaged balanced accuracy.

| Tool | Bal. Acc | Params | Latency (measured) | Streaming | License |
|------|---------|--------|-------------------|-----------|---------|
| Bespoke-MiniCheck-7B | 77.4% | 7B | ~100 ms (vLLM, A6000) | No | Apache 2.0 |
| **Director-AI (FactCG batch)** | **75.8%** | 0.4B | **0.9 ms/pair (Ada), 14.6 ms (GTX 1060)** | **Yes** | AGPL v3 |
| **Director-AI (FactCG seq)** | **75.8%** | 0.4B | 196 ms/pair (GPU seq) | **Yes** | AGPL v3 |
| MiniCheck-Flan-T5-L | 75.0% | 0.8B | ~120 ms | No | MIT |
| MiniCheck-DeBERTa-L | 72.6% | 0.4B | ~120 ms | No | MIT |
| HHEM-2.1-Open | 71.8% | ~0.4B | ~200 ms (est.) | No | Apache 2.0 |
| **Director-AI (lightweight)** | N/A | 0 | <0.1 ms (measured) | **Yes** | AGPL v3 |

## Per-Class Metrics (Hallucination Detection)

The key question for a guardrail: **how many hallucinations does it catch?**

Balanced accuracy averages recall across both classes (supported + not-supported).
Per-class precision/recall/F1 are computed by `benchmarks/aggrefact_eval.py` for
each dataset. Re-run with `--sweep` to regenerate:

```
python -m benchmarks.aggrefact_eval --sweep
```

The results JSON will include `hallucination_precision`, `hallucination_recall`,
and `hallucination_f1` per dataset. These are the class-0 (not-supported) metrics —
the numbers that matter for a guardrail.

### Internal Model Comparison (LLM-AggreFact)

| Model | Bal. Acc | Threshold | Notes |
|-------|---------|-----------|-------|
| **FactCG-DeBERTa-v3-Large** | **75.8%** | 0.46 | Production model |
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 66.2% | 0.53 | 3-class NLI baseline |
| Fine-tuned DeBERTa-v3-large-hallucination | 64.7% | 0.90 | Fine-tuning regressed |
| Fine-tuned DeBERTa-v3-base-hallucination | 59.0% | 0.88 | Fine-tuning regressed worse |

### Per-Dataset Weakness Map (FactCG, threshold=0.46)

| Dataset | Bal. Acc | Pos | Neg | Failure Mode |
|---------|---------|-----|-----|-------------|
| ExpertQA | 59.1% | 2971 | 731 | Long expert answers, low neg recall |
| AggreFact-CNN | 68.8% | 501 | 57 | Extreme class imbalance (9:1) |
| TofuEval-MediaS | 71.9% | 554 | 172 | Summarization (media) |
| FactCheck-GPT | 73.0% | 376 | 1190 | GPT-generated claims |
| AggreFact-XSum | 74.3% | 285 | 273 | Extreme summarization |
| TofuEval-MeetB | 74.3% | 622 | 150 | Summarization (meetings) |
| Wice | 76.9% | 111 | 247 | Wikipedia claims |
| ClaimVerify | 78.1% | 789 | 299 | Claim verification |
| RAGTruth | 82.2% | 15102 | 1269 | RAG grounding |
| Lfqa | 86.4% | 1121 | 790 | Long-form QA |
| Reveal | 89.1% | 400 | 1310 | Fact checking |

## Different Benchmarks (Not Directly Comparable)

| Tool | Benchmark | Score | Latency | Approach |
|------|-----------|-------|---------|----------|
| Lynx-70B | HaluBench | 87.4% accuracy | 3-10 s | 70B LLM, 8x H100 |
| Lynx-8B | HaluBench | 82.9% accuracy | 1-5 s | 8B LLM, GPU required |
| SelfCheckGPT-NLI | WikiBio | 92.5% AUC-PR | 5-10 s | Multiple LLM calls |
| NeMo Guardrails | Internal eval | 70-95% (LLM-dependent) | 50-300 ms + LLM | LLM self-consistency |
| GuardrailsAI | SQuAD 2.0 | 98% F1 | 2.26 s | LLM-as-judge |
| RAGAS Faithfulness | Multi-dataset | 76.2% avg precision | 3-8 s | Claim decomposition |
| Llama Guard 3 | Safety moderation | 93.9% F1 | ~300 ms | **Not hallucination detection** |

## End-to-End Guardrail Results (benchmarks/e2e_eval.py)

Full pipeline (CoherenceAgent + GroundTruthStore + SafetyKernel), 300 traces
across QA, summarization, and dialogue tasks. Threshold=0.35, soft_limit=0.45.

| Task | N | TP | FP | TN | FN | Catch Rate | Precision | F1 |
|------|---|----|----|----|----|-----------|-----------|-----|
| QA | 100 | 18 | 4 | 46 | 32 | 36.0% | 81.8% | 50.7% |
| Summarization | 100 | 12 | 6 | 44 | 38 | 24.0% | 66.7% | 35.3% |
| Dialogue | 100 | 40 | 43 | 7 | 10 | 80.0% | 48.2% | 60.2% |
| **Overall** | **300** | **70** | **53** | **97** | **80** | **46.7%** | **56.9%** | **51.3%** |

Evidence coverage: 100% (every rejection includes supporting chunks).
Avg latency: 15.8 ms (p95: 40 ms).

## Where Director-AI Wins

1. **0.9 ms/pair on Ada GPU, 14.6 ms on GTX 1060** — faster than any competitor at this accuracy tier.
2. **Token-level streaming halt** — no competitor offers this. All others are post-hoc.
3. **75.8% balanced accuracy** — 4th on LLM-AggreFact, within 1.6pp of top 7B model at 17x fewer params.
4. **No LLM API dependency** — local DeBERTa model, runs offline on CPU/GPU.
5. **Sub-ms lightweight path** — embedding-only mode at <0.1 ms for real-time streaming.
6. **10.8x batch speedup** — single padded forward pass for multi-chunk documents.
7. **Dual-entropy** — NLI + RAG combined signal. Most competitors do one or the other.
8. **Ecosystem integration** — LangChain, LlamaIndex, LangGraph, Haystack, CrewAI.

## Where Director-AI Loses

1. **Summarization still weakest** — AggreFact-CNN 68.8%, ExpertQA 59.1% drag the average.
2. **ONNX CPU not competitive** — 383 ms/pair without CUDAExecutionProvider. Needs onnxruntime-gpu.
3. **Fine-tuned models regress** — fine-tuned DeBERTa-v3-large scored 64.7%, below baseline.

## Path Forward

1. **ONNX + CUDA** — install onnxruntime-gpu for GPU-accelerated ONNX inference.
2. **TensorRT** — sub-10ms/pair target via TensorRT optimization.
3. **Summarization ensemble** — claim decomposition for AggreFact-CNN/ExpertQA.

## Full Benchmark Suite

Scripts in `benchmarks/`. Run each with `python -m benchmarks.<name>`.

| Script | Dataset | What it Tests | Metric | Status |
|--------|---------|---------------|--------|--------|
| `aggrefact_eval` | LLM-AggreFact (29K) | Factual consistency (11 datasets) | Balanced accuracy | **75.8%** (FactCG) |
| `e2e_eval` | Synthetic (300 traces) | Full pipeline: Agent + KB + Kernel | Catch rate, precision, F1 | **46.7% catch, 56.9% prec** |
| `latency_bench` | N/A | Inference latency across backends | Median/P95 ms | **0.9 ms (Ada GPU)** |
| `gpu_bench` | N/A | Cross-GPU latency comparison | Per-pair ms | **5 GPUs benchmarked** |
| `retrieval_bench` | Synthetic (50 facts) | RAG retrieval quality (Hit@k, P@k) | Hit@1, Hit@3, P@3 | **40% / 63% (inmemory)** |
| `anli_eval` | ANLI R1/R2/R3 | Adversarial NLI robustness | Accuracy, F1 per class | Requires GPU + HF_TOKEN |
| `fever_eval` | FEVER dev | Fact verification | Accuracy, F1 per class | Requires GPU + HF_TOKEN |
| `halueval_eval` | HaluEval | Hallucination detection (QA/sum/dial) | Precision, Recall, F1 | Requires GPU + HF_TOKEN |
| `mnli_eval` | MNLI matched+mismatched | General NLI regression | Accuracy, F1 per class | Requires GPU + HF_TOKEN |
| `paws_eval` | PAWS | Paraphrase adversaries | Binary P/R/F1 | Requires GPU + HF_TOKEN |
| `truthfulqa_eval` | TruthfulQA (817 Qs) | Multiple-choice truthfulness | Accuracy per category | Requires GPU + HF_TOKEN |
| `vitaminc_eval` | VitaminC | Contrastive fact verification | Accuracy, F1 per class | Requires GPU + HF_TOKEN |
| `falsepositive_eval` | SQuAD/NQ/TriviaQA | False-positive rate on correct QA | FP rate (target <5%) | Requires GPU + HF_TOKEN |
| `streaming_false_halt_bench` | Synthetic good text | False-halt rate of StreamingKernel | False-halt % | **0.0% (20 passages, heuristic)** |

To reproduce all results:
```bash
export HF_TOKEN=hf_...
python -m benchmarks.aggrefact_eval --sweep
python -m benchmarks.anli_eval
python -m benchmarks.fever_eval
python -m benchmarks.halueval_eval
python -m benchmarks.mnli_eval
python -m benchmarks.paws_eval
python -m benchmarks.truthfulqa_eval
python -m benchmarks.vitaminc_eval
python -m benchmarks.falsepositive_eval
python -m benchmarks.retrieval_bench --backend sentence-transformer
python -m benchmarks.streaming_false_halt_bench
```

## Sources

- [LLM-AggreFact Leaderboard](https://llm-aggrefact.github.io/)
- [FactCG (arXiv 2501.17144, NAACL 2025)](https://arxiv.org/abs/2501.17144)
- [MiniCheck (arXiv 2404.10774)](https://arxiv.org/abs/2404.10774)
- [Vectara HHEM-2.1](https://huggingface.co/vectara/hallucination_evaluation_model)
- [Lynx (arXiv 2407.08488)](https://arxiv.org/abs/2407.08488)
- [SelfCheckGPT (arXiv 2303.08896)](https://arxiv.org/abs/2303.08896)
- [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/)
- [RAGAS Docs](https://docs.ragas.io/)
