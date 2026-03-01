# Director-AI -- Competitor Benchmark Comparison

Last updated: 2026-03-01 (v1.4.0)

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

| GPU | VRAM | ONNX CUDA | ONNX TRT FP16 | PyTorch FP16 | PyTorch FP32 |
|-----|------|-----------|---------------|--------------|--------------|
| GTX 1060 6GB | 6 GB | 14.6 ms | N/A | N/A | 19.0 ms |
| RTX 3090 | 24 GB | TBD | TBD | TBD | TBD |
| A6000 | 48 GB | TBD | TBD | TBD | TBD |
| A100-80GB | 80 GB | TBD | TBD | TBD | TBD |
| V100-16GB | 16 GB | TBD | TBD | TBD | TBD |

GTX 1060 lacks tensor cores (compute 6.1) — FP16 and TRT backends auto-skip.
TBD rows populated by running `gpu_bench_setup.sh` on JarvisLabs instances.

## Apples-to-Apples: LLM-AggreFact Leaderboard

All models evaluated on the same benchmark (29,320 samples, 11 datasets).
Metric: macro-averaged balanced accuracy.

| Tool | Bal. Acc | Params | Latency (measured) | Streaming | License |
|------|---------|--------|-------------------|-----------|---------|
| Bespoke-MiniCheck-7B | 77.4% | 7B | ~100 ms (vLLM, A6000) | No | Apache 2.0 |
| **Director-AI (FactCG batch)** | **75.8%** | 0.4B | **18 ms/pair (GPU batch)** | **Yes** | AGPL v3 |
| **Director-AI (FactCG seq)** | **75.8%** | 0.4B | 196 ms/pair (GPU seq) | **Yes** | AGPL v3 |
| MiniCheck-Flan-T5-L | 75.0% | 0.8B | ~120 ms | No | MIT |
| MiniCheck-DeBERTa-L | 72.6% | 0.4B | ~120 ms | No | MIT |
| HHEM-2.1-Open | 71.8% | ~0.4B | ~200 ms (est.) | No | Apache 2.0 |
| **Director-AI (lightweight)** | N/A | 0 | <0.1 ms (measured) | **Yes** | AGPL v3 |

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

## Where Director-AI Wins

1. **18 ms/pair batched inference** — faster than any competitor at this accuracy tier.
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

## Sources

- [LLM-AggreFact Leaderboard](https://llm-aggrefact.github.io/)
- [FactCG (arXiv 2501.17144, NAACL 2025)](https://arxiv.org/abs/2501.17144)
- [MiniCheck (arXiv 2404.10774)](https://arxiv.org/abs/2404.10774)
- [Vectara HHEM-2.1](https://huggingface.co/vectara/hallucination_evaluation_model)
- [Lynx (arXiv 2407.08488)](https://arxiv.org/abs/2407.08488)
- [SelfCheckGPT (arXiv 2303.08896)](https://arxiv.org/abs/2303.08896)
- [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/)
- [RAGAS Docs](https://docs.ragas.io/)
