# Director-AI -- Competitor Benchmark Comparison

Last updated: 2026-02-26

## Measured Latency (benchmarks/latency_bench.py)

Measured on CPU (i7-class, no GPU). Iterations=50, warmup=5.

| Pipeline | Median | P95 | P99 |
|----------|--------|-----|-----|
| Lightweight (no NLI) | 0.03 ms | 0.09 ms | 0.17 ms |
| Streaming session | 0.02 ms | 0.05 ms | 0.11 ms |
| NLI forward pass | 458 ms | 681 ms | 784 ms |
| Full pipeline (NLI) | 624 ms | 1472 ms | 1620 ms |

The lightweight path (embedding similarity only) is sub-millisecond.
NLI adds a DeBERTa-v3-base forward pass; on GPU this drops to ~30-50 ms.

## Apples-to-Apples: LLM-AggreFact Leaderboard

All models below evaluated on the same benchmark (29,320 samples, 11 datasets).
Metric: macro-averaged balanced accuracy.

| Tool | Bal. Acc | Params | Latency (CPU) | Streaming | License |
|------|---------|--------|---------------|-----------|---------|
| Bespoke-MiniCheck-7B | **77.4%** | 7B | ~100 ms (vLLM, A6000) | No | Apache 2.0 |
| MiniCheck-Flan-T5-L | 75.0% | 0.8B | ~120 ms | No | MIT |
| MiniCheck-DeBERTa-L | 72.6% | 0.4B | ~120 ms | No | MIT |
| HHEM-2.1-Open | 71.8% | ~0.4B | ~200 ms (est.) | No | Apache 2.0 |
| **Director-AI (NLI)** | **66.2%** | 0.4B | 458 ms (measured) | **Yes** | AGPL v3 |
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

1. **Token-level streaming halt** -- no competitor offers this. All others are post-hoc.
2. **No LLM API dependency** -- local DeBERTa model, runs offline on CPU.
3. **Sub-ms lightweight path** -- embedding-only mode at <0.1 ms makes real-time streaming feasible.
4. **Dual-entropy** -- NLI + RAG combined signal. Most competitors do one or the other.
5. **Ecosystem integration** -- drop-in LangChain guard and LlamaIndex postprocessor.

## Where Director-AI Loses

1. **Accuracy gap** -- 66.2% vs 72.6% (MiniCheck-DeBERTa, same weight class).
2. **NLI latency on CPU** -- 458 ms median is 4x slower than MiniCheck on GPU. GPU brings parity.
3. **Fine-tuned models regress** -- fine-tuned DeBERTa-v3-large scored 64.7%, below baseline.
4. **Summarization is weak** -- AggreFact-CNN 53.0%, TofuEval-MediaS 47.7%.

## Path to Closing the Gap

1. **Swap NLI checkpoint** -- FactCG-DeBERTa-L reaches 75.6% on the same benchmark with the same architecture.
2. **Fix fine-tuning** -- current pipeline regresses; likely a learning rate or data imbalance issue.
3. **Ensemble** -- combine DeBERTa NLI with a lightweight claim decomposition step for summarization.
4. **GPU deployment** -- ONNX export or TorchScript for ~30-50 ms NLI inference.

## Sources

- [LLM-AggreFact Leaderboard](https://llm-aggrefact.github.io/)
- [MiniCheck (arXiv 2404.10774)](https://arxiv.org/abs/2404.10774)
- [Vectara HHEM-2.1](https://huggingface.co/vectara/hallucination_evaluation_model)
- [Lynx (arXiv 2407.08488)](https://arxiv.org/abs/2407.08488)
- [SelfCheckGPT (arXiv 2303.08896)](https://arxiv.org/abs/2303.08896)
- [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/)
- [RAGAS Docs](https://docs.ragas.io/)
