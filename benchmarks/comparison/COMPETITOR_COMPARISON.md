# Director-AI — Competitor Benchmark Comparison

Last updated: 2026-02-25

## Apples-to-Apples: LLM-AggreFact Leaderboard

All models below evaluated on the same benchmark (29,320 samples, 11 datasets).
Metric: macro-averaged balanced accuracy.

| Tool | Bal. Acc | Params | Latency | Streaming | License |
|------|---------|--------|---------|-----------|---------|
| Bespoke-MiniCheck-7B | **77.4%** | 7B | ~100 ms (vLLM, A6000) | No | Apache 2.0 |
| MiniCheck-Flan-T5-L | 75.0% | 0.8B | ~120 ms | No | MIT |
| MiniCheck-DeBERTa-L | 72.6% | 0.4B | ~120 ms | No | MIT |
| HHEM-2.1-Open | 71.8% | ~0.4B | ~200 ms (est.) | No | Apache 2.0 |
| **Director-AI (baseline)** | **66.2%** | 0.4B | 220 ms | **Yes** | AGPL v3 |

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

1. **Token-level streaming halt** — no competitor offers this. All others are post-hoc.
2. **No LLM API dependency** — local DeBERTa model, runs offline on CPU.
3. **Latency** — 220 ms vs 2-10 s for LLM-based approaches (SelfCheckGPT, RAGAS, GuardrailsAI).
4. **Dual-entropy** — NLI + RAG combined signal. Most competitors do one or the other.

## Where Director-AI Loses

1. **Accuracy gap** — 66.2% vs 72.6% (MiniCheck-DeBERTa, same weight class). This is the primary weakness.
2. **Fine-tuned models regress** — fine-tuned DeBERTa-v3-large scored 64.7%, *below* the 66.2% baseline.
3. **Summarization is weak** — AggreFact-CNN 53.0%, TofuEval-MediaS 47.7%.

## Path to Closing the Gap

1. **Swap NLI checkpoint** — FactCG-DeBERTa-L reaches 75.6% on the same benchmark with the same architecture.
2. **Fix fine-tuning** — current pipeline regresses; likely a learning rate or data imbalance issue.
3. **Ensemble** — combine DeBERTa NLI with a lightweight claim decomposition step for summarization.

## Sources

- [LLM-AggreFact Leaderboard](https://llm-aggrefact.github.io/)
- [MiniCheck (arXiv 2404.10774)](https://arxiv.org/abs/2404.10774)
- [Vectara HHEM-2.1](https://huggingface.co/vectara/hallucination_evaluation_model)
- [Lynx (arXiv 2407.08488)](https://arxiv.org/abs/2407.08488)
- [SelfCheckGPT (arXiv 2303.08896)](https://arxiv.org/abs/2303.08896)
- [NVIDIA NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/)
- [RAGAS Docs](https://docs.ragas.io/)
