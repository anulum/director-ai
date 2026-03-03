# Director-AI Benchmark Report

Version: 2.7.1
Date: 2026-03-03

## Hardware

All latency numbers measured on:

- **Primary**: NVIDIA GeForce GTX 1060 6 GB, Python 3.12, torch 2.6.0+cu124
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

| Dataset | Bal. Acc | Pos | Neg | Failure Mode |
|---------|---------|-----|-----|-------------|
| Reveal | 89.1% | 400 | 1310 | — |
| Lfqa | 86.4% | 1121 | 790 | — |
| RAGTruth | 82.2% | 15102 | 1269 | — |
| ClaimVerify | 78.1% | 789 | 299 | — |
| Wice | 76.9% | 111 | 247 | — |
| TofuEval-MeetB | 74.3% | 622 | 150 | Summarization |
| AggreFact-XSum | 74.3% | 285 | 273 | Extreme summarization |
| FactCheck-GPT | 73.0% | 376 | 1190 | GPT-generated claims |
| TofuEval-MediaS | 71.9% | 554 | 172 | Summarization (media) |
| AggreFact-CNN | 68.8% | 501 | 57 | Extreme class imbalance (9:1) |
| ExpertQA | 59.1% | 2971 | 731 | Long expert answers |

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
| RTX A5000 | 24 GB | 2.0 ms | 3.4 ms | 4.8 ms |
| RTX A6000 | 48 GB | 3.5 ms | 9.7 ms | 10.1 ms |
| Quadro RTX 5000 | 16 GB | 5.1 ms | 2.5 ms | 5.9 ms |
| GTX 1060 | 6 GB | 13.9 ms | N/A | 17.4 ms |

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

**Hybrid mode** (NLI + LLM judge) infrastructure is wired as of v2.7.1
but has not yet been run at scale. Expected to improve summarization catch
rate. Run with:
```bash
python -m benchmarks.e2e_eval --nli --scorer-backend hybrid \
    --llm-judge-provider openai --llm-judge-model gpt-4o-mini
```

Reproduce (heuristic+NLI): `python -m benchmarks.e2e_eval --nli`

## 4. False-Positive Rate

### Streaming False-Halt
0.0% false-halt rate across 20 known-good Wikipedia passages streamed
through `StreamingKernel` (heuristic mode).

Reproduce: `python -m benchmarks.streaming_false_halt_bench`

## 5. RAGTruth & FreshQA

Evaluation infrastructure added in v2.7.0 (`ragtruth_eval.py`,
`freshqa_eval.py`). Both require `datasets` package and GPU for NLI.
Results pending full GPU benchmark run.

```bash
pip install director-ai[nli] datasets
python -m benchmarks.ragtruth_eval --max-samples 500
python -m benchmarks.freshqa_eval --max-samples 500
```

## 6. Honest Limitations

1. **Summarization is weakest**: AggreFact-CNN 68.8%, ExpertQA 59.1%. NLI
   models under-perform on abstractive summarization where surface forms
   diverge from source.
2. **E2E catch rate is mediocre at 46.7%**: heuristic+NLI only. Hybrid
   mode (NLI + LLM judge) is expected to improve this but unvalidated.
3. **ONNX CPU not competitive**: 383 ms/pair. Requires `onnxruntime-gpu`.
4. **Fine-tuned models regressed**: DeBERTa-v3-large fine-tuned on
   hallucination data scored 64.7% — below the pre-trained FactCG 75.8%.
5. **Competitor latencies are estimates**: values marked "~" or "(est.)"
   from published papers, not our measurements.
6. **No hybrid-mode E2E numbers yet**: infrastructure exists, results pending.

## 7. Competitive Position

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
