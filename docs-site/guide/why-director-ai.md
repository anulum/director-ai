# Why Director-AI

## The Streaming Problem

Every major LLM provider defaults to streaming. OpenAI, Anthropic, Google — they all send tokens as they're generated. Users see the response character by character.

Post-hoc guardrails check *after* generation completes. By then, the user already read the hallucination. The damage is done: a wrong medication dosage displayed for 3 seconds, a fabricated legal citation copied into a brief, an incorrect refund policy quoted to an angry customer.

The industry standard — generate first, check later — is a UX failure.

## What Director-AI Does Differently

Director-AI scores coherence **as tokens arrive**, not after the full response is assembled.

**Token-level halt.** `StreamingKernel` evaluates every N tokens against your knowledge base. If coherence drops below threshold mid-stream, generation stops immediately. The user never sees the hallucinated content.

**Dual-entropy scoring.** Two independent signals:

- **H_logical** — NLI contradiction detection via DeBERTa (0.4B params). Catches logical inconsistencies between the response and your facts.
- **H_factual** — RAG retrieval against your knowledge base. Catches claims that have no supporting evidence.

The final score combines both: `coherence = 1 - (0.6 * H_logical + 0.4 * H_factual)`.

**Evidence on rejection.** Every halt includes the specific KB chunks that contradicted the response. No black-box "this was flagged" — your users (or your QA team) see exactly *why*.

**0.4B parameters, sub-millisecond latency.** FactCG-DeBERTa-v3-Large runs at 0.5 ms/pair on an L40S (FP16, batch=32). No API calls, no metering, no rate limits.

## When NOT to Use Director-AI

Director-AI solves one problem: **factual coherence** — does the LLM output match your ground truth?

It does **not** handle:

| Problem | Use Instead |
|---------|-------------|
| Toxicity / hate speech | [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/), [LLM-Guard](https://llm-guard.com/) |
| Prompt injection | [Rebuff](https://github.com/protectai/rebuff), [LLM-Guard](https://llm-guard.com/) |
| PII leakage | [Presidio](https://github.com/microsoft/presidio), [LLM-Guard](https://llm-guard.com/) |
| Content moderation | [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation), [Llama Guard](https://ai.meta.com/blog/purple-llama-llama-guard/) |
| Code safety | [Semgrep](https://semgrep.dev/), [Snyk Code](https://snyk.io/product/snyk-code/) |

You can (and should) combine Director-AI with these tools. Director-AI guards facts; the others guard behaviour.

## Decision Matrix

| Your Situation | Recommendation |
|----------------|----------------|
| RAG chatbot with a knowledge base | Director-AI with `VectorGroundTruthStore` — [KB Ingestion guide](kb-ingestion.md) |
| Streaming LLM responses to users | Director-AI `StreamingKernel` — [Streaming guide](../guide/streaming.md) |
| LLM agent making multi-step decisions | Director-AI `CoherenceAgent` — [API reference](../api/agent.md) |
| Customer support bot with product facts | Director-AI with domain-specific KB — [Support cookbook](../cookbook/customer-support.md) |
| Medical / legal / finance compliance | Director-AI with high threshold (0.7+) — domain [cookbooks](../cookbook/medical.md) |
| Toxicity filtering only | NeMo Guardrails or LLM-Guard instead |
| Prompt injection defence only | Rebuff or LLM-Guard instead |

## Cost Comparison

| System | Cost per 1K calls | Latency | Local/Offline |
|--------|-------------------|---------|---------------|
| **Director-AI (NLI mode)** | **$0** | **0.5 ms** (L40S) | **Yes** |
| Director-AI (hybrid + GPT-4o-mini) | $0.07 | 2.3 s | No |
| Director-AI (hybrid + Claude Sonnet) | $1.40 | 14.2 s | No |
| GPT-4o as judge | $1.16 | ~2 s | No |
| Claude Haiku 4.5 as judge | $0.37 | ~1.5 s | No |
| GuardrailsAI (LLM-as-judge) | LLM cost | 2.26 s | No |
| SelfCheckGPT (multi-call) | 3-5x LLM cost | 5-10 s | No |

NLI-only mode is free, fast, and fully offline. Add an LLM judge only if you need the 90.7% hybrid catch rate — and even then, GPT-4o-mini matches Claude at 6x lower cost.

---

*Next: [Quickstart](../quickstart.md) — score your first response in 2 minutes.*
