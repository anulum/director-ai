# Director-AI vs Prompt Engineering

Prompt engineering tells the LLM "please don't hallucinate." Director-AI measures whether it did.

## The Core Difference

| | Prompt Engineering | Director-AI |
|---|---|---|
| **Mechanism** | Instructions in the system prompt | NLI model + RAG retrieval scoring every response |
| **Guarantee** | Probabilistic — the LLM may ignore instructions | Deterministic — scores are computed, thresholds enforced |
| **Measurable** | No — you can't measure compliance rate without a separate eval | Yes — every response gets a coherence score with evidence |
| **Streaming** | No halt mechanism — hallucinated tokens reach the user | Token-level halt — generation stops mid-stream before the user sees it |
| **Auditable** | No artefact — the prompt is an instruction, not a measurement | Per-response scores by default, with per-claim verdicts available via `VerifiedScorer` |

## What Prompt Engineering Cannot Do

### 1. Enforce a threshold

You can tell GPT-4 "only respond with information from the provided documents." It will comply most of the time. But there is no threshold. You cannot say "block any response with coherence below 0.3" because the LLM has no coherence metric. It either follows the instruction or it doesn't, and you find out after the fact.

Director-AI scores every response against your knowledge base. If the score falls below your threshold, the response is blocked — with evidence showing which claims failed.

### 2. Halt a stream mid-generation

Prompt engineering has no mechanism to stop token generation once it starts drifting. If GPT-4 begins a response with accurate information and then fabricates a statistic in sentence three, all tokens reach the user.

Director-AI's streaming runtime can score tokens as they arrive. If coherence drops mid-stream, generation halts before the response is fully delivered.

```python
from director_ai import VoiceGuard

guard = VoiceGuard(
    facts={"refund": "30-day refund policy, no questions asked."},
    prompt="What is the refund policy?",
    threshold=0.3,
    hard_limit=0.25,
    use_nli=True,
)

approved_tokens = []
for token in llm.stream("What is the refund policy?"):
    result = guard.feed(token)
    if result.halted:
        approved_tokens.append(f" {result.recovery_text}")
        break
    approved_tokens.append(token)
```

### 3. Produce auditable evidence

"I told the model to only use provided sources" is not an audit trail. When a regulator asks "how do you verify that your AI system's outputs are factually correct?", a system prompt is not an answer.

Director-AI produces per-response evidence on the standard review path: coherence score plus supporting evidence. For claim-level audits, `VerifiedScorer` adds per-claim verdicts (supported/contradicted/fabricated/unverifiable), matched source sentences, and traceability scores. This evidence can be logged, queried, and presented in compliance reports.

### 4. Catch subtle factual errors

Prompt engineering relies on the LLM's own judgment to detect errors in its own output. The same model that generated "data is retained for 365 days" (when the docs say 90 days) is the same model you're asking to spot the error. It often can't.

Director-AI uses a separate NLI model (FactCG-DeBERTa, 0.4B params) specifically trained to detect entailment and contradiction between pairs of texts. It catches number substitutions, negation flips, and fabricated entities that the generating LLM misses.

## Where Prompt Engineering Still Wins

Prompt engineering is not useless. It is the right tool for:

- **Tone and style** — "respond formally" or "use British English." Director-AI does not evaluate style.
- **Response format** — "respond in JSON with these keys." Use Director-AI's `verify_json()` or `verify_tool_call()` for structural validation, but simple format instructions work fine.
- **Task scoping** — "only answer questions about our product." Director-AI checks factual coherence, not topic relevance.
- **Persona** — "you are a helpful customer support agent." Director-AI has no opinion on persona.

The optimal setup uses both: prompt engineering shapes the response, Director-AI verifies it.

## Layered Architecture

```
User query
    │
    ▼
┌──────────────────────────┐
│  System prompt            │  ← Prompt engineering: tone, format, scope
│  + few-shot examples      │
│  + RAG context injection  │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  LLM generates response   │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  Director-AI scoring      │  ← Factual verification: NLI + RAG
│  VoiceGuard / review      │
│  VerifiedScorer (optional)│
│  Halt if below threshold  │
└──────────┬───────────────┘
           │
           ▼
     User sees response
     (only if approved)
```

Prompt engineering is layer 1 (best-effort shaping). Director-AI is layer 2 (measured enforcement). Neither replaces the other.

## Cost of Not Measuring

A customer support chatbot with good prompt engineering might hallucinate on 2–5% of responses. On 10,000 daily conversations, that is 200–500 wrong answers per day. You will not know which ones without a measurement layer.

Director-AI with a 0.30 threshold catches the majority of these. The ones it catches get blocked with evidence. The ones it misses are still logged with scores, so you can review low-confidence responses.

| Setup | Hallucinations reaching users (est.) | Audit trail |
|-------|--------------------------------------|-------------|
| Prompt engineering only | 2–5% of responses | None |
| Prompt engineering + Director-AI (heuristic) | <1% | Full — per-response scores |
| Prompt engineering + Director-AI (NLI) | <0.5% | Full — per-response scores, with per-claim verdicts via `VerifiedScorer` |

These numbers are estimates based on published RAG hallucination rates and Director-AI's measured catch rate on internal benchmarks. Your domain will vary — run `director-ai bench` on your own data.

## Migration Path

Already using prompt engineering? Adding Director-AI takes three lines:

```python
from director_ai import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("refund", "30-day refund policy, no questions asked.")

scorer = CoherenceScorer(threshold=0.3, ground_truth_store=store, use_nli=True)

# Wrap your existing LLM call
approved, score = scorer.review(user_prompt, llm_response)
if not approved:
    llm_response = "I need to verify that information. Let me check."
```

No changes to your prompts, your RAG pipeline, or your LLM provider. Director-AI sits between the LLM and the user.

---

*Next: [Voice AI Integration](voice-ai.md) — halt hallucinations before they're spoken.*
