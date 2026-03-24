# Meta-Confidence & Contradiction Tracking

Know when to trust the guardrail — and detect when your AI contradicts itself.

## Verdict Confidence

Every `CoherenceScore` now includes `verdict_confidence`: a measure of how confident the guardrail is in its own approval/rejection decision. No other hallucination guardrail answers this question.

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(use_nli=True)
approved, score = scorer.review(prompt, response)

print(f"Approved: {approved}")
print(f"Score: {score.score:.3f}")
print(f"Verdict confidence: {score.verdict_confidence:.3f}")
print(f"Signal agreement: {score.signal_agreement:.3f}")
```

### How It Works

Three orthogonal signals are combined:

1. **Margin** — distance between the score and the threshold. Score 0.51 at threshold 0.50 = margin 0.01 = low confidence. Score 0.90 at threshold 0.50 = margin 0.40 = high confidence.

2. **Signal agreement** — do the logical and factual divergence signals agree? When `h_logical` says "fine" but `h_factual` says "hallucination", the verdict is less trustworthy.

3. **NLI model confidence** (when available) — the softmax entropy of the NLI prediction. High entropy = the model is uncertain about the entailment label.

The combined confidence is `min(margin, signal_agreement, nli_confidence)` — the weakest signal determines the verdict confidence.

### Routing Low-Confidence Results

```python
approved, score = scorer.review(prompt, response)

if score.verdict_confidence < 0.3:
    # Low confidence — route to human review
    send_to_human_queue(prompt, response, score)
elif approved:
    # High confidence approval — serve to user
    serve_response(response)
else:
    # High confidence rejection — block
    serve_fallback()
```

### CoherenceScore Fields

| Field | Type | Description |
|-------|------|-------------|
| `verdict_confidence` | `float \| None` | Combined confidence in the verdict [0, 1] |
| `nli_model_confidence` | `float \| None` | NLI softmax entropy confidence |
| `signal_agreement` | `float \| None` | Agreement between h_logical and h_factual |

---

## Contradiction Tracking

In multi-turn conversations, LLMs can contradict themselves across turns. Director-AI now tracks pairwise contradictions between all turns in a session.

```python
from director_ai import CoherenceScorer, ConversationSession

scorer = CoherenceScorer(use_nli=True)
session = ConversationSession()

# Turn 1
approved, score = scorer.review(
    "What is our return policy?",
    "We offer a 30-day return policy.",
    session=session,
)

# Turn 2
approved, score = scorer.review(
    "Can I return this after 60 days?",
    "Yes, our 60-day return policy covers this.",  # contradicts turn 1
    session=session,
)

print(f"Contradiction index: {score.contradiction_index:.3f}")
# High value = the AI is contradicting prior statements
```

### How It Works

After each turn, the new response is NLI-scored against every prior response individually (not concatenated). This builds a pairwise contradiction matrix. The `contradiction_index` on `CoherenceScore` is the maximum pairwise divergence — the worst contradiction in the conversation.

### Contradiction Report

```python
report = session.get_contradiction_report()
print(f"Contradiction index: {report.contradiction_index:.3f}")
print(f"Trend: {report.trend:+.3f}")  # positive = getting worse

if report.worst_pair:
    print(f"Worst pair: turn {report.worst_pair.turn_a} vs {report.worst_pair.turn_b}")
    print(f"Divergence: {report.worst_pair.divergence:.3f}")
```

### Cost

O(N) NLI calls per turn, where N is the number of prior turns. With `max_turns=20` and batched NLI at ~2ms/pair on GPU, that's ~40ms per turn — acceptable for multi-turn use cases that already have 200ms+ latency.
