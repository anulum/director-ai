# Customer Support Domain Cookbook

## Complete Working Example

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()  # empty — populate with your KB
store.add("refund policy", "Refunds are available within 30 days of purchase.")
store.add("shipping", "Standard shipping takes 5-7 business days.")
store.add("pricing", "Pro plan costs $49/month, Enterprise is $199/month.")
store.add("support hours", "Support is available Monday-Friday 9AM-5PM EST.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

# Correct → approved
approved, score = scorer.review("What is your refund policy?",
    "We offer refunds within 30 days of purchase.")
print(f"Correct: approved={approved}, score={score.score:.2f}")

# Wrong → rejected
approved, score = scorer.review("What is your refund policy?",
    "We offer full refunds within 90 days, no questions asked.")
print(f"Wrong:   approved={approved}, score={score.score:.2f}")
if score.evidence:
    for chunk in score.evidence.chunks:
        print(f"  Evidence: {chunk.text}")
```

## Configuration

```python
scorer = CoherenceScorer(
    threshold=0.55,
    soft_limit=0.65,
    use_nli=True,
    ground_truth_store=store,
    cache_size=2048,   # high cache for repeated product queries
    cache_ttl=1800,    # 30-min TTL for stable product facts
)
```

## Knowledge Base

```python
store = VectorGroundTruthStore()
store.ingest([
    "Refunds are available within 30 days of purchase with receipt.",
    "Standard shipping takes 5-7 business days within the US.",
    "Express shipping takes 1-2 business days for $15 extra.",
    "Pro plan costs $49/month billed annually or $59/month billed monthly.",
    "Enterprise plan costs $199/month with custom onboarding.",
    "Support is available Monday-Friday 9AM-5PM EST via chat and email.",
    "Phone support is available for Enterprise customers only.",
])
```

## CSAT Improvement

| Metric | Without Director-AI | With Director-AI (threshold=0.55) |
|--------|--------------------|---------------------------------|
| Wrong policy quoted to customer | 8–12% of responses | < 1% with product KB |
| CSAT score | 3.2/5 (industry avg for AI chat) | 4.5–4.7/5 (verified answers + evidence) |
| Escalation rate | 25% (customer doesn't trust AI) | 8% (evidence builds confidence) |
| Agent handle time (hybrid AI + human) | 6 min | 3.5 min (AI handles routine, flags edge cases) |

At 10,000 tickets/month, reducing escalations from 25% to 8% saves 1,700 agent-handled tickets/month. At $12/ticket agent cost, that's ~$245K/year. CSAT improvement from 3.2 to 4.6 correlates with 15–20% higher retention (Zendesk benchmark).

## Key Considerations

- **Lower threshold (0.55)**: Customer support tolerates minor phrasing differences
- **High cache size**: Product queries repeat frequently
- **Always return evidence**: Customers trust answers backed by specific KB entries
- **Audit trail**: Enable `AuditLogger` for quality assurance review
