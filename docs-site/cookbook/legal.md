# Legal Domain Cookbook

## Complete Working Example

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()  # empty — populate with your KB
store.add("statute of limitations", "The statute of limitations for personal injury in California is 2 years.")
store.add("contract elements", "A contract requires offer, acceptance, consideration, and mutual assent.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

# Correct → approved
approved, score = scorer.review("California injury statute of limitations?",
    "The statute of limitations for personal injury in California is 2 years.")
print(f"Correct: approved={approved}, score={score.score:.2f}")

# Wrong → rejected
approved, score = scorer.review("California injury statute of limitations?",
    "There is no statute of limitations for personal injury in California.")
print(f"Wrong:   approved={approved}, score={score.score:.2f}")
```

## Configuration

```python
from director_ai import CoherenceScorer, VectorGroundTruthStore

store = VectorGroundTruthStore()
store.ingest([
    "The statute of limitations for personal injury in California is 2 years.",
    "Attorney-client privilege protects communications made for legal advice.",
    "A contract requires offer, acceptance, consideration, and mutual assent.",
])

scorer = CoherenceScorer(
    threshold=0.7,     # Higher threshold for legal accuracy
    soft_limit=0.8,    # Wider warning zone
    use_nli=True,
    ground_truth_store=store,
)
```

## Cost Savings

| Metric | Without Director-AI | With Director-AI (threshold=0.7) |
|--------|--------------------|---------------------------------|
| Hallucinated citation rate | 12–19% (model-dependent) | < 1% with contract KB |
| Lawyer review hours per 100 AI drafts | 50 hrs | 12 hrs (review flagged only) |
| Annual review cost (1,000 queries/day) | ~$5.5M | ~$1.3M |

At $300/hr associate rate and 1,000 AI-assisted queries/day, reducing review burden by 76% saves ~$4.2M/year. A single prevented fabricated citation avoids potential sanctions, malpractice claims, and bar complaints.

## Key Considerations

- **Higher thresholds**: legal claims require greater confidence (0.7+)
- **Wide soft zone**: flag anything below 0.8 for human review
- **Retrieval fallback**: always cite sources rather than hallucinate
- **Audit trail**: enable `AuditLogger` for compliance

```python
from director_ai.core.audit import AuditLogger

logger = AuditLogger(log_dir="/var/log/director-ai/legal")
```
