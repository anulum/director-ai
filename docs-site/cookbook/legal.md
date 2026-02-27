# Legal Domain Cookbook

## Configuration

```python
from director_ai import CoherenceScorer, VectorGroundTruthStore

store = VectorGroundTruthStore(auto_index=False)
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

## Key Considerations

- **Higher thresholds**: legal claims require greater confidence (0.7+)
- **Wide soft zone**: flag anything below 0.8 for human review
- **Retrieval fallback**: always cite sources rather than hallucinate
- **Audit trail**: enable `AuditLogger` for compliance

```python
from director_ai.core.audit import AuditLogger

logger = AuditLogger(log_dir="/var/log/director-ai/legal")
```
