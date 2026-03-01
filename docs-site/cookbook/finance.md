# Finance Domain Cookbook

## Complete Working Example

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("savings APY", "Our savings account APY is 4.25% as of February 2026.")
store.add("FDIC", "FDIC insurance covers up to $250,000 per depositor per bank.")
store.add("wire transfer", "Wire transfers take 1-3 business days.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

# Correct → approved
approved, score = scorer.review("What is the savings APY?",
    "The current savings account APY is 4.25%.")
print(f"Correct: approved={approved}, score={score.score:.2f}")

# Wrong → rejected
approved, score = scorer.review("What is the FDIC limit?",
    "FDIC covers up to $500,000 per depositor.")
print(f"Wrong:   approved={approved}, score={score.score:.2f}")
```

## Configuration

```python
scorer = CoherenceScorer(
    threshold=0.65,
    soft_limit=0.75,
    use_nli=True,
    ground_truth_store=store,
    cache_size=4096,   # High cache for repeated product queries
    cache_ttl=3600,    # 1-hour cache for stable financial facts
)
```

## Knowledge Base

```python
store = VectorGroundTruthStore(auto_index=False)
store.ingest([
    "Our savings account APY is 4.25% as of February 2026.",
    "Wire transfers take 1-3 business days.",
    "FDIC insurance covers up to $250,000 per depositor per bank.",
    "Minimum balance for premium checking is $5,000.",
])
```

## Compliance Pattern

```python
from director_ai.core.audit import AuditLogger
from director_ai.core.policy import Policy

# Audit all interactions for compliance
audit = AuditLogger(log_dir="/var/log/director-ai/finance")

# Policy: block responses mentioning specific stock recommendations
policy = Policy(rules=[
    {"pattern": r"(buy|sell|short)\s+(stock|shares)", "action": "reject"},
])
```

## Key Considerations

- **Regulatory compliance**: audit all rejections and approvals
- **Rate data freshness**: financial data changes — set appropriate `cache_ttl`
- **Disclaimer on all outputs**: regulatory requirement for financial advice
- **Multi-tenant isolation**: use `TenantRouter` for different product lines
