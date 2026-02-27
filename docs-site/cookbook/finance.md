# Finance Domain Cookbook

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
