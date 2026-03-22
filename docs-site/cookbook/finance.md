# Finance Domain Cookbook

## Complete Working Example

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()  # empty — populate with your KB
store.add("savings APY", "Our savings account APY is 4.25% as of February 2026.")
store.add("FDIC", "FDIC insurance covers up to $250,000 per depositor per bank.")
store.add("wire transfer", "Wire transfers take 1-3 business days.")

scorer = CoherenceScorer(threshold=0.30, ground_truth_store=store)

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
    threshold=0.30,   # Measured on FinanceBench: 0% FPR at t≤0.30
    soft_limit=0.35,
    use_nli=True,
    ground_truth_store=store,
    cache_size=4096,   # High cache for repeated product queries
    cache_ttl=3600,    # 1-hour cache for stable financial facts
)
```

## Knowledge Base

```python
store = VectorGroundTruthStore()
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
audit = AuditLogger(path="/var/log/director-ai/finance")

# Policy: block responses mentioning specific stock recommendations
policy = Policy(
    patterns=[r"(buy|sell|short)\s+(stock|shares)"],
    forbidden=["stock recommendation", "investment advice"],
)
```

## Compliance Cost Avoidance

| Risk | Exposure Without Director-AI | With Director-AI |
|------|------------------------------|-----------------|
| Wrong product terms quoted to customer | CFPB/FCA fine ($5K–$1M per violation) | Caught mid-stream, never reaches customer |
| Hallucinated interest rate or fee | Customer dispute + regulatory review | KB-verified before display |
| Unauthorized investment advice | SEC/FINRA action ($50K–$10M) | Policy engine blocks + audit trail |

At 5,000 customer interactions/day, a 0.1% hallucination rate means 5 wrong answers daily. Over a year, that's 1,825 potential compliance incidents. With Director-AI, the catch rate reduces this to < 20/year (assuming high catch rate with KB + NLI at threshold=0.30).

## Key Considerations

- **Regulatory compliance**: audit all rejections and approvals
- **Rate data freshness**: financial data changes — set appropriate `cache_ttl`
- **Disclaimer on all outputs**: regulatory requirement for financial advice
- **Multi-tenant isolation**: use `TenantRouter` for different product lines
