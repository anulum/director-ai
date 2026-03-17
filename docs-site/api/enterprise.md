# Enterprise API

Multi-tenant scoring isolation, declarative policy rules, and audit logging. These modules are lazy-loaded — importing `director_ai` does not pull them in until accessed.

```python
from director_ai.enterprise import TenantRouter, Policy, AuditLogger
```

## TenantRouter

Isolates scoring configuration per tenant. Each tenant gets its own `CoherenceScorer` instance with independent thresholds, knowledge bases, and caching.

```python
from director_ai.enterprise import TenantRouter

router = TenantRouter()
router.register("tenant_a", threshold=0.7, use_nli=True)
router.register("tenant_b", threshold=0.5, use_nli=False)

scorer = router.get_scorer("tenant_a")
approved, score = scorer.review(query, response)
```

## Policy

Declarative rule engine for content filtering. Runs before coherence scoring.

```python
from director_ai.enterprise import Policy

policy = Policy(rules=[
    {"pattern": r"(buy|sell|short)\s+(stock|shares)", "action": "reject"},
    {"pattern": r"\b(SSN|social security)\b", "action": "redact"},
])

result = policy.evaluate(response_text)
if result.rejected:
    print(f"Policy violation: {result.rule}")
```

## AuditLogger

SQLite-backed audit logging for compliance. Records every review decision with full context.

```python
from director_ai.enterprise import AuditLogger

logger = AuditLogger(log_dir="/var/log/director-ai/audit")
logger.log(query, response, score, approved=True)

# Query audit trail
entries = logger.query(tenant_id="tenant_a", since="2026-01-01")
```

## License Matrix

| Use Case | License Required |
|----------|-----------------|
| Open-source project | AGPL-3.0 (free) |
| Internal tools | AGPL-3.0 (free) |
| SaaS product | Commercial license |
| Proprietary embedding | Commercial license |

See [Licensing](../licensing.md) for pricing and terms.

## Full API

::: director_ai.core.tenant.TenantRouter

::: director_ai.core.safety.audit.AuditLogger
