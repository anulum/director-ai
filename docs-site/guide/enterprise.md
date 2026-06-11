# Enterprise Modules

Three modules ship in the source-available Advanced & Labs tier (BUSL-1.1): free
to evaluate, with a commercial licence required for production use.

## TenantRouter

Isolates scorer instances per tenant. Each tenant gets its own `CoherenceScorer` with independent thresholds, knowledge bases, and caches.

```python
from director_ai.core import TenantRouter
from director_ai import CoherenceScorer

router = TenantRouter()
router.register("acme", CoherenceScorer(threshold=0.7, use_nli=True))
router.register("beta", CoherenceScorer(threshold=0.5))

scorer = router.get("acme")
approved, score = scorer.review(prompt, response)
```

## Policy

Declarative rules evaluated before scoring. Block prompts by pattern, enforce minimum thresholds per domain, tag violations.

```python
from director_ai.core import Policy

policy = Policy(
    name="medical",
    min_threshold=0.30,
    blocked_patterns=[r"(?i)prescri(be|ption)"],
    require_nli=True,
)
violations = policy.check(prompt, response, score)
```

## AuditLogger

SQLite-backed audit trail. Logs every review with prompt hash, score, approved/rejected, violations, and timestamp.

```python
from director_ai.core import AuditLogger

logger = AuditLogger(db_path="audit.db")
logger.log(prompt, response, score, approved, violations=[])
```

## Lazy Loading

Enterprise modules are lazy-loaded (v2.2.0+). `import director_ai` does not pull in tenant/policy/audit until you access them. No performance cost for users who don't need multi-tenancy.

```python
import director_ai  # fast — enterprise modules not loaded

# Only loaded on first access:
router = director_ai.TenantRouter()
```

## When You Need a Commercial Licence

These modules are BUSL-1.1. The Apache-2.0 core is always free; the table below
covers the **Advanced & Labs** tier only.

| Scenario | License |
|----------|---------|
| Evaluation / prototyping | BUSL-1.1 (free) |
| Research / academic, non-production | BUSL-1.1 (free) |
| Non-production internal use | BUSL-1.1 (free) |
| Production internal tool | Commercial (Indie+) |
| Production SaaS | Commercial (Pro+) |
| Multi-tenant production with SLA | Commercial (Enterprise) |

See [Licensing](../licensing.md) for pricing.
