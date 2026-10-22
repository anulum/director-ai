<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- DIRECTOR-AI — docs-site/guide/enterprise.md -->

# Enterprise Modules

Advanced/Labs uses are subject to the [permanent Use Limitation](../licensing.md); text and data mining and machine-learning uses require prior written permission.

Three modules ship in the source-available Advanced & Labs tier
(LicenseRef-BUSL-1.1-ANULUM-TDM). Non-production use, including evaluation, is
subject to the permanent Use Limitation above. Production use requires a
commercial licence.

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

These modules are LicenseRef-BUSL-1.1-ANULUM-TDM. The Apache-2.0 core is always free; the table below
covers the **Advanced & Labs** tier only. Every non-production row remains
subject to the permanent Use Limitation above.

| Scenario | License |
|----------|---------|
| Evaluation / prototyping | LicenseRef-BUSL-1.1-ANULUM-TDM (free) |
| Research / academic, non-production | LicenseRef-BUSL-1.1-ANULUM-TDM (free) |
| Non-production internal use | LicenseRef-BUSL-1.1-ANULUM-TDM (free) |
| Production internal tool | Commercial (Indie+) |
| Production SaaS | Commercial (Pro+) |
| Multi-tenant production with SLA | Commercial (Enterprise) |

See [Licensing](../licensing.md) for pricing.
