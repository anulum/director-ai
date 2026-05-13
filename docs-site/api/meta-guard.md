# Recursive Meta-Guard

`MetaGuard` monitors the guardrail's own scoring decisions for drift. In
production, recursive threshold changes should be gated so adversarial traffic
cannot steer the guard by flooding one tenant, one repeated prompt shape, or
unlabelled drift windows.

`MetaGuardProductionPolicy` turns the meta-guard into a guarded production
option:

- Page-Hinkley, Brier, and action-rate alarms are still reported;
- threshold adjustment is blocked when one tenant dominates the window;
- threshold adjustment is blocked when one prompt hash dominates the window;
- threshold adjustment is blocked when the labelled fraction is below the
  configured floor;
- blocked windows are observe-only and do not mutate the `ThresholdAdjuster`.

```python
from director_ai.core.meta_guard import (
    DecisionLog,
    MetaAnalyzer,
    MetaGuard,
    MetaGuardProductionPolicy,
    ThresholdAdjuster,
    ThresholdBundle,
)

guard = MetaGuard(
    log=DecisionLog(),
    analyzer=MetaAnalyzer(reference_mean=0.25, min_window=64),
    adjuster=ThresholdAdjuster(
        initial=ThresholdBundle(warn_threshold=0.35, halt_threshold=0.70),
        hysteresis_strikes=2,
    ),
    production_policy=MetaGuardProductionPolicy(
        min_labelled_fraction=0.20,
        max_single_tenant_fraction=0.50,
        max_duplicate_prompt_fraction=0.40,
    ),
)

verdict = guard.record(
    prompt="operator-reviewed prompt",
    score=0.82,
    action="halt",
    ground_truth=1.0,
    tenant_id="tenant-a",
)

if verdict.production.blocked:
    print(verdict.production.block_reason)
```

Use `verdict.production` for audit telemetry. A blocked production decision
means drift was detected but recursive adjustment was not considered safe for
automatic application.

::: director_ai.core.meta_guard.guard.MetaGuard

::: director_ai.core.meta_guard.guard.MetaGuardProductionPolicy

::: director_ai.core.meta_guard.guard.ProductionMetaGuardDecision
