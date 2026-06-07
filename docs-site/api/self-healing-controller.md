# Self-Healing Threshold Controller

Adapt the operating threshold from live feedback **safely**: deploy a new
threshold only after it beats the current one on a held-out split, and roll it
back automatically when a deployed change later regresses. Every decision is
audited — there is no blind self-mutation.

## How it relates to online calibration

[`OnlineCalibrator`](online-calibration.md) computes a recommended threshold from
the durable feedback store. The self-healing controller closes that loop: it
takes labelled outcomes, proposes an update, **gates the deploy on a holdout
split**, and provides **auto-rollback** — the safety machinery that lets
calibration run continuously without risking a bad self-update reaching
production.

## How it works

1. Feed labelled outcomes (`score`, ground-truth `grounded`).
2. `propose()` splits the accumulated window deterministically into a *support*
   split (where the best threshold is found — the "adaptation") and a *holdout*
   split (where it is validated — the "query"). The candidate is deployed **only
   if it strictly lowers the holdout error**; otherwise it is rejected. Either
   way the window is consumed.
3. `evaluate_regression(fresh_outcomes)` compares the active threshold against the
   one it replaced; if the active policy is worse it rolls back.

The support/holdout split is by observation index (no randomness), so the
controller is fully reproducible. The error metric weights false halts and missed
hallucinations independently.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.self_healing import LabelledOutcome

guard = ProductionGuard(DirectorConfig())
controller = guard.self_healing  # seeded at the configured coherence threshold

# Feed reviewed outcomes as they arrive.
for score, grounded in recent_reviewed_decisions:
    controller.observe(LabelledOutcome(score=score, grounded=grounded))

update = controller.propose()
if update.action == "accept":
    apply_threshold(controller.threshold)   # the host applies it
print(update.action, controller.threshold)  # e.g. "accept" 0.62

# Later, on fresh reviewed data, guard against regression:
rollback = controller.evaluate_regression(newer_outcomes)
if rollback.action == "rollback":
    apply_threshold(controller.threshold)   # restored to the prior policy
```

The controller never mutates the guard's configured threshold; the host applies
`controller.threshold` where it sees fit.

## Decisions

`propose()` and `evaluate_regression()` return a `PolicyUpdate` whose `action` is
one of:

| Action | Meaning |
|--------|---------|
| `accept` | A better threshold was deployed (holdout error fell). |
| `reject` | The candidate did not improve the holdout; threshold unchanged. |
| `rollback` | A deployed update regressed and was restored to its predecessor. |
| `stable` | The deployed update is holding up; no change. |
| `insufficient_data` | Fewer than `min_samples` outcomes; threshold unchanged. |
| `no_prior` | Nothing has been deployed yet, so there is nothing to roll back. |

`PolicyUpdate.changed` is `True` only for `accept` and `rollback`. `audit()`
returns the full ordered trail of decisions as serialisable records.

## Tuning

`TuningConfig` controls the loop:

```python
from director_ai.core.self_healing import SelfHealingThresholdController, TuningConfig

controller = SelfHealingThresholdController(
    initial_threshold=0.6,
    config=TuningConfig(
        holdout_fraction=0.34,          # fraction held out for validation
        false_halt_weight=1.0,          # penalty for halting a grounded answer
        missed_hallucination_weight=3.0,  # penalty for approving a hallucination
        min_samples=20,                 # minimum window before proposing
        regression_tolerance=0.0,       # rollback if worse than predecessor by more
    ),
)
```

Raising `missed_hallucination_weight` biases the controller toward stricter
thresholds (fewer approved hallucinations); raising `false_halt_weight` biases it
toward more permissive thresholds (fewer blocked good answers).

## Notes

- Pure deterministic control logic; no numeric hot path, so no accelerated
  backend is warranted.
- Each `guard.self_healing` is independent and persists across calls so outcomes
  accumulate; use one controller per policy domain.
