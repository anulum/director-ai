# Conformal & Uncertainty-Aware Routing

`UncertaintyRouter` turns a calibrated hallucination interval into a downstream
action. Where `RiskRouter` routes *inputs* to a scoring backend, this router
acts on the *output*: it consumes the conformal `PredictionInterval` over
hallucination probability and applies documented risk bounds.

| Condition | Action |
|---|---|
| interval upper ≤ `allow_upper` | `allow` (confidently low-risk) |
| interval lower ≥ `reject_lower` | `reject` (confidently high-risk) |
| uncertain and width ≥ `escalate_human_width` (or calibration unreliable) | `escalate_human` |
| uncertain and narrower | `escalate_model` (LLM judge / ensemble) |

The router is side-effect free and deterministic; each `UncertaintyDecision`
records the bounds it used, so the routing rationale is auditable. Dispatching
the action — to a review queue for `escalate_human`, to a stronger model for
`escalate_model` — is the caller's job.

## Online calibration

`ConformalPredictor.add_observation(score, correct_label)` folds one human
verdict into the calibration set and refreshes the conformal quantile, so the
intervals tighten as feedback accumulates. `correct_label=True` marks a correct
(non-hallucinated) response.

```python
from director_ai.core.calibration.conformal import ConformalPredictor
from director_ai.core.routing import UncertaintyRouter

predictor = ConformalPredictor(coverage=0.9, min_samples=30)
for score, correct in feedback_history:
    predictor.add_observation(score, correct_label=correct)

router = UncertaintyRouter(allow_upper=0.2, reject_lower=0.8, escalate_human_width=0.5)
decision = router.route(predictor.predict(coherence_score))
if decision.action == "escalate_human":
    review_queue.submit(...)
elif decision.action == "escalate_model":
    llm_judge.adjudicate(...)
```

## ProductionGuard integration

`ProductionGuard` wires both together. After `enable_calibration()`, call
`enable_uncertainty_routing()`; every `check()` then populates
`GuardResult.uncertainty_action` from the conformal interval. Until calibration
is reliable, the action is `escalate_human` — uncertainty defers to a person.

```python
guard.enable_calibration(alpha=0.1)
guard.enable_uncertainty_routing()
result = guard.check(prompt, response)
result.uncertainty_action  # "allow" | "reject" | "escalate_human" | "escalate_model"
```

## Full API

::: director_ai.core.routing.uncertainty_router.UncertaintyDecision

::: director_ai.core.routing.uncertainty_router.UncertaintyRouter

::: director_ai.core.calibration.conformal.ConformalPredictor
