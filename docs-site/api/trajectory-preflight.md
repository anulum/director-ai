# Trajectory Preflight

Trajectory preflight estimates whether a prompt is likely to trigger a runtime
halt before the main generation starts. `TrajectorySimulator` produces seeded
candidate trajectories and `PredictivePreHaltSteering` turns the aggregate
verdict into a calibrated allow, warn, or halt guard decision.

## Predictive Steering

`PredictivePreHaltSteering` uses three evidence gates:

- halt when the empirical halt probability crosses the calibrated risk
  threshold
- escalate when the upper confidence bound crosses the calibrated threshold
- escalate when there are too few simulations for the configured minimum

The steering payload is tenant-safe. It stores halt probability, confidence
interval bounds, backend recommendation, and trajectory IDs for failed draws; it
does not serialise prompt text or sampled token text.

```python
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.trajectory import PredictivePreHaltSteering

steering = PredictivePreHaltSteering(min_simulations=8)
decision = steering.evaluate(
    verdict,
    risk_envelope=RiskEnvelope(
        action_category="inference_steering",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.9,
    ),
    policy_id="policy.prehalt.regulated",
)

if decision.action == "escalate":
    route_to = decision.recommended_backend
```

## Rollback Hooks

`TrajectoryRollbackManager` registers tenant-safe rollback handles before a
high-risk action executes. A low-risk preflight leaves the handle unused, a warn
or escalation arms the handle for operator review, and a halt executes the hook
once. Repeated execution returns `already_executed`.

```python
from director_ai.core.trajectory import TrajectoryRollbackManager

rollback = TrajectoryRollbackManager()
handle = rollback.register(
    rollback_id="deploy-rollback-20260604-a",
    action_id="deploy-policy-overlay",
    hook=lambda handle, reason: {"rollback_store": "audit-log"},
    evidence_refs=("change:42",),
    metadata={"owner": "safety"},
)
outcome = rollback.evaluate_preflight(handle.rollback_id, verdict)
```

Rollback payloads contain rollback/action IDs, tenant ID, evidence references,
status, and tenant-safe metadata only. They do not include prompt text, sampled
token text, raw action bodies, credentials, or sensor payloads.

Use `InferenceServerHook.steer()` when the decision must affect pre-sampling
logits directly. The hook maps `escalate` to a finite negative bias for the
candidate token and maps `halt` to the same block action used by the coherence
threshold hook.

## Full API

::: director_ai.core.trajectory.simulator.TrajectorySimulator

::: director_ai.core.trajectory.simulator.PreflightVerdict

::: director_ai.core.trajectory.steering.PredictivePreHaltSteering

::: director_ai.core.trajectory.steering.PreHaltSteeringDecision

::: director_ai.core.trajectory.rollback.TrajectoryRollbackManager

::: director_ai.core.trajectory.rollback.RollbackOutcome
