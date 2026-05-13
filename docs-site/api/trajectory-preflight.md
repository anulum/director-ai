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

## Full API

::: director_ai.core.trajectory.simulator.TrajectorySimulator

::: director_ai.core.trajectory.simulator.PreflightVerdict

::: director_ai.core.trajectory.steering.PredictivePreHaltSteering

::: director_ai.core.trajectory.steering.PreHaltSteeringDecision
