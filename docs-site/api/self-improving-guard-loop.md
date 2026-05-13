# Self-Improving Guard Loop

The self-improving guard loop turns reviewed feedback into auditable proposals.
It does not train, submit jobs, change thresholds, or promote models by itself.
Production changes require a separate operator-approved deployment step.

## Proposal Gates

`SelfImprovingGuardLoop` enforces the control boundary:

- every feedback row must carry an `event_id` and `reviewer_id`
- manifests serialise event references and label counts, not prompt or response
  text
- calibration updates require enough feedback and a narrow confidence interval
- LoRA jobs are proposal-only and require held-out improvement plus rollback ID
- dataset URIs with embedded credentials are rejected

```python
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.self_evolving import SelfImprovingGuardLoop

loop = SelfImprovingGuardLoop(
    store=feedback_store,
    risk_envelope=RiskEnvelope(
        action_category="training",
        reversibility="costly",
        domain="regulated",
        calibrated_threshold=0.45,
        no_go_threshold=0.8,
    ),
    policy_id="policy.self_improving.regulated",
)

proposal = loop.propose_calibration_update(
    source_ref="feedback://recent-reviewed",
    current_threshold=0.55,
    candidate_threshold=0.58,
    confidence_low=0.51,
    confidence_high=0.61,
    rollback_id="threshold-profile-20260513-a",
)

approved = loop.approve(proposal, approval_id="review-20260513-002")
payload = loop.release(approved)
```

`release()` returns the approved proposal payload for an external deployment
controller. It does not mutate runtime configuration.

## Full API

::: director_ai.core.self_evolving.guard_loop.SelfImprovingGuardLoop

::: director_ai.core.self_evolving.guard_loop.ReviewedFeedbackManifest

::: director_ai.core.self_evolving.guard_loop.GuardLoopProposal
