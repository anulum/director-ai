# Guard Control

`GuardDecision`, `RiskEnvelope`, and `NoGoPolicy` are the shared policy
contracts used by high-risk verifier adapters. They keep risk classification,
calibrated thresholds, evidence references, and policy escalation separate from
raw tenant payloads.

## Irreversibility Forecasting

`NoGoPolicy` runs the irreversibility forecaster by default when both conditions
hold:

- the upstream decision risk score is at or above
  `risk_envelope.calibrated_threshold`
- the decision carries a tenant-safe action label or action sequence

The policy blocks when the conservative lower confidence bound of the forecast
crosses `irreversible_threshold`. This prevents a point estimate alone from
blocking high-risk actions while still stopping action sequences whose
irreversibility remains high under interval uncertainty.

For production high-risk paths, pass a `ReviewedIrreversibilityThreshold`. This
represents a reviewed conformal threshold with source reference, reviewer
identity, calibration size, coverage, and approval status. When a high-risk
decision crosses the calibrated risk threshold and the forecast lower bound
crosses the reviewed threshold, the policy returns
`no_go_reviewed_irreversibility_forecast`.

```python
from director_ai.core.guard_control import (
    GuardDecision,
    NoGoPolicy,
    ReviewedIrreversibilityThreshold,
    RiskEnvelope,
)

risk = RiskEnvelope(
    action_category="tool",
    reversibility="costly",
    domain="security",
    calibrated_threshold=0.7,
    no_go_threshold=0.95,
)

decision = GuardDecision(
    decision="warn",
    risk_score=0.78,
    confidence_low=0.69,
    confidence_high=0.86,
    policy_id="policy.ops",
    reason="operator_review",
    tenant_safe_explanation="The action needs operator review.",
    evidence_refs=("ops:change-7",),
    verifier_signals=(),
    risk_envelope=risk,
    attributes={"action_sequence": "preview deployment\ndelete production table"},
)

verdict = NoGoPolicy(
    irreversible_threshold=0.95,
    reviewed_irreversibility_threshold=ReviewedIrreversibilityThreshold(
        threshold=0.75,
        source_ref="calibration://irreversibility/2026-05-13",
        reviewer_id="reviewer-passport-a",
        calibration_size=256,
        coverage=0.95,
    ),
).evaluate(decision)
assert verdict.decision in {"warn", "block"}
```

Only tenant-safe action labels belong in `action_sequence`,
`action_description`, `proposed_action`, `tool_action`, or `physical_action`.
Raw prompts, credentials, private sensor packets, media payloads, and generated
content remain outside the policy attributes and should be referenced by opaque
evidence ids.

## Consensus Wiring

`CrossVerifierConsensus.decide()` accepts the same tenant-safe action sequence
and forwards it through the no-go policy path. Forecast metadata is preserved in
decision attributes using numeric strings suitable for telemetry:

```python
from director_ai.core.guard_control import NoGoPolicy, RiskEnvelope, VerifierSignal
from director_ai.core.scoring.consensus import CrossVerifierConsensus

signal = VerifierSignal(
    verifier="policy",
    modality="policy",
    score=0.61,
    verdict="uncertain",
    confidence_low=0.54,
    confidence_high=0.73,
    evidence_refs=("policy:change-risk",),
)

decision = CrossVerifierConsensus(no_go_policy=NoGoPolicy()).decide(
    (signal,),
    risk_envelope=RiskEnvelope(
        action_category="tool",
        reversibility="costly",
        domain="financial",
        calibrated_threshold=0.5,
        no_go_threshold=0.95,
    ),
    policy_id="policy.finance.ops",
    action_sequence=("stage plan", "transfer funds"),
)
```

When the no-go policy blocks from a forecast, `decision.reason` is
`"no_go_irreversibility_forecast"` or
`"no_go_reviewed_irreversibility_forecast"` and the attributes include
`irreversibility_forecast_p`, `irreversibility_forecast_ci_low`,
`irreversibility_forecast_ci_high`, and `irreversibility_forecast_samples`.
Reviewed thresholds also add reviewed threshold provenance attributes.

## Full API

::: director_ai.core.guard_control.GuardDecision

::: director_ai.core.guard_control.RiskEnvelope

::: director_ai.core.guard_control.VerifierSignal

::: director_ai.core.guard_control.NoGoPolicy

::: director_ai.core.guard_control.NoGoVerdict

::: director_ai.core.guard_control.ReviewedIrreversibilityThreshold
