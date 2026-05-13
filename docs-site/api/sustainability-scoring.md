# Sustainability Scoring

Sustainability scoring converts token demand, estimated energy, estimated
carbon, and tenant quota state into the shared `GuardDecision` and
`SafetyEvent` contracts.

## Policy Semantics

`SustainabilityPolicyAdapter` is a policy adapter, not a factual verifier. It
emits a `VerifierSignal` with modality `sustainability` so downstream
orchestrators can keep cost, quota, carbon, and forecast pressure separate from
truthfulness evidence.

The adapter enforces these defaults:

- estimates disclose whether hardware values are `measured`, `configured`, or
  `projected`
- quota exhaustion maps to `halt` for ordinary reversible actions
- forecast headroom exhaustion maps to `warn`
- high-carbon requests map to `warn` with `recommended_action="defer"`
- high-risk safety actions are not halted solely for sustainability reasons
- events and telemetry omit raw prompts, completions, media, credentials, and
  other tenant payloads

```python
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.sustainability import (
    HardwareProfile,
    HardwareProfileRegistry,
    SustainabilityPolicyAdapter,
    TokenEnergyCostEstimator,
)

profile = HardwareProfile(
    profile_id="edge-gpu-a",
    energy_kwh_per_1k_tokens=0.0012,
    carbon_kg_per_kwh=0.18,
    provenance="measured",
)
registry = HardwareProfileRegistry((profile,))
estimator = TokenEnergyCostEstimator(
    hardware_profile=registry.get("edge-gpu-a"),
    cost_per_1k_tokens=0.004,
)
adapter = SustainabilityPolicyAdapter(
    estimator=estimator,
    policy_id="policy.sustainability.production",
    carbon_defer_kg=0.25,
    forecast_headroom_ratio=0.2,
)

decision = adapter.evaluate(
    tenant_id="tenant-a",
    input_tokens=800,
    output_tokens=300,
    quota_remaining_tokens=1200,
    forecast_next_tokens=100,
    risk_envelope=RiskEnvelope(
        action_category="text",
        reversibility="reversible",
        domain="general",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    ),
)
```

For physical, medical, legal, security, irreversible, tool, code, or training
actions, sustainability pressure becomes a warning unless another guard-control
policy independently blocks the action.

## Tenant Telemetry

`SustainabilityTelemetry` aggregates estimates per tenant and returns threshold
alerts without serialising raw request content.

```python
from director_ai.core.sustainability import SustainabilityTelemetry

telemetry = SustainabilityTelemetry(
    token_alert_threshold=100_000,
    cost_alert_threshold=10.0,
    carbon_alert_threshold=1.0,
)
estimate = estimator.estimate(input_tokens=800, output_tokens=300)
telemetry.record("tenant-a", estimate)
```

The telemetry `record()` method accepts `SustainabilityEstimate` values. Store
only aggregate summaries in monitoring backends unless the deployment has a
separate tenant-approved data-retention policy.

## Full API

::: director_ai.core.sustainability.policy_adapter.HardwareProfile

::: director_ai.core.sustainability.policy_adapter.HardwareProfileRegistry

::: director_ai.core.sustainability.policy_adapter.SustainabilityEstimate

::: director_ai.core.sustainability.policy_adapter.TokenEnergyCostEstimator

::: director_ai.core.sustainability.policy_adapter.SustainabilityPolicyAdapter

::: director_ai.core.sustainability.policy_adapter.SustainabilityTelemetry

::: director_ai.core.sustainability.policy_adapter.SustainabilityTelemetrySummary
