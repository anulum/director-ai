# Federated Safety Signal Sharing

`FederatedSafetySignalAggregator` turns tenant-safe
`DirectorSafetySignal` envelopes into anonymous, differentially private
aggregate guard telemetry for cross-organisation sharing.

It enforces four boundaries before release:

- input must validate as a `DirectorSafetySignal`;
- each signal must carry a tenant id for contribution bounding;
- each tenant can contribute at most one count per category per release window;
- releases require at least `min_tenants` distinct tenants and omit raw counts
  by default.

```python
from director_ai.core import (
    FederatedSafetySignalAggregator,
    PrivacyAccountant,
    director_safety_signal_from_event,
)
from director_ai.core.safety_event import SafetyEvent

accountant = PrivacyAccountant(max_epsilon=5.0)
aggregator = FederatedSafetySignalAggregator(
    epsilon=0.9,
    accountant=accountant,
    min_tenants=2,
)

event = SafetyEvent.from_policy_decision(
    hook_id="stream",
    hook_scope="streaming",
    policy_decision="halt",
    halt_reason="coherence",
    tenant_safe_explanation="Tenant-safe halt summary.",
    tenant_id="tenant-a",
)
signal = director_safety_signal_from_event(
    event,
    producer_id="runtime-a",
    framework="generic",
)

aggregator.submit_signal(signal)
```

`release()` returns noised counts and privacy metadata:

```python
release = aggregator.release()
payload = release.to_dict()
```

`payload` contains `noisy_counts`, `epsilon_spent`, category names, cohort
counts, and a privacy block with `payload_classification` set to
`anonymous_dp_aggregate`. It does not include tenant ids, raw prompts, raw
completions, credentials, media, sensor payloads, or raw aggregate counts.

Raw aggregate counts can be included only for local audit/debug use:

```python
local_audit_payload = release.to_dict(include_raw=True)
```

Do not publish `include_raw=True` payloads across tenant or organisation
boundaries.

## Local Evidence Packet

Generate the local federated privacy packet before promoting a deployment that
shares guard signals across tenants or organisations:

```bash
PYTHONPATH=src python -m benchmarks.federated_privacy_evidence
```

The packet checks DP-noised release, tenant/category contribution caps, minimum
tenant cohort blocking, and additive secret-sharing aggregate reconstruction.
It does not include an external federation run or malicious-secure aggregation
proof.

::: director_ai.core.federated_privacy.signal_sharing.FederatedSafetySignalAggregator

::: director_ai.core.federated_privacy.signal_sharing.FederatedSafetySignalRelease
