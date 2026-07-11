# Safety Event Schema

`SafetyEvent` is the canonical tenant-safe telemetry record for runtime halt
and policy decisions. The same schema is used by streaming guards, inference
server hooks, containment, attestation, ontology, trajectory, cyber-physical,
swarm, and agent surfaces.

The independent interoperability specification is published at
[Director Safety Telemetry v1](../specs/director-safety-telemetry-v1.md). Its
machine-readable JSON Schema lives at `schemas/safety-event.schema.json` and is
tested against the runtime schema constant.

The schema is intentionally payload-safe:

- evidence is represented by references only;
- raw prompts, completions, media, credentials, private sensor payloads, and
  token-bearing secrets are not allowed;
- scores and thresholds are bounded to `[0, 1]`;
- hook scopes and policy decisions are closed enums;
- hook-specific telemetry goes under string-only `attributes`.

```python
from director_ai.core import (
    SAFETY_EVENT_JSON_SCHEMA,
    SafetyEvent,
    validate_safety_event_payload,
)

event = SafetyEvent.from_policy_decision(
    hook_id="inference_server.vllm.prehalt",
    hook_scope="inference_server",
    policy_decision="halt",
    halt_reason="coherence_below_threshold",
    tenant_safe_explanation="Pre-sampling halt fired.",
    threshold=0.5,
    observed_score=0.31,
    evidence_refs=("trajectory:7",),
    attributes={"server": "vllm", "token_id": "2"},
)

payload = event.to_dict()
validated = validate_safety_event_payload(payload)
assert validated == event
```

`SAFETY_EVENT_JSON_SCHEMA` is JSON-serialisable and can be handed to downstream
schema registries, inference-server adapters, dashboards, or audit pipelines.
The stdlib validator `validate_safety_event_payload()` enforces the same
critical constraints without requiring a runtime `jsonschema` dependency.

For cross-runtime sharing, wrap validated events in
[`DirectorSafetySignal`](director-safety-protocol.md). The signal validator
uses the same event validator internally, so direct event ingestion and protocol
transport stay aligned.

Two helpers keep records opaque and time-consistent:
`new_safety_event_id()` returns the opaque event ID and
`utc_timestamp()` the RFC-3339 UTC timestamp stamped into every record.

::: director_ai.core.safety_event.SafetyEvent

::: director_ai.core.safety_event.validate_safety_event_payload
