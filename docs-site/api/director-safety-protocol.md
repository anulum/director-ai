# Director Safety Protocol

The Director Safety Protocol is the public transport envelope for sharing
tenant-safe guard signals across runtimes, orchestrators, dashboards, and audit
pipelines. It wraps the existing `SafetyEvent` schema rather than replacing it:
`SafetyEvent` remains the canonical decision record, and
`DirectorSafetySignal` adds producer identity, interoperability hints, privacy
flags, and deterministic JSON serialization.

The envelope carries the wire identifier
`DIRECTOR_SAFETY_PROTOCOL_VERSION` (`director.safety_protocol.v1`);
consumers reject payloads with an unknown version.

## Contract

`DirectorSafetySignal` enforces these boundaries:

- every signal carries `director.safety_protocol.v1`
- every embedded event carries `director.safety_event.v1`
- privacy metadata must declare `payload_classification="tenant_safe"`
- `raw_payload_included` must be `false`
- evidence is represented by references only
- attributes and extensions are rejected when their keys or values indicate
  credentials, private prompts, private media, sensor payloads, or token-bearing
  content
- transport JSON is canonicalized with sorted keys and compact separators so it
  can be signed, hashed, or compared by downstream systems

```python
from director_ai.core.safety_event import SafetyEvent
from director_ai.core.safety_protocol import director_safety_signal_from_event

event = SafetyEvent.from_policy_decision(
    hook_id="streaming.kernel",
    hook_scope="streaming",
    policy_decision="halt",
    halt_reason="coherence_below_threshold",
    tenant_safe_explanation="Review grounding evidence.",
    threshold=0.5,
    observed_score=0.31,
    evidence_refs=("kb://physics#1",),
    attributes={"policy_id": "policy.streaming.regulated"},
)

signal = director_safety_signal_from_event(
    event,
    producer_id="runtime-alpha",
    framework="generic-agent",
)

payload = signal.to_transport_dict()
wire_json = signal.to_json()
```

## Transport Fields

| Field | Meaning |
|-------|---------|
| `protocol_version` | Safety protocol version, currently `director.safety_protocol.v1` |
| `schema_ref` | Public documentation URL for the envelope |
| `signal_id` | Opaque `dsp_...` identifier for the transport message |
| `emitted_at` | UTC emission timestamp |
| `producer_id` | Stable producer/runtime identifier |
| `framework` | Source runtime or orchestrator family |
| `event_schema_version` | Embedded `SafetyEvent` schema id |
| `event` | Tenant-safe `SafetyEvent.to_dict()` payload |
| `interoperability` | Decision, severity, hook scope, reason, and evidence count |
| `privacy` | Tenant-safety declaration and redaction requirements |
| `extensions` | Optional tenant-safe string metadata |

The interoperability severity mapping is deterministic:

| Decision | Severity |
|----------|----------|
| `allow` | `informational` |
| `warn` | `advisory` |
| `halt` | `terminal` |
| `block` | `terminal` |

## Validation

Use `validate_director_safety_signal()` at trust boundaries. It reconstructs the
`SafetyEvent` through `validate_safety_event_payload()`, verifies the privacy
declaration, checks severity consistency, and rejects unsafe attributes or
extensions.

```python
from director_ai.core.safety_protocol import validate_director_safety_signal

validated = validate_director_safety_signal(payload)
assert validated.event.policy_decision == "halt"
```

## Full API

::: director_ai.core.safety_protocol.DirectorSafetySignal

::: director_ai.core.safety_protocol.director_safety_signal_from_event

::: director_ai.core.safety_protocol.validate_director_safety_signal

::: director_ai.core.safety_protocol.new_director_safety_signal_id
