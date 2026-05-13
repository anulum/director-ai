# Director Safety Telemetry v1

This interoperability specification defines the tenant-safe telemetry event
that inference servers, agent runtimes, streaming kernels, dashboards, and audit
pipelines can exchange without copying raw prompts, completions, media, private
sensor packets, credentials, or token-bearing secrets.

## Version

| Field | Value |
|-------|-------|
| Schema version | `director.safety_event.v1` |
| JSON Schema | `schemas/safety-event.schema.json` |
| Schema id | `https://anulum.github.io/director-ai/schemas/safety-event.schema.json` |

The repository test suite checks that the published JSON Schema file exactly
matches the runtime `SAFETY_EVENT_JSON_SCHEMA` constant, so the public
interoperability artifact and Python validator cannot drift silently.

## Required Event Shape

Every event must include:

- `schema_version`, `event_id`, and `timestamp`
- request and tenant correlation ids, which may be empty strings when the
  runtime has no tenant boundary
- hook identity: `hook_id` and `hook_scope`
- policy output: `policy_decision`, `halt_reason`, threshold, observed score,
  and latency
- tenant-safe evidence references
- a tenant-safe explanation
- optional structured trace attribution
- string-only attributes

No additional top-level fields are allowed.

## Privacy Boundary

The event is an audit and routing record, not a payload transport. Producers
must use opaque references for evidence and keep the following outside the
event:

- raw prompts and completions
- credentials, API keys, passwords, private keys, and tokens
- raw image, audio, video, camera, IMU, actuator, or simulator packets
- private customer documents or retrieved chunk text

Consumers should reject events that include unsafe attribute names or unsafe
attribute values before writing them to shared logs or dashboards.

## Minimal Example

```json
{
  "schema_version": "director.safety_event.v1",
  "event_id": "sevt_0123456789abcdef0123456789abcdef",
  "timestamp": "2026-05-13T12:00:00Z",
  "request_id": "req-7",
  "tenant_id": "tenant-3",
  "hook_id": "streaming.kernel",
  "hook_scope": "streaming",
  "policy_decision": "halt",
  "halt_reason": "coherence_below_threshold",
  "threshold": 0.5,
  "observed_score": 0.31,
  "latency_ms": 12.4,
  "evidence_refs": ["kb://physics#1"],
  "tenant_safe_explanation": "Review grounding evidence.",
  "trace_attribution": null,
  "attributes": {
    "policy_id": "policy.streaming.regulated"
  }
}
```

## Consumer Requirements

1. Validate against `schemas/safety-event.schema.json`.
2. Reject unknown top-level fields.
3. Treat `evidence_refs` as references only; do not dereference them outside the
   tenant boundary.
4. Preserve `schema_version` when forwarding.
5. Forward transport metadata through `DirectorSafetySignal` when the event is
   exchanged across frameworks.

## Full Runtime API

See [Safety Event Schema](../api/safety-event-schema.md) and
[Director Safety Protocol](../api/director-safety-protocol.md).
