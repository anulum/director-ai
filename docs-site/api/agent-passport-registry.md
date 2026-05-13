# Agent Passport Registry

The agent passport registry makes signed agent identity, capability claims, and
coherence history auditable through the shared guard-control contracts.

## Decision Semantics

`AgentPassportRegistry` wraps the existing HMAC passport signer and verifier.
It fails closed when a passport cannot be verified:

- expired passports return a `block` guard decision
- unknown key ids return a `block` guard decision
- revoked passport signatures return a `block` guard decision
- missing capabilities return a `block` guard decision for tool, physical, and
  training actions
- no-go policy escalation is applied after passport and capability checks

```python
from director_ai.core.agent_identity import AgentPassportRegistry, PassportSigner
from director_ai.core.guard_control import RiskEnvelope

signer = PassportSigner(
    active_key=b"x" * 32,
    active_key_id="k1",
)
registry = AgentPassportRegistry(signer=signer)
passport = registry.issue_passport(
    agent_id="tenant-a/worker/tool",
    role="worker",
    tenant_id="tenant-a",
    capabilities=("tool:search",),
)

verdict = registry.evaluate_action(
    passport=passport,
    required_capability="tool:search",
    risk_envelope=RiskEnvelope(
        action_category="tool",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    ),
    event_ref="event://tool-call-1",
)
```

The returned `PassportActionVerdict.guard_decision` can be serialised directly
to a `SafetyEvent` with the same tenant-safe path used by other guard-control
modules.

## Rotation And Revocation

`rotate_signer()` delegates to the existing `PassportSigner.rotate()` path. Old
passports keep verifying under the rotated-out inactive key until they expire or
are revoked.

```python
registry.rotate_signer(new_active_key=b"y" * 32, new_active_key_id="k2")
registry.revoke(passport, reason="operator_rotation")
```

Revocation is exact-signature based. This lets operators revoke one issued
passport without invalidating every passport for the same agent id.

## Coherence History

`record_coherence()` links coherence outcomes to event references, not raw
prompts or completions. `export_agent()` returns a privacy-preserving summary:

- agent id, role, tenant id, capabilities, key id, issue time, expiry time
- revoked flag
- event-linked coherence scores and decisions
- aggregate count, minimum, mean, and latest coherence

It does not export signatures, signing keys, credentials, raw prompts, raw
completions, tool payloads, or retrieved evidence text.

## Full API

::: director_ai.core.agent_identity.registry.AgentPassportRegistry

::: director_ai.core.agent_identity.registry.PassportActionVerdict

::: director_ai.core.agent_identity.registry.CoherenceHistoryEntry
