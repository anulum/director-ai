# Simulation Containment & Reality Anchor

`director_ai.core.containment` puts a cryptographic anchor on every
agent session so the gateway can refuse actions whose scope does not
match the anchor, and an event-stream monitor can raise a breakout
alarm the moment the agent attempts to cross the sandbox boundary.

The subpackage ships three composable pieces:

1. `ContainmentScope` — a `Literal` taxonomy: `sandbox`, `simulator`,
   `shadow`, `production`. Only `production` permits real-world effects
   (the `scope_allows_real_effects` helper encodes that rule).
2. `RealityAnchor` + `ContainmentAttestor` — HMAC-SHA256 signed
   attestation binding a session to a scope.
3. `BreakoutDetector` + `ContainmentGuard` — rule-based scanner and
   the orchestrator that composes attestor and detector into a single
   `allow` / `warn` / `block` verdict.

## Quick start

```python
from director_ai.core.containment import (
    BreakoutDetector, ContainmentAttestor, ContainmentGuard,
)

attestor = ContainmentAttestor(key=secrets_vault.get("host_hmac_key"),
                               issuer="host://edge-11")
guard = ContainmentGuard(attestor=attestor, detector=BreakoutDetector())

# At session open — host signs the anchor for this run.
anchor = attestor.mint(session_id="sess-42", scope="sandbox")

# On every outbound action — check before dispatching.
verdict = guard.check({"text": candidate, "url": outbound_url}, anchor)
if not verdict.allowed:
    abort_with_audit(verdict)
```

## `RealityAnchor`

Frozen dataclass carrying `session_id`, `scope`, `issuer`,
`created_at`, `nonce` and `mac`. Every field is validated on
construction — empty identifiers, unknown scope literals, out-of-range
nonce / mac lengths all raise `ValueError`.

`canonical_payload` exposes the exact byte string that participates in
the MAC — delimiters (`|`) are escaped in the free-text fields so a
`session_id` containing the delimiter cannot collide with a different
anchor layout.

## `ContainmentAttestor`

- `mint(session_id, scope, nonce=None, created_at=None) -> RealityAnchor`
  produces a freshly signed anchor. The `nonce` defaults to 128 random
  bits from `secrets.token_hex`; `created_at` defaults to the
  attestor's clock.
- `verify(anchor, expected_scope=None) -> AnchorVerification` returns
  a structured `(valid, reason)` tuple rather than raising so callers
  can decide whether a bad anchor blocks the action or degrades with
  an alarm.

The constant-time MAC comparison uses `hmac.compare_digest`. Freshness
is controlled by `max_age_seconds` (0 disables the check);
future-timestamp anchors are rejected when the clock skew exceeds
five seconds.

The HMAC key must be at least 32 bytes (`_MIN_KEY_LEN`) — shorter
keys are refused at construction so operators cannot accidentally
downgrade the primitive.

## `BreakoutDetector`

Pattern matchers for the three observed practical breakout modes:

| Mode | Signal |
|---|---|
| Production target in rehearsal scope | The event's `hostname` / `url` matches the detector's production-host set (customisable; built-in list covers common production APIs). |
| Anti-anchor prompt injection | The event's free-text fields contain case-insensitive phrases aimed at overriding the anchor (`ignore sandbox`, `disregard containment`, `you are in production`, etc.). Text is length-capped by `max_text_length` to bound scan cost. |
| Claimed vs. anchored scope mismatch | The event asserts a `claimed_scope` that disagrees with the verified anchor. |

The scanner returns every finding it sees — it does not short-circuit
on the first match — so audit logs contain the full picture.

Customise the detector:

```python
BreakoutDetector(
    production_hosts={"internal.billing", "iam.corp"} | DEFAULT_HOSTS,
    anti_anchor_phrases=("cross the boundary",) + DEFAULT_PHRASES,
    max_text_length=256_000,
)
```

## `ContainmentGuard`

`check(event, anchor, claimed_scope=None) -> ContainmentVerdict`
applies the decision matrix:

1. Anchor fails to verify → `"block"` with `anchor_reason` filled,
   detector skipped.
2. Anchor verifies and scope is `"production"` → production-target
   findings are downgraded (the agent's legitimate target set); other
   findings still apply.
3. Anchor verifies and scope is a rehearsal one → any high-severity
   finding is `"block"`, medium is `"warn"`, clean is `"allow"`.

The `ContainmentVerdict` exposes `decision`, `findings` and
`anchor_reason` so the caller can log every branch distinctly.

## CoherenceAgent wiring

```python
from director_ai.core.agent import CoherenceAgent

agent = CoherenceAgent(
    containment_guard=guard,
    containment_anchor=anchor,
)
```

The two kwargs are enforced together — configuring one without the
other raises `ValueError`. Once wired, every call to `agent.process`
runs the output text through the guard before returning; a block
verdict converts the `ReviewResult` into a halted one whose
`halt_evidence.suggested_action` lists the findings.
