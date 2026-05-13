# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Halt Recovery Patterns

# Halt Recovery Patterns

Structured recovery is for parser-facing streams where a halt must not leave
downstream systems trying to parse unsafe partial text. Plain text streams keep
the normal hard-interrupt behaviour unless the caller passes
`StructuredRecoveryConfig`.

## Parser-safe recovery

Use `last_valid` when the downstream system expects JSON, tool-call envelopes,
or reasoning-chain envelopes:

```python
from director_ai.core import StreamingKernel, StructuredRecoveryConfig

kernel = StreamingKernel(hard_limit=0.5)
session = kernel.stream_tokens(
    token_generator,
    coherence_callback=score,
    structured_recovery=StructuredRecoveryConfig(
        kind="json",
        policy="last_valid",
        json_schema={"type": "object"},
    ),
)

if session.structured_recovery and session.structured_recovery.valid:
    payload = session.structured_recovery.last_valid_output
    halted_at = session.structured_recovery.halted_at
```

The recovery layer never completes braces, invents tool arguments, or generates
replacement reasoning. It only returns a checkpoint that was valid before the
halt.

## Recovery policies

| Policy | Use when | Returned payload |
| --- | --- | --- |
| `last_valid` | A parser can safely continue from the last validated checkpoint. | `last_valid_output` plus `halted_at` |
| `redacted` | Regulated review queues must not expose partial text. | Halt metadata without payload text |
| `raw_partial` | An operator explicitly needs the exact halted text for forensics. | `raw_partial`, marked invalid |

Choose `last_valid` for normal JSON, tool-call, and reasoning-chain recovery.
Choose `redacted` when policy requires review before exposing generated text.
Use `raw_partial` only in controlled diagnostics, because downstream parsers
must treat it as invalid.

## KB refresh

Use this when halt evidence points to stale or missing facts:

- inspect `session.halt_evidence_structured` and the latest safety event
- refresh, retract, or replace the source facts
- rerun the halted prompt in dry-run mode
- lower thresholds only after the source state is verified

## Threshold rollback

Use this when a recent tuning profile increases false halts:

- restore the previous profile overlay
- keep `StructuredRecoveryConfig(policy="last_valid")` while validating the
  rollback
- compare halt rate and false-positive feedback before promoting the profile

## Human review routing

Use `redacted` for review queues where partial text must not be shown until an
operator approves evidence release. The reviewer receives halt metadata, the
token offset, and the safety event ID instead of the raw partial payload.

```python
session = kernel.stream_tokens(
    token_generator,
    coherence_callback=score,
    structured_recovery=StructuredRecoveryConfig(
        kind="tool_call",
        policy="redacted",
        tool_manifest=manifest,
    ),
)

event_id = session.safety_events[-1].event_id if session.safety_events else ""
```

## Approval-gated correction

Use `CorrectionLoop` when a halted response needs a corrected candidate rather
than a retry. The loop consumes verifier signals, emits an unreleased
`CorrectionProposal`, and releases candidate text only after both conditions
hold:

- cross-verifier consensus returns `allow`
- an operator supplies a non-empty approval ID

```python
from director_ai.core import CorrectionLoop
from director_ai.core.guard_control import RiskEnvelope, VerifierSignal
from director_ai.core.scoring.consensus import CrossVerifierConsensus

loop = CorrectionLoop(
    consensus=CrossVerifierConsensus(),
    risk_envelope=RiskEnvelope(
        action_category="text",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.65,
        no_go_threshold=0.9,
    ),
    policy_id="policy.correction.regulated",
)

proposal = loop.propose(
    candidate_text="Corrected response text.",
    signals=[
        VerifierSignal(
            verifier="nli",
            modality="text",
            score=0.08,
            verdict="supported",
            confidence_low=0.03,
            confidence_high=0.14,
            evidence_refs=("kb://fact-1",),
        )
    ],
    evidence_refs=("trace://halt-7",),
    structured_recovery=session.structured_recovery,
)

approved = loop.approve(proposal, approval_id="review-20260513-001")
released_text = loop.release(approved)
```

The default audit payload is tenant-safe: `proposal.to_dict()` excludes the
generated candidate text and excludes raw structured-recovery payloads. Physical
or irreversible actions are rejected at proposal time and must remain in
human-review or no-go handling.

## Temporary policy fallback

Use temporary fallback only with an expiry:

- switch high-risk structured endpoints to `redacted`
- route requests to human review or asynchronous completion
- keep the hard halt active
- remove the fallback once KB or threshold repair is validated

Fallback changes must be operationally visible: record the affected endpoint,
tenant scope, expiry time, and rollback owner in the deployment change log.
