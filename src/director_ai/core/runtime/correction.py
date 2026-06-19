# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Approval-Gated Correction Loop

"""Approval-gated correction proposals after a halted or warned response."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from director_ai.core.guard_control import GuardDecision, RiskEnvelope, VerifierSignal
from director_ai.core.safety_event import SafetyEvent
from director_ai.core.scoring.consensus import CrossVerifierConsensus
from director_ai.core.types import HaltEvidence

from .structured_recovery import StructuredRecoveryResult

__all__ = [
    "CorrectionLoop",
    "CorrectionProposal",
    "GroundedCorrectionDraft",
    "HaltCorrectionContext",
]


@dataclass(frozen=True)
class HaltCorrectionContext:
    """Tenant-local evidence context passed to a correction continuation builder."""

    halt_reason: str
    last_score: float
    evidence_texts: Sequence[str]
    source_refs: Sequence[str]
    trace_refs: Sequence[str] = ()
    suggested_action: str = ""

    def __post_init__(self) -> None:
        """Reject a blank halt reason."""
        if not self.halt_reason.strip():
            raise ValueError("halt_reason is required")
        object.__setattr__(
            self,
            "evidence_texts",
            tuple(text for text in map(str, self.evidence_texts) if text.strip()),
        )
        object.__setattr__(
            self,
            "source_refs",
            tuple(ref for ref in map(str, self.source_refs) if ref.strip()),
        )
        object.__setattr__(
            self,
            "trace_refs",
            tuple(ref for ref in map(str, self.trace_refs) if ref.strip()),
        )
        if not self.evidence_texts:
            raise ValueError("evidence_texts are required")
        if not self.source_refs:
            raise ValueError("source_refs are required")


@dataclass(frozen=True)
class GroundedCorrectionDraft:
    """Candidate continuation and verifier evidence produced from halt evidence."""

    candidate_text: str
    verifier_signals: Sequence[VerifierSignal]
    evidence_refs: Sequence[str]

    def __post_init__(self) -> None:
        """Reject blank candidate text."""
        if not self.candidate_text.strip():
            raise ValueError("candidate_text is required")
        object.__setattr__(self, "verifier_signals", tuple(self.verifier_signals))
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(ref for ref in map(str, self.evidence_refs) if ref.strip()),
        )
        if not self.verifier_signals:
            raise ValueError("verifier_signals are required")
        if not self.evidence_refs:
            raise ValueError("grounding evidence_refs are required")


@dataclass(frozen=True)
class CorrectionProposal:
    """Correction candidate with verifier evidence and an explicit release gate."""

    proposal_id: str
    candidate_text: str
    evidence_refs: Sequence[str]
    guard_decision: GuardDecision
    structured_recovery: StructuredRecoveryResult | None = None
    approved: bool = False
    approval_id: str = ""

    def __post_init__(self) -> None:
        """Reject a blank proposal id."""
        if not self.proposal_id.strip():
            raise ValueError("proposal_id is required")
        if not self.candidate_text.strip():
            raise ValueError("candidate_text is required")
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))
        if self.approved and not self.approval_id.strip():
            raise ValueError("approval_id is required for approved proposals")

    def to_dict(self, *, include_candidate: bool = False) -> dict[str, Any]:
        """Serialise a tenant-safe audit payload.

        Generated candidate text is excluded by default because audit streams
        often cross tenant and operator boundaries.
        """
        payload: dict[str, Any] = {
            "proposal_id": self.proposal_id,
            "evidence_refs": list(self.evidence_refs),
            "guard_decision": self.guard_decision.to_dict(),
            "structured_recovery": _recovery_audit_payload(self.structured_recovery),
            "approved": self.approved,
            "approval_id": self.approval_id,
        }
        if include_candidate:
            payload["candidate_text"] = self.candidate_text
        return payload

    def to_safety_event(
        self,
        *,
        hook_id: str,
        hook_scope: str,
        request_id: str = "",
        tenant_id: str = "",
        latency_ms: float | None = None,
    ) -> SafetyEvent:
        """Convert the proposal decision to the shared safety-event schema."""
        return self.guard_decision.to_safety_event(
            hook_id=hook_id,
            hook_scope=hook_scope,
            request_id=request_id,
            tenant_id=tenant_id,
            latency_ms=latency_ms,
        )


class CorrectionLoop:
    """Build and release correction proposals under verifier consensus control."""

    def __init__(
        self,
        *,
        consensus: CrossVerifierConsensus,
        risk_envelope: RiskEnvelope,
        policy_id: str,
    ) -> None:
        if not policy_id.strip():
            raise ValueError("policy_id is required")
        self._consensus = consensus
        self._risk_envelope = risk_envelope
        self._policy_id = policy_id

    def propose(
        self,
        *,
        candidate_text: str,
        signals: Sequence[VerifierSignal],
        evidence_refs: Sequence[str] = (),
        structured_recovery: StructuredRecoveryResult | None = None,
    ) -> CorrectionProposal:
        """Return an unreleased correction proposal guarded by verifier consensus."""
        self._reject_non_automatic_domain()
        if not candidate_text.strip():
            raise ValueError("candidate_text is required")
        signal_tuple = tuple(signals)
        decision = self._consensus.decide(
            signal_tuple,
            risk_envelope=self._risk_envelope,
            policy_id=self._policy_id,
        )
        merged_refs = _merge_refs(evidence_refs, decision.evidence_refs)
        return CorrectionProposal(
            proposal_id=f"correction-{uuid4()}",
            candidate_text=candidate_text,
            evidence_refs=merged_refs,
            guard_decision=decision,
            structured_recovery=structured_recovery,
        )

    def propose_from_halt(
        self,
        *,
        halt_evidence: HaltEvidence,
        continuation_builder: Callable[
            [HaltCorrectionContext],
            GroundedCorrectionDraft,
        ],
        structured_recovery: StructuredRecoveryResult | None = None,
    ) -> CorrectionProposal:
        """Build an unreleased grounded correction proposal from halt evidence."""
        context = _halt_context(halt_evidence)
        draft = continuation_builder(context)
        _validate_draft_refs(draft.evidence_refs, context)
        return self.propose(
            candidate_text=draft.candidate_text,
            signals=draft.verifier_signals,
            evidence_refs=_merge_refs(draft.evidence_refs, context.trace_refs),
            structured_recovery=structured_recovery,
        )

    def approve(
        self,
        proposal: CorrectionProposal,
        *,
        approval_id: str,
    ) -> CorrectionProposal:
        """Approve a proposal that consensus already allowed."""
        if proposal.guard_decision.decision != "allow":
            raise PermissionError(
                f"cannot approve correction with decision "
                f"{proposal.guard_decision.decision!r}"
            )
        if not approval_id.strip():
            raise ValueError("approval_id is required")
        return replace(proposal, approved=True, approval_id=approval_id)

    def release(self, proposal: CorrectionProposal) -> str:
        """Return candidate text only after consensus allow and explicit approval."""
        if not proposal.approved:
            raise PermissionError("correction proposal is not approved")
        if proposal.guard_decision.decision != "allow":
            raise PermissionError(
                f"cannot release correction with decision "
                f"{proposal.guard_decision.decision!r}"
            )
        return proposal.candidate_text

    def _reject_non_automatic_domain(self) -> None:
        if (
            self._risk_envelope.action_category == "physical"
            or self._risk_envelope.domain == "physical"
            or self._risk_envelope.reversibility == "irreversible"
        ):
            raise ValueError(
                "correction proposals are not allowed for physical or "
                "irreversible actions"
            )


def _merge_refs(
    explicit_refs: Sequence[str],
    decision_refs: Sequence[str],
) -> tuple[str, ...]:
    refs: list[str] = []
    seen: set[str] = set()
    for ref in (*explicit_refs, *decision_refs):
        ref_s = str(ref)
        if ref_s not in seen:
            refs.append(ref_s)
            seen.add(ref_s)
    return tuple(refs)


def _halt_context(halt_evidence: HaltEvidence) -> HaltCorrectionContext:
    source_refs = tuple(
        chunk.source for chunk in halt_evidence.evidence_chunks if chunk.source.strip()
    )
    trace_refs: list[str] = []
    trace = halt_evidence.trace_attribution
    if trace is not None:
        if trace.fact_source.strip():
            trace_refs.append(trace.fact_source)
        if trace.retrieval_path.strip():
            trace_refs.append(f"trace://{trace.retrieval_path}")
        if trace.scorer_path.strip():
            trace_refs.append(f"trace://{trace.scorer_path}")
    return HaltCorrectionContext(
        halt_reason=halt_evidence.reason,
        last_score=halt_evidence.last_score,
        evidence_texts=tuple(chunk.text for chunk in halt_evidence.evidence_chunks),
        source_refs=source_refs,
        trace_refs=_merge_refs(source_refs, trace_refs),
        suggested_action=halt_evidence.suggested_action,
    )


def _validate_draft_refs(
    evidence_refs: Sequence[str],
    context: HaltCorrectionContext,
) -> None:
    allowed_refs = {*context.source_refs, *context.trace_refs}
    unknown_refs = tuple(ref for ref in evidence_refs if ref not in allowed_refs)
    if unknown_refs:
        refs = ", ".join(unknown_refs)
        raise ValueError(f"unknown grounding evidence_refs: {refs}")


def _recovery_audit_payload(
    recovery: StructuredRecoveryResult | None,
) -> dict[str, Any] | None:
    if recovery is None:
        return None
    return {
        "kind": recovery.kind,
        "policy": recovery.policy,
        "halted_at": recovery.halted_at,
        "valid": recovery.valid,
        "errors": list(recovery.errors),
        "metadata": dict(recovery.metadata),
    }
