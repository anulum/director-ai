# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Correction-loop tests for approval-gated post-halt remediation."""

from __future__ import annotations

import pytest

from director_ai.core.guard_control import GuardDecision, RiskEnvelope, VerifierSignal
from director_ai.core.runtime.correction import (
    CorrectionLoop,
    CorrectionProposal,
    GroundedCorrectionDraft,
    HaltCorrectionContext,
)
from director_ai.core.runtime.structured_recovery import StructuredRecoveryResult
from director_ai.core.scoring.consensus import CrossVerifierConsensus
from director_ai.core.types import EvidenceChunk, HaltEvidence, HaltTraceAttribution


def _text_envelope() -> RiskEnvelope:
    return RiskEnvelope(
        action_category="text",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.65,
        no_go_threshold=0.9,
    )


def _supported_signal() -> VerifierSignal:
    return VerifierSignal(
        verifier="nli",
        modality="text",
        score=0.08,
        verdict="supported",
        confidence_low=0.03,
        confidence_high=0.14,
        evidence_refs=("kb://fact-1",),
        latency_ms=3.2,
    )


def test_grounded_correction_requires_allow_decision_and_approval_before_release():
    loop = CorrectionLoop(
        consensus=CrossVerifierConsensus(),
        risk_envelope=_text_envelope(),
        policy_id="policy.correction.regulated",
    )
    recovery = StructuredRecoveryResult(
        kind="json",
        policy="last_valid",
        halted_at=12,
        last_valid_output='{"status":"valid"}',
        valid=True,
        metadata={"json_root": "object"},
    )

    proposal = loop.propose(
        candidate_text="The corrected answer cites the validated source.",
        signals=[_supported_signal()],
        evidence_refs=("kb://fact-1",),
        structured_recovery=recovery,
    )

    assert proposal.approved is False
    assert proposal.guard_decision.decision == "allow"
    with pytest.raises(PermissionError, match="not approved"):
        loop.release(proposal)

    approved = loop.approve(proposal, approval_id="review-20260513-001")

    assert approved.approved is True
    assert approved.approval_id == "review-20260513-001"
    assert loop.release(approved) == "The corrected answer cites the validated source."


def test_correction_loop_builds_grounded_candidate_from_halt_evidence():
    loop = CorrectionLoop(
        consensus=CrossVerifierConsensus(),
        risk_envelope=_text_envelope(),
        policy_id="policy.correction.regulated",
    )
    halt_evidence = HaltEvidence(
        reason="hard_limit breach",
        last_score=0.21,
        evidence_chunks=[
            EvidenceChunk(
                text="The validated source says the launch date is 2026-05-13.",
                distance=0.02,
                source="kb://fact-launch-date",
            )
        ],
        trace_attribution=HaltTraceAttribution(
            fact_source="kb://fact-launch-date",
            retrieval_path="retrieval://query-7",
            scorer_path="nli://scorer",
            token_offset=18,
            threshold=0.4,
            causal_contribution=0.19,
        ),
    )
    observed_contexts: list[HaltCorrectionContext] = []

    def build_candidate(context: HaltCorrectionContext) -> GroundedCorrectionDraft:
        observed_contexts.append(context)
        return GroundedCorrectionDraft(
            candidate_text=(
                "The launch date is 2026-05-13 [source: kb://fact-launch-date]."
            ),
            verifier_signals=(_supported_signal(),),
            evidence_refs=("kb://fact-launch-date",),
        )

    proposal = loop.propose_from_halt(
        halt_evidence=halt_evidence,
        continuation_builder=build_candidate,
    )

    assert proposal.guard_decision.decision == "allow"
    assert proposal.evidence_refs == (
        "kb://fact-launch-date",
        "trace://retrieval://query-7",
        "trace://nli://scorer",
        "kb://fact-1",
    )
    assert observed_contexts[0].halt_reason == "hard_limit breach"
    assert observed_contexts[0].source_refs == ("kb://fact-launch-date",)
    assert observed_contexts[0].evidence_texts == (
        "The validated source says the launch date is 2026-05-13.",
    )
    assert "candidate_text" not in proposal.to_dict()
    approved = loop.approve(proposal, approval_id="review-20260513-003")
    assert loop.release(approved) == (
        "The launch date is 2026-05-13 [source: kb://fact-launch-date]."
    )


def test_correction_loop_rejects_halt_correction_without_grounding_refs():
    loop = CorrectionLoop(
        consensus=CrossVerifierConsensus(),
        risk_envelope=_text_envelope(),
        policy_id="policy.correction.regulated",
    )
    halt_evidence = HaltEvidence(
        reason="hard_limit breach",
        last_score=0.21,
        evidence_chunks=[
            EvidenceChunk(text="Source text", distance=0.02, source="kb://fact-9")
        ],
    )

    def build_candidate(context: HaltCorrectionContext) -> GroundedCorrectionDraft:
        return GroundedCorrectionDraft(
            candidate_text="Correction without a source.",
            verifier_signals=(_supported_signal(),),
            evidence_refs=(),
        )

    with pytest.raises(ValueError, match="grounding evidence_refs"):
        loop.propose_from_halt(
            halt_evidence=halt_evidence,
            continuation_builder=build_candidate,
        )


def test_correction_loop_rejects_halt_correction_with_unknown_refs():
    loop = CorrectionLoop(
        consensus=CrossVerifierConsensus(),
        risk_envelope=_text_envelope(),
        policy_id="policy.correction.regulated",
    )
    halt_evidence = HaltEvidence(
        reason="hard_limit breach",
        last_score=0.21,
        evidence_chunks=[
            EvidenceChunk(text="Source text", distance=0.02, source="kb://fact-9")
        ],
    )

    def build_candidate(context: HaltCorrectionContext) -> GroundedCorrectionDraft:
        return GroundedCorrectionDraft(
            candidate_text="Correction with an unrelated source.",
            verifier_signals=(_supported_signal(),),
            evidence_refs=("kb://unrelated",),
        )

    with pytest.raises(ValueError, match="unknown grounding evidence_refs"):
        loop.propose_from_halt(
            halt_evidence=halt_evidence,
            continuation_builder=build_candidate,
        )


def test_correction_proposal_blocks_unsafe_consensus():
    loop = CorrectionLoop(
        consensus=CrossVerifierConsensus(),
        risk_envelope=_text_envelope(),
        policy_id="policy.correction.regulated",
    )

    proposal = loop.propose(
        candidate_text="Unsupported correction.",
        signals=[
            VerifierSignal(
                verifier="nli",
                modality="text",
                score=0.93,
                verdict="contradiction",
                confidence_low=0.84,
                confidence_high=0.98,
                evidence_refs=("kb://fact-2",),
            )
        ],
    )

    assert proposal.guard_decision.decision == "halt"
    with pytest.raises(PermissionError, match="cannot approve"):
        loop.approve(proposal, approval_id="review-unsafe")
    with pytest.raises(PermissionError, match="not approved"):
        loop.release(proposal)


def test_correction_dataclasses_validate_required_fields():
    with pytest.raises(ValueError, match="halt_reason"):
        HaltCorrectionContext(
            halt_reason=" ",
            last_score=0.4,
            evidence_texts=("source",),
            source_refs=("kb://fact",),
        )
    with pytest.raises(ValueError, match="evidence_texts"):
        HaltCorrectionContext(
            halt_reason="halt",
            last_score=0.4,
            evidence_texts=(" ",),
            source_refs=("kb://fact",),
        )
    with pytest.raises(ValueError, match="candidate_text"):
        GroundedCorrectionDraft(
            candidate_text=" ",
            verifier_signals=(_supported_signal(),),
            evidence_refs=("kb://fact",),
        )
    with pytest.raises(ValueError, match="verifier_signals"):
        GroundedCorrectionDraft(
            candidate_text="Candidate.",
            verifier_signals=(),
            evidence_refs=("kb://fact",),
        )
    with pytest.raises(ValueError, match="proposal_id"):
        CorrectionProposal(
            proposal_id=" ",
            candidate_text="Candidate.",
            evidence_refs=("kb://fact",),
            guard_decision=loop_decision("allow"),
        )
    with pytest.raises(ValueError, match="approval_id"):
        CorrectionProposal(
            proposal_id="correction-1",
            candidate_text="Candidate.",
            evidence_refs=("kb://fact",),
            guard_decision=loop_decision("allow"),
            approved=True,
        )


def test_correction_loop_blocks_physical_or_irreversible_domains():
    for envelope in [
        RiskEnvelope(
            action_category="physical",
            reversibility="reversible",
            domain="regulated",
            calibrated_threshold=0.65,
            no_go_threshold=0.9,
        ),
        RiskEnvelope(
            action_category="text",
            reversibility="irreversible",
            domain="regulated",
            calibrated_threshold=0.65,
            no_go_threshold=0.9,
        ),
        RiskEnvelope(
            action_category="text",
            reversibility="reversible",
            domain="physical",
            calibrated_threshold=0.65,
            no_go_threshold=0.9,
        ),
    ]:
        loop = CorrectionLoop(
            consensus=CrossVerifierConsensus(),
            risk_envelope=envelope,
            policy_id="policy.correction.regulated",
        )
        with pytest.raises(ValueError, match="physical or irreversible"):
            loop.propose(
                candidate_text="Candidate.",
                signals=[_supported_signal()],
                evidence_refs=("kb://fact",),
            )


def test_correction_loop_rejects_blank_policy_and_bad_approval():
    with pytest.raises(ValueError, match="policy_id"):
        CorrectionLoop(
            consensus=CrossVerifierConsensus(),
            risk_envelope=_text_envelope(),
            policy_id=" ",
        )
    loop = CorrectionLoop(
        consensus=CrossVerifierConsensus(),
        risk_envelope=_text_envelope(),
        policy_id="policy.correction.regulated",
    )
    proposal = loop.propose(
        candidate_text="Candidate.",
        signals=[_supported_signal()],
        evidence_refs=("kb://fact",),
    )
    with pytest.raises(ValueError, match="approval_id"):
        loop.approve(proposal, approval_id=" ")


def loop_decision(decision: str):
    signal = _supported_signal()
    return GuardDecision(
        decision=decision,
        risk_score=0.08,
        confidence_low=0.02,
        confidence_high=0.15,
        policy_id="policy.correction.regulated",
        reason="supported",
        tenant_safe_explanation="Review approved evidence.",
        evidence_refs=("kb://fact",),
        verifier_signals=(signal,),
        risk_envelope=_text_envelope(),
    )


@pytest.mark.parametrize(
    "envelope",
    [
        RiskEnvelope(
            action_category="physical",
            reversibility="reversible",
            domain="physical",
            calibrated_threshold=0.5,
            no_go_threshold=0.7,
        ),
        RiskEnvelope(
            action_category="tool",
            reversibility="irreversible",
            domain="security",
            calibrated_threshold=0.5,
            no_go_threshold=0.7,
        ),
    ],
)
def test_correction_loop_rejects_physical_or_irreversible_actions(
    envelope: RiskEnvelope,
):
    loop = CorrectionLoop(
        consensus=CrossVerifierConsensus(),
        risk_envelope=envelope,
        policy_id="policy.correction.no-auto",
    )

    with pytest.raises(ValueError, match="physical or irreversible"):
        loop.propose(
            candidate_text="Do not auto-correct this action.",
            signals=[_supported_signal()],
        )


def test_correction_proposal_serialisation_is_tenant_safe_by_default():
    loop = CorrectionLoop(
        consensus=CrossVerifierConsensus(),
        risk_envelope=_text_envelope(),
        policy_id="policy.correction.regulated",
    )

    proposal = loop.propose(
        candidate_text="Private generated payload with internal marker",
        signals=[_supported_signal()],
        evidence_refs=("kb://fact-1", "trace://halt-7"),
    )

    audit_payload = proposal.to_dict()

    assert "candidate_text" not in audit_payload
    assert audit_payload["evidence_refs"] == ["kb://fact-1", "trace://halt-7"]
    assert audit_payload["guard_decision"]["decision"] == "allow"
    assert audit_payload["structured_recovery"] is None
    assert proposal.to_safety_event(hook_id="correction", hook_scope="agent")
