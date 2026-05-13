# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Correction-loop tests for approval-gated post-halt remediation."""

from __future__ import annotations

import pytest

from director_ai.core.guard_control import RiskEnvelope, VerifierSignal
from director_ai.core.runtime.correction import CorrectionLoop
from director_ai.core.runtime.structured_recovery import StructuredRecoveryResult
from director_ai.core.scoring.consensus import CrossVerifierConsensus


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
