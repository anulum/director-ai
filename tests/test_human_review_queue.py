# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Human Review Queue Tests
"""Tests for durable human-in-the-loop review gates."""

from __future__ import annotations

import pytest

from director_ai.core.guard_control import GuardDecision, RiskEnvelope, VerifierSignal
from director_ai.core.runtime.correction import CorrectionProposal
from director_ai.core.runtime.human_review import HumanReviewQueue
from director_ai.core.safety_event import SafetyEvent


def _allow_decision() -> GuardDecision:
    envelope = RiskEnvelope(
        action_category="text",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.65,
        no_go_threshold=0.9,
    )
    signal = VerifierSignal(
        verifier="nli",
        modality="text",
        score=0.08,
        verdict="supported",
        confidence_low=0.02,
        confidence_high=0.15,
        evidence_refs=("kb://fact-1",),
    )
    return GuardDecision(
        decision="allow",
        risk_score=0.08,
        confidence_low=0.02,
        confidence_high=0.15,
        policy_id="policy.review.test",
        reason="supported by verifier consensus",
        tenant_safe_explanation="Correction may be reviewed for release.",
        evidence_refs=("kb://fact-1",),
        verifier_signals=(signal,),
        risk_envelope=envelope,
    )


def _proposal() -> CorrectionProposal:
    return CorrectionProposal(
        proposal_id="correction-1",
        candidate_text="Use the validated source-backed correction.",
        evidence_refs=("kb://fact-1", "trace://halt-1"),
        guard_decision=_allow_decision(),
    )


def test_review_queue_requires_reviewer_approval_before_release(tmp_path):
    queue = HumanReviewQueue(tmp_path / "review.db")

    case = queue.enqueue_case(
        candidate_text="Release only after explicit review.",
        evidence_refs=("kb://fact-1",),
        tenant_id="tenant-a",
        request_id="req-1",
        source_kind="halt",
        reason="coherence halt",
    )

    assert case.status == "pending"
    with pytest.raises(PermissionError, match="not approved"):
        queue.release(case.case_id, reviewer_id="reviewer-1", release_id="rel-1")

    approved = queue.decide(
        case.case_id,
        reviewer_id="reviewer-1",
        action="approve",
        reason="source verified",
    )
    released_text = queue.release(
        case.case_id,
        reviewer_id="reviewer-1",
        release_id="rel-1",
    )

    assert approved.status == "approved"
    assert released_text == "Release only after explicit review."
    assert queue.get_case(case.case_id).status == "released"
    assert [d.action for d in queue.decisions(case.case_id)] == ["approve", "release"]


def test_review_queue_retry_payload_requires_retry_decision(tmp_path):
    queue = HumanReviewQueue(tmp_path / "review.db")
    case = queue.enqueue_case(
        candidate_text="Candidate with missing support.",
        evidence_refs=("kb://fact-2",),
        tenant_id="tenant-a",
        request_id="req-2",
        source_kind="correction",
    )

    with pytest.raises(PermissionError, match="retry was not requested"):
        queue.retry_payload(case.case_id)

    queue.decide(
        case.case_id,
        reviewer_id="reviewer-1",
        action="request_retry",
        reason="needs fresher source",
        metadata={"retry_budget": 1},
    )
    payload = queue.retry_payload(case.case_id)

    assert payload == {
        "case_id": case.case_id,
        "tenant_id": "tenant-a",
        "request_id": "req-2",
        "evidence_refs": ["kb://fact-2"],
        "reason": "needs fresher source",
        "retry_budget": "1",
    }
    assert "candidate_text" not in payload


def test_review_queue_rejection_blocks_release_and_survives_reopen(tmp_path):
    db_path = tmp_path / "review.db"
    queue = HumanReviewQueue(db_path)
    case = queue.enqueue_case(
        candidate_text="Unsafe candidate.",
        evidence_refs=("kb://fact-3",),
        tenant_id="tenant-b",
    )
    queue.decide(
        case.case_id,
        reviewer_id="reviewer-2",
        action="reject",
        reason="unsupported claim",
    )
    queue.close()

    reopened = HumanReviewQueue(db_path)
    restored = reopened.get_case(case.case_id)

    assert restored.status == "rejected"
    with pytest.raises(PermissionError, match="not approved"):
        reopened.release(case.case_id, reviewer_id="reviewer-2", release_id="rel-2")
    assert reopened.list_cases(status="rejected", tenant_id="tenant-b")[0].case_id == (
        case.case_id
    )
    reopened.close()


def test_review_queue_enqueues_correction_proposal_without_public_payload(tmp_path):
    queue = HumanReviewQueue(tmp_path / "review.db")

    case = queue.enqueue_correction_proposal(
        _proposal(),
        tenant_id="tenant-c",
        request_id="req-3",
        reason="post-halt correction proposal",
    )

    audit_payload = case.to_dict()
    assert audit_payload["source_kind"] == "correction"
    assert audit_payload["evidence_refs"] == ["kb://fact-1", "trace://halt-1"]
    assert "candidate_text" not in audit_payload
    assert case.to_dict(include_candidate=True)["candidate_text"].startswith("Use")


def test_review_queue_carries_tenant_safe_safety_event(tmp_path):
    queue = HumanReviewQueue(tmp_path / "review.db")
    event = SafetyEvent.from_policy_decision(
        hook_id="review.test",
        hook_scope="agent",
        policy_decision="halt",
        halt_reason="coherence",
        tenant_safe_explanation="Human review is required.",
        request_id="req-4",
        tenant_id="tenant-d",
        evidence_refs=("kb://fact-4",),
    )

    case = queue.enqueue_case(
        candidate_text="Candidate text stays gated.",
        evidence_refs=event.evidence_refs,
        tenant_id=event.tenant_id,
        request_id=event.request_id,
        source_kind="halt",
        safety_event=event,
    )

    payload = case.to_dict()
    assert payload["safety_event"]["event_id"] == event.event_id
    assert payload["safety_event"]["tenant_safe_explanation"] == (
        "Human review is required."
    )
    assert "Candidate text" not in str(payload)


def test_review_queue_rejects_invalid_transitions_and_reviewers(tmp_path):
    queue = HumanReviewQueue(tmp_path / "review.db")
    case = queue.enqueue_case(
        candidate_text="Candidate.",
        evidence_refs=("kb://fact-5",),
    )

    with pytest.raises(ValueError, match="reviewer_id"):
        queue.decide(case.case_id, reviewer_id="", action="approve")
    with pytest.raises(ValueError, match="action"):
        queue.decide(case.case_id, reviewer_id="reviewer-1", action="escalate")

    queue.decide(case.case_id, reviewer_id="reviewer-1", action="approve")
    queue.release(case.case_id, reviewer_id="reviewer-1", release_id="rel-5")

    with pytest.raises(PermissionError, match="already released"):
        queue.decide(case.case_id, reviewer_id="reviewer-1", action="reject")


def test_human_review_queue_public_exports():
    from director_ai import HumanReviewQueue as TopLevelHumanReviewQueue
    from director_ai.core import HumanReviewCase, HumanReviewDecision

    assert TopLevelHumanReviewQueue is HumanReviewQueue
    assert HumanReviewCase.__name__ == "HumanReviewCase"
    assert HumanReviewDecision.__name__ == "HumanReviewDecision"
