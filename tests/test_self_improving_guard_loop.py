# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Self-improving guard loop proposal-gate tests."""

from __future__ import annotations

import pytest

from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.self_evolving import (
    FeedbackEvent,
    InMemoryFeedbackStore,
    SelfImprovingGuardLoop,
)


def _store(n_each: int = 8) -> InMemoryFeedbackStore:
    store = InMemoryFeedbackStore()
    event_index = 0
    for label in ("safe", "unsafe", "false_positive", "false_negative"):
        for item_index in range(n_each):
            store.append(
                FeedbackEvent(
                    prompt=f"reviewed prompt {label} {item_index}",
                    response=f"reviewed response {label} {item_index}",
                    label=label,  # type: ignore[arg-type]
                    metadata={
                        "event_id": f"sevt-{event_index}",
                        "reviewer_id": "reviewer-passport-1",
                    },
                    timestamp=float(event_index),
                )
            )
            event_index += 1
    return store


def _envelope() -> RiskEnvelope:
    return RiskEnvelope(
        action_category="training",
        reversibility="costly",
        domain="regulated",
        calibrated_threshold=0.45,
        no_go_threshold=0.8,
    )


def test_feedback_manifest_requires_reviewed_provenance_and_is_tenant_safe():
    loop = SelfImprovingGuardLoop(
        store=_store(),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    manifest = loop.build_manifest(source_ref="feedback://recent-reviewed")
    payload = manifest.to_dict()

    assert manifest.event_count == 32
    assert manifest.label_counts["false_negative"] == 8
    assert payload["reviewer_ids"] == ["reviewer-passport-1"]
    assert "reviewed prompt" not in str(payload)
    assert "reviewed response" not in str(payload)


def test_calibration_update_proposal_requires_approval_before_application():
    loop = SelfImprovingGuardLoop(
        store=_store(),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    proposal = loop.propose_calibration_update(
        source_ref="feedback://recent-reviewed",
        current_threshold=0.55,
        candidate_threshold=0.58,
        confidence_low=0.51,
        confidence_high=0.61,
        rollback_id="threshold-profile-20260513-a",
        min_feedback=16,
        max_interval_width=0.2,
    )

    assert proposal.proposal_type == "calibration_update"
    assert proposal.guard_decision.decision == "allow"
    assert proposal.approved is False
    with pytest.raises(PermissionError, match="not approved"):
        loop.release(proposal)

    approved = loop.approve(proposal, approval_id="review-20260513-002")

    assert approved.approved is True
    assert loop.release(approved)["candidate_threshold"] == 0.58


def test_wide_calibration_interval_warns_and_cannot_be_approved():
    loop = SelfImprovingGuardLoop(
        store=_store(),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    proposal = loop.propose_calibration_update(
        source_ref="feedback://recent-reviewed",
        current_threshold=0.55,
        candidate_threshold=0.58,
        confidence_low=0.2,
        confidence_high=0.8,
        rollback_id="threshold-profile-20260513-a",
        min_feedback=16,
        max_interval_width=0.2,
    )

    assert proposal.guard_decision.decision == "warn"
    assert proposal.guard_decision.reason == "self_improvement_interval_too_wide"
    with pytest.raises(PermissionError, match="cannot approve"):
        loop.approve(proposal, approval_id="review-wide")


def test_training_job_is_proposal_only_and_requires_rollback_id():
    loop = SelfImprovingGuardLoop(
        store=_store(),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    proposal = loop.propose_lora_job(
        source_ref="feedback://recent-reviewed",
        dataset_uri="env://DIRECTOR_REVIEWED_FEEDBACK_DATASET",
        base_model_ref="registry://guard-base@sha256:abc",
        rollback_id="guard-model-v12",
        heldout_score=0.91,
        baseline_score=0.88,
        min_improvement=0.02,
        min_feedback=16,
    )

    payload = proposal.to_dict()

    assert proposal.proposal_type == "lora_training_job"
    assert proposal.guard_decision.decision == "allow"
    assert payload["submitted"] is False
    assert payload["promotion_status"] == "proposed"
    assert loop.approve(proposal, approval_id="review-train").approved is True


def test_training_job_rejects_dataset_uri_with_embedded_credentials():
    loop = SelfImprovingGuardLoop(
        store=_store(),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    with pytest.raises(ValueError, match="embedded credentials"):
        loop.propose_lora_job(
            source_ref="feedback://recent-reviewed",
            dataset_uri="https://reviewer@example.test/dataset.jsonl",
            base_model_ref="registry://guard-base@sha256:abc",
            rollback_id="guard-model-v12",
            heldout_score=0.91,
            baseline_score=0.88,
        )


def test_manifest_rejects_unreviewed_feedback():
    store = InMemoryFeedbackStore()
    store.append(
        FeedbackEvent(
            prompt="unreviewed prompt",
            response="unreviewed response",
            label="unsafe",
            metadata={"event_id": "sevt-1"},
        )
    )
    loop = SelfImprovingGuardLoop(
        store=store,
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    with pytest.raises(ValueError, match="reviewer_id"):
        loop.build_manifest(source_ref="feedback://missing-reviewer")
