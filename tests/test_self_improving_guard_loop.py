# SPDX-License-Identifier: Apache-2.0
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
    GuardLoopProposal,
    InMemoryFeedbackStore,
    ReviewedFeedbackManifest,
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


def test_guard_loop_validates_required_identifiers_and_empty_feedback():
    with pytest.raises(ValueError, match="policy_id"):
        SelfImprovingGuardLoop(
            store=_store(),
            risk_envelope=_envelope(),
            policy_id="",
        )

    loop = SelfImprovingGuardLoop(
        store=InMemoryFeedbackStore(),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )
    with pytest.raises(ValueError, match="source_ref"):
        loop.build_manifest(source_ref="")
    with pytest.raises(ValueError, match="reviewed events"):
        loop.build_manifest(source_ref="feedback://empty")


def test_guard_loop_rejects_invalid_calibration_inputs():
    loop = SelfImprovingGuardLoop(
        store=_store(),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    invalid_calls = [
        {"current_threshold": -0.1},
        {"candidate_threshold": 1.1},
        {"confidence_low": 0.8, "confidence_high": 0.2},
        {"max_interval_width": 1.1},
        {"min_feedback": 0},
    ]
    for override in invalid_calls:
        kwargs = {
            "source_ref": "feedback://recent-reviewed",
            "current_threshold": 0.5,
            "candidate_threshold": 0.6,
            "confidence_low": 0.4,
            "confidence_high": 0.6,
            "rollback_id": "threshold-profile-rollback",
            "min_feedback": 16,
            "max_interval_width": 0.2,
            **override,
        }
        with pytest.raises(ValueError):
            loop.propose_calibration_update(**kwargs)


def test_guard_loop_warns_for_insufficient_feedback_and_training_regression():
    loop = SelfImprovingGuardLoop(
        store=_store(n_each=1),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    calibration = loop.propose_calibration_update(
        source_ref="feedback://small-reviewed",
        current_threshold=0.55,
        candidate_threshold=0.56,
        confidence_low=0.5,
        confidence_high=0.57,
        rollback_id="threshold-profile-rollback",
        min_feedback=32,
    )
    training = loop.propose_lora_job(
        source_ref="feedback://small-reviewed",
        dataset_uri="env://DIRECTOR_REVIEWED_FEEDBACK_DATASET",
        base_model_ref="registry://guard-base@sha256:abc",
        rollback_id="guard-model-v12",
        heldout_score=0.86,
        baseline_score=0.88,
        min_improvement=0.02,
        min_feedback=1,
    )

    assert calibration.guard_decision.reason == "self_improvement_insufficient_feedback"
    assert training.guard_decision.reason == "self_improvement_heldout_regression"
    assert training.guard_decision.risk_score == pytest.approx(0.04)


def test_guard_loop_rejects_invalid_training_inputs_and_serialises_event():
    loop = SelfImprovingGuardLoop(
        store=_store(),
        risk_envelope=_envelope(),
        policy_id="policy.self_improving.regulated",
    )

    invalid_calls = [
        {"dataset_uri": ""},
        {"base_model_ref": ""},
        {"heldout_score": -0.1},
        {"baseline_score": 1.1},
        {"min_improvement": -0.01},
        {"min_feedback": 0},
    ]
    for override in invalid_calls:
        kwargs = {
            "source_ref": "feedback://recent-reviewed",
            "dataset_uri": "env://DIRECTOR_REVIEWED_FEEDBACK_DATASET",
            "base_model_ref": "registry://guard-base@sha256:abc",
            "rollback_id": "guard-model-v12",
            "heldout_score": 0.91,
            "baseline_score": 0.88,
            **override,
        }
        with pytest.raises(ValueError):
            loop.propose_lora_job(**kwargs)

    proposal = loop.propose_lora_job(
        source_ref="feedback://recent-reviewed",
        dataset_uri="env://DIRECTOR_REVIEWED_FEEDBACK_DATASET",
        base_model_ref="registry://guard-base@sha256:abc",
        rollback_id="guard-model-v12",
        heldout_score=0.91,
        baseline_score=0.88,
        min_feedback=16,
    )
    event = proposal.to_safety_event(hook_id="guard.loop", tenant_id="tenant-a")

    assert event.hook_id == "guard.loop"
    assert event.tenant_id == "tenant-a"


def test_guard_loop_dataclasses_reject_invalid_states():
    manifest = ReviewedFeedbackManifest(
        manifest_id="manifest-1",
        source_ref="feedback://reviewed",
        event_count=1,
        label_counts={"unsafe": 1},
        reviewer_ids=("reviewer-1",),
        event_refs=("sevt-1",),
    )
    decision = loop_decision = (
        SelfImprovingGuardLoop(
            store=_store(),
            risk_envelope=_envelope(),
            policy_id="policy.self_improving.regulated",
        )
        .propose_calibration_update(
            source_ref="feedback://recent-reviewed",
            current_threshold=0.55,
            candidate_threshold=0.56,
            confidence_low=0.5,
            confidence_high=0.57,
            rollback_id="threshold-profile-rollback",
            min_feedback=16,
        )
        .guard_decision
    )

    with pytest.raises(ValueError, match="manifest_id"):
        ReviewedFeedbackManifest("", "feedback://reviewed", 1, {}, (), ("sevt-1",))
    with pytest.raises(ValueError, match="source_ref"):
        ReviewedFeedbackManifest("manifest-1", "", 1, {}, (), ("sevt-1",))
    with pytest.raises(ValueError, match="event_count"):
        ReviewedFeedbackManifest("manifest-1", "feedback://reviewed", 0, {}, (), ())
    with pytest.raises(ValueError, match="proposal_id"):
        GuardLoopProposal("", "calibration_update", manifest, "rollback", decision, {})
    with pytest.raises(ValueError, match="unsupported proposal_type"):
        GuardLoopProposal("proposal-1", "bad", manifest, "rollback", decision, {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rollback_id"):
        GuardLoopProposal(
            "proposal-1", "calibration_update", manifest, "", decision, {}
        )
    with pytest.raises(ValueError, match="must not submit"):
        GuardLoopProposal(
            "proposal-1",
            "calibration_update",
            manifest,
            "rollback",
            loop_decision,
            {},
            submitted=True,
        )
    with pytest.raises(ValueError, match="approval_id"):
        GuardLoopProposal(
            "proposal-1",
            "calibration_update",
            manifest,
            "rollback",
            decision,
            {},
            approved=True,
        )
