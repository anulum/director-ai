# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Adaptive Threshold Tests

import pytest

import director_ai.core.calibration.adaptive_threshold as adaptive_mod
from director_ai.core import (
    AdaptiveThresholdArm,
    AdaptiveThresholdLearner,
    ThresholdFeedback,
)


def _feedback() -> list[ThresholdFeedback]:
    return [
        ThresholdFeedback(score=0.92, human_approved=True),
        ThresholdFeedback(score=0.83, human_approved=True),
        ThresholdFeedback(score=0.74, human_approved=True),
        ThresholdFeedback(score=0.62, human_approved=True),
        ThresholdFeedback(score=0.49, human_approved=False),
        ThresholdFeedback(score=0.42, human_approved=False),
        ThresholdFeedback(score=0.31, human_approved=False),
        ThresholdFeedback(score=0.22, human_approved=False),
    ]


def test_updates_all_threshold_arms_from_labelled_feedback():
    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.3, 0.5, 0.7],
        current_threshold=0.5,
        min_samples=4,
        random_seed=7,
    )

    report = learner.observe_batch(_feedback())

    assert report.total_feedback == 8
    assert report.best_observed_threshold == 0.5
    assert learner.arm(0.5).pulls == 8
    assert learner.arm(0.5).successes == 8
    assert learner.arm(0.7).false_negatives == 1


def test_threshold_feedback_rejects_invalid_weight():
    with pytest.raises(ValueError, match="weight"):
        ThresholdFeedback(score=0.5, human_approved=True, weight=0.0)


def test_adaptive_arm_rejects_invalid_priors_and_reports_empty_metrics():
    with pytest.raises(ValueError, match="Beta priors"):
        AdaptiveThresholdArm(threshold=0.5, alpha_prior=0.0)

    arm = AdaptiveThresholdArm(threshold=0.5)

    assert arm.accuracy == 0.0
    assert arm.false_positive_rate == 0.0
    assert arm.false_negative_rate == 0.0


def test_adaptive_arm_observe_records_all_confusion_outcomes():
    arm = AdaptiveThresholdArm(threshold=0.5)

    arm.observe(ThresholdFeedback(score=0.9, human_approved=True))
    arm.observe(ThresholdFeedback(score=0.9, human_approved=False))
    arm.observe(ThresholdFeedback(score=0.1, human_approved=True))
    arm.observe(ThresholdFeedback(score=0.1, human_approved=False))

    assert arm.true_positives == 1
    assert arm.false_positives == 1
    assert arm.false_negatives == 1
    assert arm.true_negatives == 1
    assert arm.successes == 2
    assert arm.false_positive_rate == pytest.approx(0.5)
    assert arm.false_negative_rate == pytest.approx(0.5)


def test_recommendation_requires_human_approval_and_has_rollback_overlay():
    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.3, 0.5, 0.7],
        current_threshold=0.3,
        min_samples=8,
        min_expected_lift=0.05,
        random_seed=11,
    )
    learner.observe_batch(_feedback())

    recommendation = learner.recommend()

    assert recommendation.recommended_threshold == 0.5
    assert recommendation.current_threshold == 0.3
    assert recommendation.requires_human_approval is True
    assert recommendation.rollback_threshold == 0.3
    assert recommendation.expected_lift > 0.05
    overlay = recommendation.to_profile_overlay(profile="adaptive-medical")
    assert overlay["coherence_threshold"] == 0.5
    assert overlay["extra"]["adaptive_requires_human_approval"] == "true"
    assert overlay["extra"]["adaptive_rollback_threshold"] == "0.3000"


def test_returns_noop_when_data_is_insufficient():
    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.4, 0.6],
        current_threshold=0.4,
        min_samples=10,
        random_seed=3,
    )
    learner.observe(score=0.8, human_approved=True)

    recommendation = learner.recommend()

    assert recommendation.recommended_threshold is None
    assert recommendation.reason == "insufficient_feedback"
    assert recommendation.requires_human_approval is True


def test_no_candidate_satisfies_safety_constraints():
    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.4, 0.8],
        current_threshold=0.4,
        min_samples=4,
        max_false_positive_rate=0.0,
        max_false_negative_rate=0.0,
        random_seed=5,
    )
    learner.observe_batch(
        [
            ThresholdFeedback(score=0.9, human_approved=False),
            ThresholdFeedback(score=0.85, human_approved=False),
            ThresholdFeedback(score=0.2, human_approved=True),
            ThresholdFeedback(score=0.25, human_approved=True),
        ]
    )

    recommendation = learner.recommend()

    assert recommendation.recommended_threshold is None
    assert recommendation.reason == "no_candidate_satisfies_safety_constraints"


def test_safety_constraints_block_high_false_negative_threshold():
    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.4, 0.8],
        current_threshold=0.4,
        min_samples=4,
        max_false_negative_rate=0.0,
        min_expected_lift=-1.0,
        random_seed=5,
    )
    for item in _feedback():
        learner.observe(item.score, item.human_approved)

    recommendation = learner.recommend()

    assert recommendation.recommended_threshold == 0.4
    assert recommendation.safety_constraints["max_false_negative_rate"] == 0.0
    assert learner.arm(0.8).false_negative_rate > 0.0


def test_rejects_invalid_candidates_and_feedback():
    with pytest.raises(ValueError, match="candidate thresholds"):
        AdaptiveThresholdLearner(candidate_thresholds=[], current_threshold=0.5)
    with pytest.raises(ValueError, match="unique"):
        AdaptiveThresholdLearner(candidate_thresholds=[0.5, 0.5], current_threshold=0.5)
    with pytest.raises(ValueError, match="current_threshold"):
        AdaptiveThresholdLearner(candidate_thresholds=[0.4], current_threshold=1.5)
    with pytest.raises(ValueError, match="min_samples"):
        AdaptiveThresholdLearner(
            candidate_thresholds=[0.4],
            current_threshold=0.4,
            min_samples=0,
        )

    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.4], current_threshold=0.4
    )
    with pytest.raises(ValueError, match="score"):
        learner.observe(score=float("nan"), human_approved=True)
    with pytest.raises(KeyError, match="unknown threshold arm"):
        learner.arm(0.6)


def test_report_and_recommendation_serialise_stable_shapes():
    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.4, 0.6],
        current_threshold=0.5,
        min_samples=2,
        random_seed=13,
    )
    report = learner.observe_batch(_feedback()[:2])
    recommendation = learner.recommend()

    report_payload = report.to_dict()
    overlay = recommendation.to_profile_overlay()

    assert report_payload["current_threshold"] == 0.5
    assert len(report_payload["arms"]) == 2
    assert overlay["profile"] == "adaptive"
    assert overlay["extra"]["adaptive_expected_lift"]


def test_rust_posterior_kernel_is_used_when_available(monkeypatch):
    monkeypatch.setattr(adaptive_mod, "_RUST_ADAPTIVE", True)
    calls = {"count": 0}

    def _posterior(alpha_prior, beta_prior, successes, pulls):
        calls["count"] += 1
        return (alpha_prior + successes) / (alpha_prior + beta_prior + pulls)

    monkeypatch.setattr(
        adaptive_mod,
        "rust_beta_posterior_mean",
        _posterior,
        raising=True,
    )
    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.4], current_threshold=0.4
    )
    learner.observe_batch(_feedback())
    value = learner.arm(0.4).posterior_mean
    assert value > 0.0
    assert calls["count"] >= 1


def test_rust_posterior_type_error_is_mandatory_failure(monkeypatch):
    monkeypatch.setattr(adaptive_mod, "_RUST_ADAPTIVE", True)
    monkeypatch.setattr(
        adaptive_mod,
        "rust_beta_posterior_mean",
        lambda *_args: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
        raising=True,
    )
    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.4], current_threshold=0.4
    )
    learner.observe_batch(_feedback())
    with pytest.raises(TypeError, match="ffi signature mismatch"):
        _ = learner.arm(0.4).posterior_mean
