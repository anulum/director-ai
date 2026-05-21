# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Adaptive Threshold Tests

import pytest

import director_ai.core.calibration.adaptive_threshold as adaptive_mod
from director_ai.core import (
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

    learner = AdaptiveThresholdLearner(
        candidate_thresholds=[0.4], current_threshold=0.4
    )
    with pytest.raises(ValueError, match="score"):
        learner.observe(score=float("nan"), human_approved=True)


def test_rust_posterior_kernel_is_used_when_available(monkeypatch):
    monkeypatch.setattr(adaptive_mod, "_RUST_ADAPTIVE", True)
    calls = {"count": 0}

    def _posterior(alpha_prior, beta_prior, successes, pulls):
        calls["count"] += 1
        return (alpha_prior + successes) / (
            alpha_prior + beta_prior + pulls
        )

    monkeypatch.setattr(
        adaptive_mod,
        "rust_beta_posterior_mean",
        _posterior,
        raising=True,
    )
    learner = AdaptiveThresholdLearner(candidate_thresholds=[0.4], current_threshold=0.4)
    learner.observe_batch(_feedback())
    value = learner.arm(0.4).posterior_mean
    assert value > 0.0
    assert calls["count"] >= 1


def test_rust_posterior_type_error_falls_back_to_python(monkeypatch):
    monkeypatch.setattr(adaptive_mod, "_RUST_ADAPTIVE", True)
    monkeypatch.setattr(
        adaptive_mod,
        "rust_beta_posterior_mean",
        lambda *_args: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
        raising=True,
    )
    learner = AdaptiveThresholdLearner(candidate_thresholds=[0.4], current_threshold=0.4)
    learner.observe_batch(_feedback())
    value = learner.arm(0.4).posterior_mean
    assert 0.0 <= value <= 1.0
