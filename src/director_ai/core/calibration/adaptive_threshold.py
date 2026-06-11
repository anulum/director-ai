# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Adaptive Threshold Learning

"""Human-gated adaptive threshold recommendations.

This module uses a Beta-Bernoulli Thompson-sampling model per threshold
arm. Feedback is replayed counterfactually across every candidate
threshold because a labelled score lets us know how each threshold would
have classified the sample. The learner emits recommendations only; it
does not mutate live scorer configuration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from ..mandatory import mandatory_execution

try:
    from backfire_kernel import rust_beta_posterior_mean

    _RUST_ADAPTIVE = True
except Exception:  # pragma: no cover - mandatory dependency
    _RUST_ADAPTIVE = True

    def rust_beta_posterior_mean(
        _alpha_prior: float,
        _beta_prior: float,
        _successes: int,
        _pulls: int,
    ) -> float:
        raise RuntimeError("backfire_kernel rust_beta_posterior_mean is unavailable")


__all__ = [
    "AdaptiveThresholdArm",
    "AdaptiveThresholdLearner",
    "AdaptiveThresholdRecommendation",
    "AdaptiveThresholdReport",
    "ThresholdFeedback",
]


def _validate_probability(value: float, name: str) -> float:
    """Return a validated finite probability value."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return float(value)


@dataclass(frozen=True)
class ThresholdFeedback:
    """One human-labelled score used for threshold replay."""

    score: float
    human_approved: bool
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate feedback score and replay weight."""
        _validate_probability(self.score, "score")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("weight must be finite and positive")


@dataclass
class AdaptiveThresholdArm:
    """Posterior state and replay metrics for one candidate threshold."""

    threshold: float
    alpha_prior: float = 1.0
    beta_prior: float = 1.0
    pulls: int = 0
    successes: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def __post_init__(self) -> None:
        """Validate arm threshold and Beta prior parameters."""
        _validate_probability(self.threshold, "threshold")
        if self.alpha_prior <= 0.0 or self.beta_prior <= 0.0:
            raise ValueError("Beta priors must be positive")

    @property
    def failures(self) -> int:
        """Return the number of replayed samples this arm misclassified."""
        return self.pulls - self.successes

    @property
    def alpha(self) -> float:
        """Return posterior alpha after successful classifications."""
        return self.alpha_prior + self.successes

    @property
    def beta(self) -> float:
        """Return posterior beta after failed classifications."""
        return self.beta_prior + self.failures

    @property
    def posterior_mean(self) -> float:
        """Return the posterior expected success probability."""
        if _RUST_ADAPTIVE:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                return float(
                    rust_beta_posterior_mean(
                        self.alpha_prior,
                        self.beta_prior,
                        self.successes,
                        self.pulls,
                    )
                )
        return self.alpha / (self.alpha + self.beta)

    @property
    def accuracy(self) -> float:
        """Return empirical accuracy across replayed feedback."""
        return self.successes / self.pulls if self.pulls else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Return the false-positive rate for human-rejected samples."""
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom else 0.0

    @property
    def false_negative_rate(self) -> float:
        """Return the false-negative rate for human-approved samples."""
        denom = self.false_negatives + self.true_positives
        return self.false_negatives / denom if denom else 0.0

    def observe(self, feedback: ThresholdFeedback) -> None:
        """Replay one labelled score against this threshold arm."""
        predicted_approved = feedback.score >= self.threshold
        self.pulls += 1
        correct = predicted_approved == feedback.human_approved
        if correct:
            self.successes += 1
        if predicted_approved and feedback.human_approved:
            self.true_positives += 1
        elif predicted_approved and not feedback.human_approved:
            self.false_positives += 1
        elif not predicted_approved and feedback.human_approved:
            self.false_negatives += 1
        else:
            self.true_negatives += 1

    def sample_success_probability(self, rng: random.Random) -> float:
        """Sample a Thompson posterior success probability for this arm."""
        return rng.betavariate(self.alpha, self.beta)

    def to_dict(self) -> dict[str, Any]:
        """Serialise this arm's posterior and confusion metrics."""
        return {
            "threshold": self.threshold,
            "pulls": self.pulls,
            "successes": self.successes,
            "posterior_mean": self.posterior_mean,
            "accuracy": self.accuracy,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
        }


@dataclass(frozen=True)
class AdaptiveThresholdReport:
    """Snapshot after replaying feedback across threshold arms."""

    total_feedback: int
    current_threshold: float
    best_observed_threshold: float | None
    arms: tuple[AdaptiveThresholdArm, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the threshold replay report."""
        return {
            "total_feedback": self.total_feedback,
            "current_threshold": self.current_threshold,
            "best_observed_threshold": self.best_observed_threshold,
            "arms": [arm.to_dict() for arm in self.arms],
        }


@dataclass(frozen=True)
class AdaptiveThresholdRecommendation:
    """Human-review-gated threshold recommendation."""

    current_threshold: float
    recommended_threshold: float | None
    expected_success_probability: float
    current_success_probability: float
    expected_lift: float
    reason: str
    requires_human_approval: bool = True
    rollback_threshold: float | None = None
    safety_constraints: dict[str, float] = field(default_factory=dict)

    def to_profile_overlay(self, *, profile: str = "adaptive") -> dict[str, object]:
        """Return a profile overlay that can be reviewed before promotion."""
        threshold = (
            self.current_threshold
            if self.recommended_threshold is None
            else self.recommended_threshold
        )
        return {
            "profile": profile,
            "coherence_threshold": threshold,
            "extra": {
                "adaptive_reason": self.reason,
                "adaptive_expected_success_probability": (
                    f"{self.expected_success_probability:.4f}"
                ),
                "adaptive_current_success_probability": (
                    f"{self.current_success_probability:.4f}"
                ),
                "adaptive_expected_lift": f"{self.expected_lift:.4f}",
                "adaptive_requires_human_approval": str(
                    self.requires_human_approval
                ).lower(),
                "adaptive_rollback_threshold": (
                    f"{self.rollback_threshold:.4f}"
                    if self.rollback_threshold is not None
                    else ""
                ),
            },
        }


class AdaptiveThresholdLearner:
    """Offline Thompson-sampling threshold recommender.

    The learner is intentionally side-effect free with respect to runtime
    scorer configuration. Production deployments should route the returned
    recommendation through human review/change-management and keep the
    rollback threshold recorded with the approved overlay.
    """

    def __init__(
        self,
        *,
        candidate_thresholds: list[float] | tuple[float, ...],
        current_threshold: float,
        min_samples: int = 20,
        min_expected_lift: float = 0.01,
        max_false_positive_rate: float = 1.0,
        max_false_negative_rate: float = 1.0,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        random_seed: int | None = None,
    ) -> None:
        if not candidate_thresholds:
            raise ValueError("candidate thresholds must be non-empty")
        thresholds = tuple(
            sorted(
                _validate_probability(t, "candidate threshold")
                for t in candidate_thresholds
            )
        )
        if len(set(thresholds)) != len(thresholds):
            raise ValueError("candidate thresholds must be unique")
        self.current_threshold = _validate_probability(
            current_threshold, "current_threshold"
        )
        if min_samples < 1:
            raise ValueError("min_samples must be positive")
        self.min_samples = min_samples
        self.min_expected_lift = float(min_expected_lift)
        self.max_false_positive_rate = _validate_probability(
            max_false_positive_rate, "max_false_positive_rate"
        )
        self.max_false_negative_rate = _validate_probability(
            max_false_negative_rate, "max_false_negative_rate"
        )
        self._rng = random.Random(random_seed)
        self._arms = {
            threshold: AdaptiveThresholdArm(
                threshold=threshold,
                alpha_prior=alpha_prior,
                beta_prior=beta_prior,
            )
            for threshold in thresholds
        }
        self._feedback_count = 0

    def arm(self, threshold: float) -> AdaptiveThresholdArm:
        """Return the candidate arm for a validated threshold."""
        key = _validate_probability(threshold, "threshold")
        try:
            return self._arms[key]
        except KeyError as exc:
            raise KeyError(f"unknown threshold arm {threshold}") from exc

    def observe(self, score: float, human_approved: bool) -> AdaptiveThresholdReport:
        """Replay one labelled score across all candidate thresholds."""
        feedback = ThresholdFeedback(score=score, human_approved=human_approved)
        for arm in self._arms.values():
            arm.observe(feedback)
        self._feedback_count += 1
        return self.report()

    def observe_batch(
        self,
        feedback: list[ThresholdFeedback] | tuple[ThresholdFeedback, ...],
    ) -> AdaptiveThresholdReport:
        """Replay a batch of labelled feedback across all candidate thresholds."""
        for item in feedback:
            for arm in self._arms.values():
                arm.observe(item)
            self._feedback_count += 1
        return self.report()

    def report(self) -> AdaptiveThresholdReport:
        """Return the current replay summary without making a recommendation."""
        eligible = [arm for arm in self._arms.values() if arm.pulls > 0]
        best = (
            max(eligible, key=lambda arm: arm.accuracy).threshold if eligible else None
        )
        return AdaptiveThresholdReport(
            total_feedback=self._feedback_count,
            current_threshold=self.current_threshold,
            best_observed_threshold=best,
            arms=tuple(self._arms.values()),
        )

    def recommend(self) -> AdaptiveThresholdRecommendation:
        """Return a human-gated threshold recommendation from replayed evidence."""
        current = self._current_arm_or_nearest()
        constraints = {
            "max_false_positive_rate": self.max_false_positive_rate,
            "max_false_negative_rate": self.max_false_negative_rate,
        }
        if self._feedback_count < self.min_samples:
            return AdaptiveThresholdRecommendation(
                current_threshold=self.current_threshold,
                recommended_threshold=None,
                expected_success_probability=current.posterior_mean,
                current_success_probability=current.posterior_mean,
                expected_lift=0.0,
                reason="insufficient_feedback",
                rollback_threshold=self.current_threshold,
                safety_constraints=constraints,
            )

        feasible = [
            arm
            for arm in self._arms.values()
            if arm.false_positive_rate <= self.max_false_positive_rate
            and arm.false_negative_rate <= self.max_false_negative_rate
        ]
        if not feasible:
            return AdaptiveThresholdRecommendation(
                current_threshold=self.current_threshold,
                recommended_threshold=None,
                expected_success_probability=current.posterior_mean,
                current_success_probability=current.posterior_mean,
                expected_lift=0.0,
                reason="no_candidate_satisfies_safety_constraints",
                rollback_threshold=self.current_threshold,
                safety_constraints=constraints,
            )

        sampled_winner = max(
            feasible,
            key=lambda arm: (
                arm.posterior_mean,
                arm.sample_success_probability(self._rng),
                -abs(arm.threshold - self.current_threshold),
            ),
        )
        lift = sampled_winner.posterior_mean - current.posterior_mean
        if lift < self.min_expected_lift:
            return AdaptiveThresholdRecommendation(
                current_threshold=self.current_threshold,
                recommended_threshold=self.current_threshold,
                expected_success_probability=current.posterior_mean,
                current_success_probability=current.posterior_mean,
                expected_lift=0.0,
                reason="expected_lift_below_minimum",
                rollback_threshold=self.current_threshold,
                safety_constraints=constraints,
            )
        return AdaptiveThresholdRecommendation(
            current_threshold=self.current_threshold,
            recommended_threshold=sampled_winner.threshold,
            expected_success_probability=sampled_winner.posterior_mean,
            current_success_probability=current.posterior_mean,
            expected_lift=lift,
            reason="candidate_passed_offline_replay_and_safety_constraints",
            rollback_threshold=self.current_threshold,
            safety_constraints=constraints,
        )

    def _current_arm_or_nearest(self) -> AdaptiveThresholdArm:
        """Return the exact current-threshold arm or nearest candidate."""
        if self.current_threshold in self._arms:
            return self._arms[self.current_threshold]
        return min(
            self._arms.values(),
            key=lambda arm: abs(arm.threshold - self.current_threshold),
        )
