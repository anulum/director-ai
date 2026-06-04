# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Conformal prediction intervals for guardrail decisions.

Provides calibrated, distribution-free uncertainty estimates on
hallucination probability. Instead of binary approved/rejected,
returns a prediction interval: "95% confident hallucination
probability is between 5% and 15%."

Implements split conformal prediction using accumulated human
feedback from FeedbackStore as calibration data.

Reference: Mohri & Hashimoto (ICML 2024), "Conformal Factuality."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ConformalPredictor",
    "ConformalRoutingDecision",
    "ConformalRoutingPolicy",
    "PredictionInterval",
    "RoutingAction",
]

RoutingAction = Literal["allow", "human_review", "escalate", "reject"]

try:
    from backfire_kernel import rust_conformal_quantile

    _RUST_CONFORMAL = True
except Exception:  # pragma: no cover - mandatory dependency
    _RUST_CONFORMAL = True

    def rust_conformal_quantile(_residuals: list[float], _coverage: float) -> float:
        raise RuntimeError("backfire_kernel rust_conformal_quantile is unavailable")


@dataclass
class PredictionInterval:
    """Calibrated prediction interval for hallucination probability."""

    point_estimate: float  # P(hallucination) point estimate from score
    lower: float  # lower bound (e.g., 5%)
    upper: float  # upper bound (e.g., 15%)
    coverage: float  # target coverage (e.g., 0.95)
    calibration_size: int  # how many calibration examples used
    is_reliable: bool  # True if calibration_size >= min_samples


@dataclass(frozen=True)
class ConformalRoutingDecision:
    """Operational route selected from a calibrated risk interval."""

    action: RoutingAction
    reason: str
    route_to: str
    risk_lower: float
    risk_upper: float
    coverage: float
    calibration_size: int
    is_reliable: bool
    interval: PredictionInterval


class ConformalRoutingPolicy:
    """Conservative routing policy for calibrated hallucination risk.

    The policy intentionally routes on interval bounds instead of the point
    estimate. A response is allowed only when the upper risk bound is low
    enough, rejected only when the lower bound is high enough, and ambiguous
    intervals are sent to human review or a stronger model path.
    """

    def __init__(
        self,
        *,
        allow_max_risk: float = 0.05,
        escalate_min_risk: float = 0.20,
        reject_min_risk: float = 0.70,
        min_samples: int = 30,
        allow_route: str = "current_path",
        human_review_route: str = "human_review",
        escalation_route: str = "stronger_model",
        reject_route: str = "reject",
    ) -> None:
        if not 0.0 <= allow_max_risk < escalate_min_risk < reject_min_risk <= 1.0:
            raise ValueError(
                "expected 0 <= allow_max_risk < escalate_min_risk "
                "< reject_min_risk <= 1"
            )
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self.allow_max_risk = allow_max_risk
        self.escalate_min_risk = escalate_min_risk
        self.reject_min_risk = reject_min_risk
        self.min_samples = min_samples
        self.allow_route = allow_route
        self.human_review_route = human_review_route
        self.escalation_route = escalation_route
        self.reject_route = reject_route

    def decide(self, interval: PredictionInterval) -> ConformalRoutingDecision:
        """Choose an operational route from a conformal risk interval."""
        if not interval.is_reliable or interval.calibration_size < self.min_samples:
            return self._decision(
                "human_review",
                "unreliable_calibration",
                self.human_review_route,
                interval,
            )
        if interval.upper <= self.allow_max_risk:
            return self._decision(
                "allow",
                "upper_risk_within_allowance",
                self.allow_route,
                interval,
            )
        if interval.lower >= self.reject_min_risk:
            return self._decision(
                "reject",
                "lower_risk_exceeds_reject_threshold",
                self.reject_route,
                interval,
            )
        if interval.upper >= self.escalate_min_risk:
            return self._decision(
                "escalate",
                "upper_risk_exceeds_escalation_threshold",
                self.escalation_route,
                interval,
            )
        return self._decision(
            "human_review",
            "ambiguous_calibrated_interval",
            self.human_review_route,
            interval,
        )

    @staticmethod
    def _decision(
        action: RoutingAction,
        reason: str,
        route_to: str,
        interval: PredictionInterval,
    ) -> ConformalRoutingDecision:
        return ConformalRoutingDecision(
            action=action,
            reason=reason,
            route_to=route_to,
            risk_lower=interval.lower,
            risk_upper=interval.upper,
            coverage=interval.coverage,
            calibration_size=interval.calibration_size,
            is_reliable=interval.is_reliable,
            interval=interval,
        )


class ConformalPredictor:
    """Split conformal prediction for hallucination probability.

    Uses nonconformity scores derived from (guardrail_score, human_label)
    pairs to construct prediction intervals.

    Parameters
    ----------
    coverage : float
        Target coverage probability (e.g., 0.95 for 95% intervals).
    min_samples : int
        Minimum calibration samples for reliable intervals. Below
        this, intervals are returned but marked unreliable.
    """

    def __init__(self, coverage: float = 0.95, min_samples: int = 30):
        if not 0 < coverage < 1:
            raise ValueError(f"coverage must be in (0, 1), got {coverage}")
        self._coverage = coverage
        self._min_samples = min_samples
        self._scores: list[float] = []
        self._labels: list[bool] = []  # True = was actually hallucination
        self._quantile: float | None = None

    def calibrate(self, scores: list[float], labels: list[bool]) -> None:
        """Calibrate from (score, label) pairs.

        Parameters
        ----------
        scores : list[float]
            Guardrail coherence scores (higher = more coherent).
        labels : list[bool]
            True if the response was actually a hallucination
            (human-verified).
        """
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have same length")
        self._scores = list(scores)
        self._labels = list(labels)
        self._quantile = self._compute_quantile()

    def add_observation(self, score: float, correct_label: bool) -> None:
        """Add one human-labelled observation and refresh calibration.

        ``correct_label=True`` means the checked response was correct, while
        the conformal label stores whether it was actually a hallucination.
        """
        self._scores.append(score)
        self._labels.append(not correct_label)
        self._quantile = self._compute_quantile()

    def calibrate_from_feedback(self, feedback_store) -> None:
        """Calibrate from a FeedbackStore instance.

        Reads all entries where human_label is not None and uses
        (score, human_label) as calibration data.
        """
        entries = feedback_store.query()
        scores = []
        labels = []
        for e in entries:
            if e.human_label is not None:
                scores.append(e.score)
                labels.append(not e.human_label)  # human_label True = correct
            elif hasattr(e, "approved") and hasattr(e, "human_override"):
                if e.human_override is not None:
                    scores.append(e.score)
                    labels.append(e.human_override != e.approved)
        self.calibrate(scores, labels)

    def predict(self, score: float) -> PredictionInterval:
        """Predict hallucination probability interval for a new score.

        Parameters
        ----------
        score : float
            Guardrail coherence score for the new response.

        Returns
        -------
        PredictionInterval
            Calibrated interval with coverage guarantee.
        """
        n = len(self._scores)
        point_est = self._score_to_prob(score)

        if n == 0 or self._quantile is None:
            return PredictionInterval(
                point_estimate=point_est,
                lower=0.0,
                upper=1.0,
                coverage=self._coverage,
                calibration_size=0,
                is_reliable=False,
            )

        half_width = self._quantile
        lower = max(0.0, point_est - half_width)
        upper = min(1.0, point_est + half_width)

        return PredictionInterval(
            point_estimate=point_est,
            lower=lower,
            upper=upper,
            coverage=self._coverage,
            calibration_size=n,
            is_reliable=n >= self._min_samples,
        )

    def predict_interval(self, score: float) -> tuple[float, float]:
        """Return the interval tuple expected by ProductionGuard."""
        interval = self.predict(score)
        return (interval.lower, interval.upper)

    def route(
        self,
        score: float,
        policy: ConformalRoutingPolicy | None = None,
    ) -> ConformalRoutingDecision:
        """Route a score using calibrated uncertainty bounds."""
        interval = self.predict(score)
        routing_policy = policy or ConformalRoutingPolicy(min_samples=self._min_samples)
        return routing_policy.decide(interval)

    def _compute_quantile(self) -> float | None:
        """Compute the conformal quantile from calibration data.

        Uses the nonconformity score: |predicted_prob - actual_label|.
        The quantile at level ceil((n+1)*coverage)/n gives the interval
        half-width.
        """
        n = len(self._scores)
        if n == 0:
            return None

        # Nonconformity scores: absolute residuals
        residuals = []
        for s, lab in zip(self._scores, self._labels, strict=True):
            pred_prob = self._score_to_prob(s)
            actual = 1.0 if lab else 0.0
            residuals.append(abs(pred_prob - actual))

        if _RUST_CONFORMAL:
            try:
                return float(rust_conformal_quantile(residuals, self._coverage))
            except Exception:
                pass

        return self._python_quantile(residuals, self._coverage)

    @staticmethod
    def _python_quantile(residuals: list[float], coverage: float) -> float:
        """Pure-Python split-conformal quantile fallback."""
        n = len(residuals)
        residuals.sort()
        q_idx = math.ceil((n + 1) * coverage) - 1
        q_idx = min(q_idx, n - 1)
        return residuals[q_idx]

    @staticmethod
    def _score_to_prob(score: float) -> float:
        """Convert coherence score to hallucination probability.

        Simple inversion: P(hallucination) ≈ 1 - score.
        """
        return max(0.0, min(1.0, 1.0 - score))
