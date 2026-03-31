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

__all__ = ["ConformalPredictor", "PredictionInterval"]


@dataclass
class PredictionInterval:
    """Calibrated prediction interval for hallucination probability."""

    point_estimate: float  # P(hallucination) point estimate from score
    lower: float  # lower bound (e.g., 5%)
    upper: float  # upper bound (e.g., 15%)
    coverage: float  # target coverage (e.g., 0.95)
    calibration_size: int  # how many calibration examples used
    is_reliable: bool  # True if calibration_size >= min_samples


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

        residuals.sort()

        # Conformal quantile: ceil((n+1) * coverage) / n
        q_idx = math.ceil((n + 1) * self._coverage) - 1
        q_idx = min(q_idx, n - 1)
        return residuals[q_idx]

    @staticmethod
    def _score_to_prob(score: float) -> float:
        """Convert coherence score to hallucination probability.

        Simple inversion: P(hallucination) ≈ 1 - score.
        """
        return max(0.0, min(1.0, 1.0 - score))
