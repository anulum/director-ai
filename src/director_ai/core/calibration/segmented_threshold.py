# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-segment adaptive thresholds

"""Learn a separate halt threshold per domain, model, or tenant.

:class:`~director_ai.core.calibration.adaptive_threshold.AdaptiveThresholdLearner`
learns one global threshold from human-labelled feedback. But the right halt
threshold is not global: a clinical-domain answer and a casual chat tolerate very
different false-positive rates, and two models hallucinate at different rates.
This wraps the global learner with a per-segment routing layer so each segment
(any string key — ``"clinical"``, a model id, a tenant id, or a composite)
accumulates its own evidence and gets its own recommendation.

Cold start is handled by a global pool that sees *every* observation: a segment
with fewer than ``min_samples`` of its own feedback falls back to the pooled
recommendation, so a brand-new segment behaves sensibly until it has earned its
own threshold. Each per-segment learner shares the global learner's candidate
grid, safety constraints, and priors, so the segmentation is purely a routing and
bookkeeping layer — the Beta-posterior arithmetic (and its Rust
``rust_beta_posterior_mean`` fast path) is inherited unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adaptive_threshold import (
    AdaptiveThresholdLearner,
    AdaptiveThresholdRecommendation,
    AdaptiveThresholdReport,
)

__all__ = [
    "SegmentRecommendation",
    "SegmentedThresholdLearner",
]

_GLOBAL_SEGMENT = "__global__"


@dataclass(frozen=True)
class SegmentRecommendation:
    """A per-segment recommendation and where its evidence came from."""

    segment: str
    source: str  # "segment" once the segment has enough feedback, else "global"
    feedback_count: int
    recommendation: AdaptiveThresholdRecommendation


class SegmentedThresholdLearner:
    """Route adaptive-threshold feedback and recommendations by segment.

    Parameters mirror
    :class:`~director_ai.core.calibration.adaptive_threshold.AdaptiveThresholdLearner`;
    they are stored and used to build each per-segment learner on first use. The
    ``promote_after`` count (defaulting to the learner's ``min_samples``) is the
    per-segment feedback at or above which the segment's own recommendation is
    used instead of the global pool's.
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
        promote_after: int | None = None,
    ) -> None:
        self._learner_kwargs: dict[str, Any] = {
            "candidate_thresholds": tuple(candidate_thresholds),
            "current_threshold": current_threshold,
            "min_samples": min_samples,
            "min_expected_lift": min_expected_lift,
            "max_false_positive_rate": max_false_positive_rate,
            "max_false_negative_rate": max_false_negative_rate,
            "alpha_prior": alpha_prior,
            "beta_prior": beta_prior,
        }
        if promote_after is not None and promote_after < 1:
            raise ValueError("promote_after must be positive")
        self._promote_after = (
            promote_after if promote_after is not None else min_samples
        )
        self._base_seed = random_seed
        self._seed_counter = 0
        self._segments: dict[str, AdaptiveThresholdLearner] = {}
        self._counts: dict[str, int] = {}
        self._global = self._new_learner()

    def _new_learner(self) -> AdaptiveThresholdLearner:
        seed: int | None = None
        if self._base_seed is not None:
            seed = self._base_seed + self._seed_counter
            self._seed_counter += 1
        return AdaptiveThresholdLearner(random_seed=seed, **self._learner_kwargs)

    @staticmethod
    def _key(segment: str) -> str:
        key = segment.strip()
        if not key:
            raise ValueError("segment must be a non-empty string")
        if key == _GLOBAL_SEGMENT:
            raise ValueError(f"{_GLOBAL_SEGMENT!r} is reserved")
        return key

    def _learner_for(self, segment: str) -> AdaptiveThresholdLearner:
        learner = self._segments.get(segment)
        if learner is None:
            learner = self._new_learner()
            self._segments[segment] = learner
            self._counts[segment] = 0
        return learner

    def observe(
        self, score: float, human_approved: bool, *, segment: str
    ) -> AdaptiveThresholdReport:
        """Record one labelled score for *segment* and the global pool."""
        key = self._key(segment)
        learner = self._learner_for(key)
        report = learner.observe(score, human_approved)
        self._counts[key] += 1
        self._global.observe(score, human_approved)
        return report

    def recommend(self, *, segment: str) -> SegmentRecommendation:
        """Recommend a threshold for *segment*, falling back to the global pool.

        The segment's own learner is used once it has at least ``promote_after``
        observations; until then the pooled recommendation is returned so a cold
        segment is never starved of guidance.
        """
        key = self._key(segment)
        count = self._counts.get(key, 0)
        if count >= self._promote_after:
            return SegmentRecommendation(
                segment=key,
                source="segment",
                feedback_count=count,
                recommendation=self._segments[key].recommend(),
            )
        return SegmentRecommendation(
            segment=key,
            source="global",
            feedback_count=count,
            recommendation=self._global.recommend(),
        )

    def report(self, *, segment: str | None = None) -> AdaptiveThresholdReport:
        """Return the replay report for *segment*, or the global pool if None."""
        if segment is None:
            return self._global.report()
        return self._learner_for(self._key(segment)).report()

    def segments(self) -> list[str]:
        """Return the known segment keys in first-seen order."""
        return list(self._segments)

    def to_dict(self) -> dict[str, object]:
        """Serialise per-segment feedback counts and reports plus the global pool."""
        return {
            "promote_after": self._promote_after,
            "global": self._global.report().to_dict(),
            "segments": {
                key: {
                    "feedback_count": self._counts[key],
                    "report": learner.report().to_dict(),
                }
                for key, learner in self._segments.items()
            },
        }
