# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HierarchicalAligner

"""Compose per-scale :class:`ScaleScorer` outputs into a single
:class:`AlignmentReport`.

The aligner takes one scorer per scale, evaluates an action
against each, and folds the per-scale scores into:

* A :class:`ScaleScoreTable` — raw per-scale scores.
* A composite alignment score in ``[0, 1]`` — weighted mean of
  the per-scale scores using a caller-supplied weight vector.
* A list of scales whose score fell below the allow threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .scorer import Action, AlignmentScale, ScaleScorer

_ORDER: tuple[AlignmentScale, ...] = ("agent", "swarm", "org", "planetary")


@dataclass(frozen=True)
class ScaleScoreTable:
    """Per-scale score mapping.

    Iteration order is ``agent`` → ``swarm`` → ``org`` →
    ``planetary``; missing scales are absent from the mapping
    rather than defaulting to a sentinel so callers can tell
    "unobserved" from "observed zero".
    """

    scores: Mapping[AlignmentScale, float]

    def __getitem__(self, scale: AlignmentScale) -> float:
        return self.scores[scale]

    def __contains__(self, scale: object) -> bool:
        return scale in self.scores

    def ordered(self) -> tuple[tuple[AlignmentScale, float], ...]:
        return tuple(
            (scale, self.scores[scale]) for scale in _ORDER if scale in self.scores
        )


@dataclass(frozen=True)
class AlignmentReport:
    """Result of one :meth:`HierarchicalAligner.evaluate` call."""

    action: Action
    table: ScaleScoreTable
    composite: float
    failing_scales: tuple[AlignmentScale, ...] = field(default_factory=tuple)

    @property
    def aligned(self) -> bool:
        """``True`` when every observed scale cleared the allow
        threshold — no failing scales."""
        return not self.failing_scales


class HierarchicalAligner:
    """Compose per-scale :class:`ScaleScorer` outputs.

    Parameters
    ----------
    scorers :
        Non-empty sequence of :class:`ScaleScorer`. Every scorer
        must carry a distinct ``scale``.
    weights :
        Optional weight mapping from scale to non-negative float.
        Missing scales inherit a uniform weight. Weights are
        normalised to sum to 1.0 at construction.
    allow_threshold :
        A per-scale score below this is flagged in
        :attr:`AlignmentReport.failing_scales`. Default 0.4.
    """

    def __init__(
        self,
        *,
        scorers: Sequence[ScaleScorer],
        weights: Mapping[AlignmentScale, float] | None = None,
        allow_threshold: float = 0.4,
    ) -> None:
        if not scorers:
            raise ValueError("scorers must be non-empty")
        if not 0.0 <= allow_threshold <= 1.0:
            raise ValueError("allow_threshold must be in [0, 1]")
        seen: dict[AlignmentScale, ScaleScorer] = {}
        for scorer in scorers:
            if scorer.scale in seen:
                raise ValueError(f"duplicate scorer for scale {scorer.scale!r}")
            seen[scorer.scale] = scorer
        self._scorers: dict[AlignmentScale, ScaleScorer] = seen
        self._weights = self._normalise_weights(weights)
        self._allow_threshold = allow_threshold

    def _normalise_weights(
        self, weights: Mapping[AlignmentScale, float] | None
    ) -> dict[AlignmentScale, float]:
        if weights is None:
            uniform = 1.0 / len(self._scorers)
            return {scale: uniform for scale in self._scorers}
        for scale, weight in weights.items():
            if scale not in self._scorers:
                raise ValueError(f"weight for unknown scale {scale!r}")
            if weight < 0:
                raise ValueError(f"weight for scale {scale!r} must be non-negative")
        total = sum(weights.get(scale, 0.0) for scale in self._scorers)
        if total <= 0:
            raise ValueError("weights must sum to a positive number")
        return {scale: weights.get(scale, 0.0) / total for scale in self._scorers}

    def evaluate(self, action: Action) -> AlignmentReport:
        scores: dict[AlignmentScale, float] = {}
        for scale, scorer in self._scorers.items():
            raw = float(scorer.score(action))
            scores[scale] = max(0.0, min(1.0, raw))
        composite = sum(scores[scale] * self._weights[scale] for scale in self._scorers)
        failing = tuple(
            scale
            for scale in _ORDER
            if scale in scores and scores[scale] < self._allow_threshold
        )
        return AlignmentReport(
            action=action,
            table=ScaleScoreTable(scores=scores),
            composite=max(0.0, min(1.0, composite)),
            failing_scales=failing,
        )

    @property
    def scales(self) -> tuple[AlignmentScale, ...]:
        return tuple(scale for scale in _ORDER if scale in self._scorers)
