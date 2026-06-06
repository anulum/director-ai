# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Active labelling cockpit

"""Triage what to label, measure error, recommend a threshold, export a packet.

:class:`ActiveLabellingCockpit` turns a stream of scored guard decisions into a
review workflow:

* :meth:`rank_for_labelling` surfaces the most informative unlabelled items —
  the ones whose score sits closest to the decision boundary, where a label
  resolves the most uncertainty.
* :meth:`error_breakdown` splits the labelled items into false halts (grounded
  answers the guard blocked) and missed hallucinations (hallucinations the guard
  approved).
* :meth:`tradeoff_curve` sweeps the threshold and reports both error rates at
  each boundary.
* :meth:`recommend_threshold` picks the threshold (optionally per domain) that
  minimises the weighted error.
* :meth:`export_packet` emits a deterministic train/eval packet for retraining.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .items import GROUNDED, HALLUCINATION, LabelItem

__all__ = [
    "ActiveLabellingCockpit",
    "ErrorBreakdown",
    "ThresholdPoint",
    "ThresholdRecommendation",
]


@dataclass(frozen=True)
class ErrorBreakdown:
    """Counts of guard decisions against reviewer ground truth."""

    false_halts: int
    missed_hallucinations: int
    correct: int

    @property
    def labelled_total(self) -> int:
        """Total labelled items considered."""
        return self.false_halts + self.missed_hallucinations + self.correct

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "false_halts": self.false_halts,
            "missed_hallucinations": self.missed_hallucinations,
            "correct": self.correct,
            "labelled_total": self.labelled_total,
        }


@dataclass(frozen=True)
class ThresholdPoint:
    """Error rates at one candidate threshold (approve when score >= t)."""

    threshold: float
    false_halts: int
    missed_hallucinations: int

    @property
    def total_errors(self) -> int:
        """Sum of both error counts at this threshold."""
        return self.false_halts + self.missed_hallucinations

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "threshold": self.threshold,
            "false_halts": self.false_halts,
            "missed_hallucinations": self.missed_hallucinations,
            "total_errors": self.total_errors,
        }


@dataclass(frozen=True)
class ThresholdRecommendation:
    """A recommended threshold and the trade-off curve it was chosen from."""

    threshold: float
    domain: str
    point: ThresholdPoint
    curve: tuple[ThresholdPoint, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "threshold": self.threshold,
            "domain": self.domain,
            "point": self.point.to_dict(),
            "curve": [p.to_dict() for p in self.curve],
        }


class ActiveLabellingCockpit:
    """Active-learning review workflow over scored guard decisions.

    Parameters
    ----------
    threshold:
        The current operating threshold; used to rank items by proximity to the
        decision boundary.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.threshold = threshold

    def rank_for_labelling(
        self,
        items: Sequence[LabelItem],
        *,
        top_n: int = 50,
    ) -> list[LabelItem]:
        """Return the most informative unlabelled items to label next.

        Ranks unlabelled items by how close their score is to the operating
        threshold (most uncertain first); ties break by item id for stability.
        """
        if top_n < 0:
            raise ValueError("top_n must be non-negative")
        unlabelled = [item for item in items if not item.labelled]
        unlabelled.sort(key=lambda i: (abs(i.score - self.threshold), i.item_id))
        return unlabelled[:top_n]

    def error_breakdown(self, items: Sequence[LabelItem]) -> ErrorBreakdown:
        """Split labelled items into false halts, misses, and correct calls.

        Uses each item's actual ``guard_approved`` decision against its label.
        """
        false_halts = missed = correct = 0
        for item in items:
            if not item.labelled:
                continue
            if item.label == GROUNDED and not item.guard_approved:
                false_halts += 1
            elif item.label == HALLUCINATION and item.guard_approved:
                missed += 1
            else:
                correct += 1
        return ErrorBreakdown(false_halts, missed, correct)

    def tradeoff_curve(
        self,
        items: Sequence[LabelItem],
        *,
        domain: str | None = None,
    ) -> list[ThresholdPoint]:
        """Sweep candidate thresholds and report both error counts at each.

        Candidates are the labelled items' scores plus the ``0.0`` and ``1.0``
        endpoints, so the curve is exact at every decision boundary. An item is
        approved at threshold ``t`` when ``score >= t``.
        """
        labelled = self._labelled(items, domain)
        candidates = sorted({0.0, 1.0, *(i.score for i in labelled)})
        curve: list[ThresholdPoint] = []
        for t in candidates:
            false_halts = sum(
                1 for i in labelled if i.label == GROUNDED and i.score < t
            )
            missed = sum(
                1 for i in labelled if i.label == HALLUCINATION and i.score >= t
            )
            curve.append(ThresholdPoint(round(t, 4), false_halts, missed))
        return curve

    def recommend_threshold(
        self,
        items: Sequence[LabelItem],
        *,
        domain: str | None = None,
        false_halt_weight: float = 1.0,
        miss_weight: float = 1.0,
    ) -> ThresholdRecommendation:
        """Recommend the threshold minimising weighted error.

        Cost at a threshold is
        ``false_halt_weight * false_halts + miss_weight * missed``. Ties break
        toward the lower threshold for a more permissive operating point.
        """
        if not self._labelled(items, domain):
            raise ValueError("no labelled items to recommend a threshold from")
        curve = self.tradeoff_curve(items, domain=domain)

        def _cost(point: ThresholdPoint) -> float:
            return (
                false_halt_weight * point.false_halts
                + miss_weight * point.missed_hallucinations
            )

        best = min(curve, key=lambda p: (_cost(p), p.threshold))
        return ThresholdRecommendation(
            threshold=best.threshold,
            domain=domain or "",
            point=best,
            curve=tuple(curve),
        )

    def export_packet(
        self,
        items: Sequence[LabelItem],
        *,
        eval_fraction: float = 0.2,
    ) -> dict[str, Any]:
        """Emit a deterministic train/eval packet from the labelled items.

        Items are ordered by id and split deterministically (every ``k``-th item
        to eval, by ``eval_fraction``), so a re-export with the same labels gives
        the same split — no randomness.
        """
        if not 0.0 <= eval_fraction < 1.0:
            raise ValueError("eval_fraction must be in [0, 1)")
        labelled = sorted(self._labelled(items, None), key=lambda i: i.item_id)
        train: list[LabelItem] = []
        evaluation: list[LabelItem] = []
        stride = round(1 / eval_fraction) if eval_fraction > 0 else 0
        for index, item in enumerate(labelled):
            if stride and index % stride == 0:
                evaluation.append(item)
            else:
                train.append(item)
        return {
            "train": [i.to_packet_row() for i in train],
            "eval": [i.to_packet_row() for i in evaluation],
            "counts": {
                "labelled": len(labelled),
                "train": len(train),
                "eval": len(evaluation),
            },
        }

    @staticmethod
    def _labelled(
        items: Sequence[LabelItem],
        domain: str | None,
    ) -> list[LabelItem]:
        return [
            item
            for item in items
            if item.labelled and (domain is None or item.domain == domain)
        ]
