# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ScaleConflictDetector

"""Detect cross-scale disagreement with a conformal threshold.

The detector calibrates on a holdout set of per-scale score
vectors drawn from trusted actions, computes the ``(1 - α)``
quantile of pairwise score deltas, and reports a
:class:`ScaleConflict` whenever an observed action's deltas
exceed the quantile on any pair of scales.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .aligner import ScaleScoreTable
from .scorer import AlignmentScale

_ORDER: tuple[AlignmentScale, ...] = ("agent", "swarm", "org", "planetary")


@dataclass(frozen=True)
class ScaleConflict:
    """One cross-scale disagreement.

    ``scales`` is an ordered pair with the wider-scoped scale
    first (``org`` vs ``agent`` comes as ``("org", "agent")``).
    ``delta`` is the absolute difference between the scales'
    scores. ``threshold`` is the conformal quantile above which
    the delta is flagged.
    """

    scales: tuple[AlignmentScale, AlignmentScale]
    delta: float
    threshold: float

    @property
    def is_severe(self) -> bool:
        """Delta at least twice the threshold — shorthand callers
        use to escalate past the usual review queue."""
        return self.delta >= 2.0 * self.threshold


class ScaleConflictDetector:
    """Split-conformal detector for cross-scale alignment
    conflicts.

    Parameters
    ----------
    target_coverage :
        Conformal coverage target ``1 - α``. Default 0.90.
    """

    def __init__(self, *, target_coverage: float = 0.90) -> None:
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(
                f"target_coverage must be in (0, 1); got {target_coverage!r}"
            )
        self._target = target_coverage
        self._threshold: float | None = None

    @property
    def threshold(self) -> float | None:
        return self._threshold

    def calibrate(self, calibration_tables: Sequence[ScaleScoreTable]) -> float:
        """Set the per-pair threshold from the calibration set and
        return it.

        Deltas are collected across every scale pair in every
        table; the ``(1 - α)(n + 1) / n`` order statistic sets the
        threshold. Raises :class:`ValueError` when fewer than two
        tables carry at least two scales each.
        """
        deltas: list[float] = []
        for table in calibration_tables:
            ordered = table.ordered()
            for i, (_, a) in enumerate(ordered):
                for _, b in ordered[i + 1 :]:
                    deltas.append(abs(a - b))
        if len(deltas) < 2:
            raise ValueError("calibration requires at least two pairwise deltas")
        sorted_deltas = sorted(deltas)
        n = len(sorted_deltas)
        q_index = min(max(int((self._target * (n + 1)) - 1), 0), n - 1)
        threshold = float(sorted_deltas[q_index])
        self._threshold = threshold
        return threshold

    def detect(self, table: ScaleScoreTable) -> tuple[ScaleConflict, ...]:
        """Return every scale pair whose absolute delta exceeds
        the calibrated threshold.

        Raises :class:`ValueError` when :meth:`calibrate` has not
        run yet — the detector refuses to return an uncalibrated
        verdict since thresholds without calibration would be
        arbitrary."""
        if self._threshold is None:
            raise ValueError("calibrate() before detect()")
        conflicts: list[ScaleConflict] = []
        ordered = table.ordered()
        for i, (scale_a, score_a) in enumerate(ordered):
            for scale_b, score_b in ordered[i + 1 :]:
                delta = abs(score_a - score_b)
                if delta > self._threshold:
                    wider, narrower = _wider_first(scale_a, scale_b)
                    conflicts.append(
                        ScaleConflict(
                            scales=(wider, narrower),
                            delta=delta,
                            threshold=self._threshold,
                        )
                    )
        return tuple(conflicts)


def _wider_first(
    a: AlignmentScale, b: AlignmentScale
) -> tuple[AlignmentScale, AlignmentScale]:
    order = {scale: idx for idx, scale in enumerate(_ORDER)}
    if order[a] > order[b]:
        return a, b
    return b, a
