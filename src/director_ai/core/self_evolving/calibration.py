# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — conformal calibrator

"""Split-conformal threshold for a fresh :class:`TrainedGuardrail`.

The caller supplies a held-out calibration set of labelled
events; the calibrator scores each with the guardrail, collects
the non-conformity scores (``|score - label|``), and returns the
``(1 - α)(n + 1) / n`` order statistic. The returned threshold
is what the hot-swap registry installs — requests whose score
exceeds it are flagged as unsafe.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .feedback import FeedbackEvent
from .trainer import TrainedGuardrail, _event_target


@dataclass(frozen=True)
class ConformalResult:
    """Output of one calibration run."""

    threshold: float
    coverage_target: float
    calibration_size: int


class ConformalCalibrator:
    """Split-conformal prediction over the feedback set.

    Parameters
    ----------
    target_coverage :
        The miscoverage-bound target ``1 - α``. Default 0.90.
    """

    def __init__(self, *, target_coverage: float = 0.90) -> None:
        if not 0.0 < target_coverage < 1.0:
            raise ValueError(
                f"target_coverage must be in (0, 1); got {target_coverage!r}"
            )
        self._target = target_coverage

    def calibrate(
        self,
        guardrail: TrainedGuardrail,
        calibration_set: Sequence[FeedbackEvent],
    ) -> ConformalResult:
        labelled: list[tuple[FeedbackEvent, int]] = []
        for event in calibration_set:
            mapped = _event_target(event)
            if mapped is not None:
                labelled.append((event, mapped))
        if not labelled:
            raise ValueError("calibration_set must contain labelled events")
        non_conformity = []
        for event, target_int in labelled:
            score = guardrail.score(event.prompt)
            non_conformity.append(abs(score - float(target_int)))
        sorted_scores = sorted(non_conformity)
        n = len(sorted_scores)
        q_index = min(max(int((self._target * (n + 1)) - 1), 0), n - 1)
        threshold = float(sorted_scores[q_index])
        return ConformalResult(
            threshold=threshold,
            coverage_target=self._target,
            calibration_size=n,
        )
