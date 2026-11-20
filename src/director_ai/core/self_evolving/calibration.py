# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — conformal calibrator

"""Split-conformal threshold for a fresh :class:`TrainedGuardrail`.

The caller supplies a held-out calibration set of labelled
events; the calibrator scores each with the guardrail, collects
the non-conformity scores (``|score - label|``), and returns the
``(1 - α)(n + 1) / n`` order statistic. The returned threshold
is what the hot-swap registry installs — requests whose score
exceeds it are flagged as unsafe.
"""

from __future__ import annotations

import math
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

    @property
    def target_coverage(self) -> float:
        """Return the configured coverage target ``1 - α``."""
        return self._target

    @property
    def minimum_calibration_size(self) -> int:
        """Return the smallest labelled fold that supports the target.

        Split-conformal coverage needs the ``ceil((n + 1)(1 - α))``-th
        order statistic to exist among ``n`` scores. With fewer scores
        :meth:`calibrate` can only fall back to the largest score, which
        carries no coverage guarantee. Callers refuse such folds before
        they train. The search starts two below the closed-form bound
        ``t / (1 - t)``, so it stays short when the target lies very
        close to one, and then steps up with the same expression as
        :meth:`calibrate`. The returned size therefore always yields an
        unclamped order statistic in :meth:`calibrate`.
        """
        target = self._target
        n = max(1, math.floor(target / (1.0 - target)) - 2)
        while math.ceil(target * (n + 1)) > n:
            n += 1
        return n

    def calibrate(
        self,
        guardrail: TrainedGuardrail,
        calibration_set: Sequence[FeedbackEvent],
    ) -> ConformalResult:
        """Fit the conformal threshold so ``guardrail`` meets the target coverage."""
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
        # Split-conformal threshold: the ceil((n+1)*target)-th smallest
        # non-conformity score (1-indexed). Truncating with int() selects a
        # lower order statistic and under-covers; math.ceil gives the
        # conservative quantile the coverage guarantee requires.
        q_index = min(max(math.ceil(self._target * (n + 1)) - 1, 0), n - 1)
        threshold = float(sorted_scores[q_index])
        return ConformalResult(
            threshold=threshold,
            coverage_target=self._target,
            calibration_size=n,
        )
