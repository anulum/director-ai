# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Online calibrator — adjust guardrail thresholds from deployment feedback.

Uses accumulated human corrections to:
1. Find the optimal threshold for the deployment's actual data
2. Compute deployment-specific error rates with confidence intervals
3. Track accuracy trends over time

Usage::

    calibrator = OnlineCalibrator(store)
    report = calibrator.calibrate()
    print(f"Optimal threshold: {report.optimal_threshold}")
    print(f"FPR: {report.fpr:.3f} ± {report.fpr_ci:.3f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .feedback_store import FeedbackStore

__all__ = ["CalibrationReport", "OnlineCalibrator"]


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson score confidence interval half-width."""
    if total == 0:
        return 1.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return min(spread, center, 1 - center)


@dataclass
class CalibrationReport:
    """Result of online threshold calibration."""

    correction_count: int
    optimal_threshold: float | None  # None if insufficient data
    current_accuracy: float
    tpr: float  # true positive rate (correctly approved)
    tnr: float  # true negative rate (correctly rejected)
    fpr: float  # false positive rate (approved but shouldn't be)
    fnr: float  # false negative rate (rejected but shouldn't be)
    fpr_ci: float  # 95% CI half-width on FPR
    fnr_ci: float  # 95% CI half-width on FNR


class OnlineCalibrator:
    """Calibrate guardrail threshold from accumulated feedback.

    Parameters
    ----------
    store : FeedbackStore
        The feedback store to read corrections from.
    min_corrections : int
        Minimum corrections before calibration is attempted.
    """

    def __init__(self, store: FeedbackStore, min_corrections: int = 20):
        self._store = store
        self._min_corrections = min_corrections

    def calibrate(self, domain: str | None = None) -> CalibrationReport:
        """Compute calibration metrics from accumulated feedback.

        Returns a CalibrationReport with error rates and optimal threshold.
        """
        corrections = self._store.get_corrections(domain=domain)
        n = len(corrections)

        if n == 0:
            return CalibrationReport(
                correction_count=0,
                optimal_threshold=None,
                current_accuracy=0.0,
                tpr=0.0,
                tnr=0.0,
                fpr=0.0,
                fnr=0.0,
                fpr_ci=1.0,
                fnr_ci=1.0,
            )

        # Confusion matrix from guardrail vs human verdicts
        tp = sum(1 for c in corrections if c.guardrail_approved and c.human_approved)
        tn = sum(
            1 for c in corrections if not c.guardrail_approved and not c.human_approved
        )
        fp = sum(
            1 for c in corrections if c.guardrail_approved and not c.human_approved
        )
        fn = sum(
            1 for c in corrections if not c.guardrail_approved and c.human_approved
        )

        accuracy = (tp + tn) / n if n > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        fpr_ci = _wilson_ci(fp, fp + tn)
        fnr_ci = _wilson_ci(fn, fn + tp)

        # Optimal threshold via score sweep (only if we have scores)
        optimal_threshold = None
        if n >= self._min_corrections:
            scored = [
                (c.guardrail_score, c.human_approved)
                for c in corrections
                if c.guardrail_score > 0
            ]
            if len(scored) >= self._min_corrections:
                optimal_threshold = self._sweep_threshold(scored)

        return CalibrationReport(
            correction_count=n,
            optimal_threshold=optimal_threshold,
            current_accuracy=accuracy,
            tpr=tpr,
            tnr=tnr,
            fpr=fpr,
            fnr=fnr,
            fpr_ci=fpr_ci,
            fnr_ci=fnr_ci,
        )

    @staticmethod
    def _sweep_threshold(
        scored: list[tuple[float, bool]],
    ) -> float | None:
        """Find threshold maximizing balanced accuracy on labeled data.

        Returns None if the data has only one class (no calibration signal).
        Sweeps thresholds from 0.05 to 0.95 in 1% steps.
        """
        pos = sum(1 for _, h in scored if h)
        neg = len(scored) - pos
        if pos == 0 or neg == 0:
            return None

        best_t = 0.5
        best_ba = 0.0
        for t_int in range(5, 96):
            t = t_int / 100.0
            tp = sum(1 for s, h in scored if s >= t and h)
            tn = sum(1 for s, h in scored if s < t and not h)
            tpr = tp / pos
            tnr = tn / neg
            ba = (tpr + tnr) / 2
            if ba > best_ba:
                best_ba = ba
                best_t = t
        return best_t
