# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MetaAnalyzer

"""Drift statistics over a :class:`DecisionLog` window.

Three signals:

* **Page-Hinkley** — change-point detection on the score mean.
  Computes the cumulative sum of ``(x - mean_reference - delta)``
  and reports a shift when the running minimum of the sum and the
  current value differ by more than ``ph_threshold``. Robust on
  bounded scores (``[0, 1]``) and cheap to update incrementally.
* **Brier score drift** — calibration error. When ground-truth
  labels are present in the window, the analyser computes the
  Brier score and compares it to a caller-supplied reference. A
  rising Brier indicates the scorer is becoming worse-calibrated
  even if its mean hasn't moved.
* **Action-rate drift** — per-action rate (allow / warn / halt)
  against a caller-supplied reference distribution. Reported as
  a chi-square-style divergence so a single rate spike is
  flagged even when overall means are steady.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from .log import ScoringAction, ScoringDecision

_ACTIONS: tuple[ScoringAction, ...] = ("allow", "warn", "halt")


@dataclass(frozen=True)
class MetaAnalysis:
    """Result of one :meth:`MetaAnalyzer.analyse` call."""

    window_size: int
    mean_score: float
    page_hinkley_statistic: float
    page_hinkley_alarm: bool
    brier_score: float | None
    brier_delta: float | None
    brier_alarm: bool
    action_rates: dict[ScoringAction, float] = field(default_factory=dict)
    action_divergence: float = 0.0
    action_alarm: bool = False

    @property
    def any_alarm(self) -> bool:
        return self.page_hinkley_alarm or self.brier_alarm or self.action_alarm


class MetaAnalyzer:
    """Windowed drift detector.

    Parameters
    ----------
    reference_mean :
        Expected score mean from the calibration distribution.
    ph_delta :
        Page-Hinkley slack — positive number that absorbs
        small-magnitude noise. Default 0.005 — tuned for
        guardrail scores in ``[0, 1]``.
    ph_threshold :
        Cumulative-sum cutoff that triggers the Page-Hinkley
        alarm. Default 0.05 — adjust per deployment noise level.
    reference_brier :
        Expected Brier score. ``None`` disables the Brier
        channel. Default 0.10.
    brier_tolerance :
        Absolute increase over ``reference_brier`` that triggers
        the Brier alarm. Default 0.05.
    reference_action_rates :
        Expected per-action rate distribution. Must sum to 1.0
        when supplied. ``None`` disables the channel.
    action_tolerance :
        Chi-square divergence above which the action-rate alarm
        fires. Default 0.1.
    min_window :
        Minimum window size before any alarm fires. Default 32.
    """

    def __init__(
        self,
        *,
        reference_mean: float,
        ph_delta: float = 0.005,
        ph_threshold: float = 0.05,
        reference_brier: float | None = 0.10,
        brier_tolerance: float = 0.05,
        reference_action_rates: dict[ScoringAction, float] | None = None,
        action_tolerance: float = 0.1,
        min_window: int = 32,
    ) -> None:
        if not 0.0 <= reference_mean <= 1.0:
            raise ValueError(
                f"reference_mean must be in [0, 1]; got {reference_mean!r}"
            )
        if ph_delta < 0:
            raise ValueError("ph_delta must be non-negative")
        if ph_threshold <= 0:
            raise ValueError("ph_threshold must be positive")
        if reference_brier is not None and not 0.0 <= reference_brier <= 1.0:
            raise ValueError(
                f"reference_brier must be in [0, 1]; got {reference_brier!r}"
            )
        if brier_tolerance < 0:
            raise ValueError("brier_tolerance must be non-negative")
        if reference_action_rates is not None:
            total = sum(reference_action_rates.values())
            if not 0.999 <= total <= 1.001:
                raise ValueError(f"reference_action_rates must sum to 1.0; got {total}")
            for action, rate in reference_action_rates.items():
                if action not in _ACTIONS:
                    raise ValueError(
                        f"unknown action {action!r} in reference_action_rates"
                    )
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(f"action rate must be in [0, 1]; got {rate!r}")
        if action_tolerance < 0:
            raise ValueError("action_tolerance must be non-negative")
        if min_window <= 0:
            raise ValueError("min_window must be positive")
        self._reference_mean = reference_mean
        self._ph_delta = ph_delta
        self._ph_threshold = ph_threshold
        self._reference_brier = reference_brier
        self._brier_tolerance = brier_tolerance
        self._reference_action_rates = (
            dict(reference_action_rates) if reference_action_rates is not None else None
        )
        self._action_tolerance = action_tolerance
        self._min_window = min_window

    def analyse(self, window: Sequence[ScoringDecision]) -> MetaAnalysis:
        n = len(window)
        if n == 0:
            return MetaAnalysis(
                window_size=0,
                mean_score=0.0,
                page_hinkley_statistic=0.0,
                page_hinkley_alarm=False,
                brier_score=None,
                brier_delta=None,
                brier_alarm=False,
            )
        mean_score = sum(d.score for d in window) / n
        ph_stat = self._page_hinkley(window)
        ph_alarm = n >= self._min_window and ph_stat > self._ph_threshold
        brier_score, brier_delta, brier_alarm = self._brier(window, n)
        rates, divergence, action_alarm = self._action_rates(window, n)
        return MetaAnalysis(
            window_size=n,
            mean_score=mean_score,
            page_hinkley_statistic=ph_stat,
            page_hinkley_alarm=ph_alarm,
            brier_score=brier_score,
            brier_delta=brier_delta,
            brier_alarm=brier_alarm,
            action_rates=rates,
            action_divergence=divergence,
            action_alarm=action_alarm,
        )

    def _page_hinkley(self, window: Sequence[ScoringDecision]) -> float:
        """Two-sided Page-Hinkley statistic.

        Each tracker accumulates a drift-specific deviation. Under
        a steady signal the cumulative sum drifts toward a running
        minimum (the ``-ph_delta`` drift is absorbed by the
        ``min_*`` anchor); the statistic fires when the current
        sum rises above its running minimum by more than
        ``ph_threshold``.
        """
        cum_up = 0.0
        min_up = 0.0
        cum_down = 0.0
        min_down = 0.0
        for d in window:
            cum_up += d.score - self._reference_mean - self._ph_delta
            min_up = min(min_up, cum_up)
            cum_down += self._reference_mean - d.score - self._ph_delta
            min_down = min(min_down, cum_down)
        upward = cum_up - min_up
        downward = cum_down - min_down
        return max(upward, downward)

    def _brier(
        self, window: Sequence[ScoringDecision], n: int
    ) -> tuple[float | None, float | None, bool]:
        if self._reference_brier is None:
            return None, None, False
        labelled: list[tuple[float, float]] = []
        for d in window:
            if d.ground_truth is not None:
                labelled.append((d.score, d.ground_truth))
        if len(labelled) < self._min_window:
            return None, None, False
        brier = sum((s - g) ** 2 for s, g in labelled) / len(labelled)
        delta = brier - self._reference_brier
        alarm = n >= self._min_window and delta > self._brier_tolerance
        return brier, delta, alarm

    def _action_rates(
        self, window: Sequence[ScoringDecision], n: int
    ) -> tuple[dict[ScoringAction, float], float, bool]:
        rates: dict[ScoringAction, float] = {action: 0.0 for action in _ACTIONS}
        for d in window:
            rates[d.action] = rates.get(d.action, 0.0) + 1.0
        rates = {action: count / n for action, count in rates.items()}
        if self._reference_action_rates is None:
            return rates, 0.0, False
        divergence = 0.0
        for action, ref in self._reference_action_rates.items():
            observed = rates.get(action, 0.0)
            if ref > 0:
                divergence += ((observed - ref) ** 2) / ref
            elif observed > 0:
                divergence = math.inf
                break
        alarm = n >= self._min_window and divergence > self._action_tolerance
        return rates, divergence, alarm
