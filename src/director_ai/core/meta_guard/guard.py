# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MetaGuard orchestrator

"""Bind a :class:`DecisionLog`, :class:`MetaAnalyzer`, and
:class:`ThresholdAdjuster` into a single ``.record(...)`` entry
point.

The orchestrator lets the caller fold new scoring decisions in as
they happen. Every ``record`` call returns a :class:`MetaVerdict`
that reports the observed drift and any threshold change the
adjuster applied. Callers push the new thresholds into the
scoring layer at their own cadence — the guard does not reach
into the scorer directly so it stays coupling-free.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adjuster import ThresholdAdjuster, ThresholdBundle
from .analyzer import MetaAnalysis, MetaAnalyzer
from .log import DecisionLog, ScoringAction, ScoringDecision


@dataclass(frozen=True)
class MetaVerdict:
    """One ``record`` outcome.

    ``analysis`` carries the drift statistics; ``thresholds`` is
    the updated :class:`ThresholdBundle` when the adjuster moved
    (``None`` otherwise); ``decision`` echoes the stored record
    so callers can log it without re-hashing.
    """

    decision: ScoringDecision
    analysis: MetaAnalysis
    thresholds: ThresholdBundle | None

    @property
    def adjusted(self) -> bool:
        return self.thresholds is not None


class MetaGuard:
    """Record decisions, analyse drift, auto-adjust thresholds.

    Parameters
    ----------
    log :
        Decision store.
    analyzer :
        Drift detector.
    adjuster :
        Threshold mover — ``None`` disables auto-adjustment
        and the guard runs in observe-only mode.
    window_last_n :
        How many recent decisions the analyser sees per call.
        Default 256 — large enough for meaningful statistics,
        small enough to respond to drift quickly.
    """

    def __init__(
        self,
        *,
        log: DecisionLog,
        analyzer: MetaAnalyzer,
        adjuster: ThresholdAdjuster | None = None,
        window_last_n: int = 256,
    ) -> None:
        if window_last_n <= 0:
            raise ValueError("window_last_n must be positive")
        self._log = log
        self._analyzer = analyzer
        self._adjuster = adjuster
        self._window = window_last_n

    @property
    def adjuster(self) -> ThresholdAdjuster | None:
        return self._adjuster

    @property
    def log(self) -> DecisionLog:
        return self._log

    def record(
        self,
        *,
        prompt: str,
        score: float,
        action: ScoringAction,
        ground_truth: float | None = None,
        tenant_id: str = "",
    ) -> MetaVerdict:
        """Fold a decision in and return the resulting verdict."""
        decision = self._log.record(
            prompt=prompt,
            score=score,
            action=action,
            ground_truth=ground_truth,
            tenant_id=tenant_id,
        )
        window = self._log.window(last_n=self._window)
        analysis = self._analyzer.analyse(window)
        adjusted = (
            self._adjuster.observe(analysis) if self._adjuster is not None else None
        )
        return MetaVerdict(
            decision=decision,
            analysis=analysis,
            thresholds=adjusted,
        )

    def latest_analysis(self) -> MetaAnalysis:
        """Run the analyser over the current window without
        recording anything."""
        window = self._log.window(last_n=self._window)
        return self._analyzer.analyse(window)
