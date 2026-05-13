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

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

from .adjuster import ThresholdAdjuster, ThresholdBundle
from .analyzer import MetaAnalysis, MetaAnalyzer
from .log import DecisionLog, ScoringAction, ScoringDecision


@dataclass(frozen=True)
class ProductionMetaGuardDecision:
    """Production gate result for recursive guard adjustment."""

    enabled: bool
    blocked: bool = False
    block_reason: str = ""
    window_size: int = 0
    labelled_fraction: float = 0.0
    single_tenant_fraction: float = 0.0
    duplicate_prompt_fraction: float = 0.0
    evasion_score: float = 0.0


class _WindowMetrics(TypedDict):
    window_size: int
    labelled_fraction: float
    single_tenant_fraction: float
    duplicate_prompt_fraction: float
    evasion_score: float


@dataclass(frozen=True)
class MetaGuardProductionPolicy:
    """Gate recursive threshold changes in production deployments.

    The policy does not replace drift detection. It decides whether a detected
    drift window is safe enough for the recursive guard layer to adjust
    thresholds. Dominated windows are treated as possible evasion or traffic
    skew and are observe-only until operator review.
    """

    min_labelled_fraction: float = 0.2
    max_single_tenant_fraction: float = 0.5
    max_duplicate_prompt_fraction: float = 0.4

    def __post_init__(self) -> None:
        _validate_fraction("min_labelled_fraction", self.min_labelled_fraction)
        _validate_fraction(
            "max_single_tenant_fraction", self.max_single_tenant_fraction
        )
        _validate_fraction(
            "max_duplicate_prompt_fraction", self.max_duplicate_prompt_fraction
        )

    def evaluate(
        self,
        *,
        window: Sequence[ScoringDecision],
        analysis: MetaAnalysis,
    ) -> ProductionMetaGuardDecision:
        """Return whether the current recursive adjustment may proceed."""
        metrics = _window_metrics(window)
        if not analysis.any_alarm:
            return _production_decision(enabled=True, metrics=metrics)
        if metrics["window_size"] == 0:
            return _production_decision(enabled=True, metrics=metrics)
        if metrics["single_tenant_fraction"] > self.max_single_tenant_fraction:
            return _production_decision(
                enabled=True,
                blocked=True,
                block_reason="single_tenant_dominance",
                metrics=metrics,
            )
        if metrics["duplicate_prompt_fraction"] > self.max_duplicate_prompt_fraction:
            return _production_decision(
                enabled=True,
                blocked=True,
                block_reason="duplicate_prompt_dominance",
                metrics=metrics,
            )
        if metrics["labelled_fraction"] < self.min_labelled_fraction:
            return _production_decision(
                enabled=True,
                blocked=True,
                block_reason="insufficient_labels",
                metrics=metrics,
            )
        return _production_decision(enabled=True, metrics=metrics)


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
    production: ProductionMetaGuardDecision = ProductionMetaGuardDecision(enabled=False)

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
    production_policy :
        Optional production gate. When supplied, detected drift windows must
        pass diversity/evasion checks before the adjuster is allowed to mutate
        thresholds.
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
        production_policy: MetaGuardProductionPolicy | None = None,
        window_last_n: int = 256,
    ) -> None:
        if window_last_n <= 0:
            raise ValueError("window_last_n must be positive")
        self._log = log
        self._analyzer = analyzer
        self._adjuster = adjuster
        self._production_policy = production_policy
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
        production = (
            self._production_policy.evaluate(window=window, analysis=analysis)
            if self._production_policy is not None
            else ProductionMetaGuardDecision(enabled=False)
        )
        adjusted = (
            self._adjuster.observe(analysis)
            if self._adjuster is not None and not production.blocked
            else None
        )
        return MetaVerdict(
            decision=decision,
            analysis=analysis,
            thresholds=adjusted,
            production=production,
        )

    def latest_analysis(self) -> MetaAnalysis:
        """Run the analyser over the current window without
        recording anything."""
        window = self._log.window(last_n=self._window)
        return self._analyzer.analyse(window)


def _production_decision(
    *,
    enabled: bool,
    metrics: _WindowMetrics,
    blocked: bool = False,
    block_reason: str = "",
) -> ProductionMetaGuardDecision:
    return ProductionMetaGuardDecision(
        enabled=enabled,
        blocked=blocked,
        block_reason=block_reason,
        window_size=metrics["window_size"],
        labelled_fraction=metrics["labelled_fraction"],
        single_tenant_fraction=metrics["single_tenant_fraction"],
        duplicate_prompt_fraction=metrics["duplicate_prompt_fraction"],
        evasion_score=metrics["evasion_score"],
    )


def _window_metrics(window: Sequence[ScoringDecision]) -> _WindowMetrics:
    size = len(window)
    if size == 0:
        return {
            "window_size": 0,
            "labelled_fraction": 0.0,
            "single_tenant_fraction": 0.0,
            "duplicate_prompt_fraction": 0.0,
            "evasion_score": 0.0,
        }
    labelled = sum(1 for decision in window if decision.ground_truth is not None)
    tenant_counts = Counter(decision.tenant_id for decision in window)
    prompt_counts = Counter(decision.prompt_hash for decision in window)
    single_tenant_fraction = max(tenant_counts.values()) / size
    duplicate_prompt_fraction = max(prompt_counts.values()) / size
    evasion_score = max(single_tenant_fraction, duplicate_prompt_fraction)
    return {
        "window_size": size,
        "labelled_fraction": labelled / size,
        "single_tenant_fraction": single_tenant_fraction,
        "duplicate_prompt_fraction": duplicate_prompt_fraction,
        "evasion_score": evasion_score,
    }


def _validate_fraction(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
