# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — reversibility estimator

"""Per-action reversibility scoring.

The Protocol is the stable boundary; everything else is one
implementation. :class:`RuleReversibility` matches a handful of
common keywords against an action description. It is deliberately
narrow so callers treat it as a bootstrap floor — real deployments
are expected to drop in a classifier or a causal-graph-backed
estimator on the same Protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ReversibilityScore:
    """Scored action. ``score`` is the probability the action is
    reversible; ``1 - score`` is the irreversibility that the
    forecaster accumulates."""

    score: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"ReversibilityScore.score must be in [0, 1]; got {self.score!r}"
            )
        if not self.reason:
            raise ValueError("ReversibilityScore.reason must be non-empty")


@runtime_checkable
class ReversibilityEstimator(Protocol):
    """Score one action. Callers pass a free-form string plus an
    optional ``context`` mapping (tenant, prior state, etc.) that
    richer estimators may consume. The Protocol requires only
    ``action``; ``context`` is optional."""

    def score(
        self,
        action: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> ReversibilityScore: ...


_DEFAULT_IRREVERSIBLE_MARKERS: tuple[str, ...] = (
    "delete",
    "drop table",
    "truncate",
    "rm -rf",
    "format",
    "force push",
    "publish",
    "send email",
    "transfer funds",
    "merge to main",
)
_DEFAULT_REVERSIBLE_MARKERS: tuple[str, ...] = (
    "preview",
    "dry-run",
    "stage",
    "validate",
    "simulate",
    "lint",
    "read-only",
)


class RuleReversibility:
    """Keyword-based stub estimator.

    Parameters
    ----------
    irreversible_markers :
        Phrases that drive the score toward 0. Defaults to a short
        bootstrap list.
    reversible_markers :
        Phrases that drive the score toward 1.
    baseline :
        Score returned when no markers match. Default 0.5 —
        "unknown, escalate to a better estimator or a human".
    """

    def __init__(
        self,
        *,
        irreversible_markers: Iterable[str] = _DEFAULT_IRREVERSIBLE_MARKERS,
        reversible_markers: Iterable[str] = _DEFAULT_REVERSIBLE_MARKERS,
        baseline: float = 0.5,
    ) -> None:
        if not 0.0 <= baseline <= 1.0:
            raise ValueError(f"baseline must be in [0, 1]; got {baseline!r}")
        self._irreversible = tuple(m.lower() for m in irreversible_markers)
        self._reversible = tuple(m.lower() for m in reversible_markers)
        self._baseline = baseline

    def score(
        self,
        action: str,
        *,
        context: Mapping[str, object] | None = None,
    ) -> ReversibilityScore:
        _ = context  # reserved for richer estimators
        if not action or not action.strip():
            return ReversibilityScore(score=self._baseline, reason="empty action")
        lowered = action.lower()
        hits_irreversible = [m for m in self._irreversible if m in lowered]
        hits_reversible = [m for m in self._reversible if m in lowered]
        if hits_irreversible and not hits_reversible:
            return ReversibilityScore(
                score=0.05,
                reason=f"matched irreversible marker: {hits_irreversible[0]}",
            )
        if hits_reversible and not hits_irreversible:
            return ReversibilityScore(
                score=0.95,
                reason=f"matched reversible marker: {hits_reversible[0]}",
            )
        if hits_reversible and hits_irreversible:
            return ReversibilityScore(
                score=self._baseline,
                reason="both reversible and irreversible markers matched",
            )
        return ReversibilityScore(score=self._baseline, reason="no markers matched")
