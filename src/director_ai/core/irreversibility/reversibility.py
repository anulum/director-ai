# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — reversibility estimator

"""Per-action reversibility scoring.

The Protocol is the stable boundary; everything else is one
implementation. :class:`RuleReversibility` matches configurable
critical-action phrases against an action description and fails
fast on ambiguous marker configuration. Deployments that require
learned or causal-graph-backed semantics can drop in another
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
    """Deterministic phrase-based reversibility estimator.

    Parameters
    ----------
    irreversible_markers :
        Phrases that drive the score toward 0. Defaults to a short
        production baseline list.
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
        self._irreversible = _normalise_markers(
            "irreversible_markers", irreversible_markers
        )
        self._reversible = _normalise_markers("reversible_markers", reversible_markers)
        overlap = set(self._irreversible).intersection(self._reversible)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(
                f"irreversible_markers and reversible_markers overlap: {joined}"
            )
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


def _normalise_markers(name: str, markers: Iterable[str]) -> tuple[str, ...]:
    normalised = tuple(marker.strip().lower() for marker in markers)
    if not normalised:
        raise ValueError(f"{name} must contain at least one marker")
    if any(not marker for marker in normalised):
        raise ValueError(f"{name} must not contain blank markers")
    return normalised
