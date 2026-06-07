# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — LTL Runtime Monitor
"""Three-valued LTL runtime monitor over a growing trace of states.

A :class:`LTLMonitor` drives a single LTL formula forward one observed state at a
time using formula progression. The verdict is three-valued (RV-LTL style): a
:data:`Verdict.VIOLATED` or :data:`Verdict.SATISFIED` is *definitive* and latches
— no later state can change it — while :data:`Verdict.INCONCLUSIVE` means the
property still depends on future states. :meth:`LTLMonitor.finalize` resolves any
residual obligation under finite-trace end semantics.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from .formula import (
    BOTTOM,
    TOP,
    Formula,
    progress,
    value_at_end,
)


class Verdict(StrEnum):
    """Three-valued runtime verdict for a monitored LTL property."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"


def _classify(formula: Formula) -> Verdict:
    if formula is TOP:
        return Verdict.SATISFIED
    if formula is BOTTOM:
        return Verdict.VIOLATED
    return Verdict.INCONCLUSIVE


class LTLMonitor:
    """Progress one LTL formula over a sequence of states into a verdict.

    Parameters
    ----------
    formula:
        The LTL property to monitor.
    name:
        Optional human-readable label used in reports.
    """

    def __init__(self, formula: Formula, name: str = "") -> None:
        self._initial = formula
        self._residual = formula
        self._name = name
        self._steps = 0
        self._verdict = _classify(formula)

    @property
    def name(self) -> str:
        """The monitor's label."""
        return self._name

    @property
    def initial(self) -> Formula:
        """The original (un-progressed) formula being monitored."""
        return self._initial

    @property
    def verdict(self) -> Verdict:
        """The current three-valued verdict."""
        return self._verdict

    @property
    def residual(self) -> Formula:
        """The obligation that remains for future states."""
        return self._residual

    @property
    def steps(self) -> int:
        """Number of states observed so far."""
        return self._steps

    @property
    def is_definitive(self) -> bool:
        """True once the verdict can no longer change (SATISFIED/VIOLATED)."""
        return self._verdict is not Verdict.INCONCLUSIVE

    def push(self, state: Iterable[str]) -> Verdict:
        """Observe one state (a set of true atomic propositions); return verdict.

        A definitive verdict latches: once violated or satisfied the residual is
        not progressed further, so later states cannot flip the result.
        """
        self._steps += 1
        if self.is_definitive:
            return self._verdict
        self._residual = progress(self._residual, frozenset(state))
        self._verdict = _classify(self._residual)
        return self._verdict

    def finalize(self) -> Verdict:
        """Resolve the residual obligation as if the trace has ended.

        A still-inconclusive property is decided under finite-trace end
        semantics (see :func:`director_ai.core.temporal_logic.formula.value_at_end`):
        a pending eventuality is a violation; a never-violated ``Always`` is
        satisfied. A definitive verdict is returned unchanged.
        """
        if self.is_definitive:
            return self._verdict
        self._verdict = (
            Verdict.SATISFIED if value_at_end(self._residual) else Verdict.VIOLATED
        )
        return self._verdict
