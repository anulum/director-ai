# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CounterfactualVerifier

"""Generate a handful of counterfactual branches around a decision
point and report which branches preserve a caller-supplied safety
invariant.

The verifier is the glue that turns the :class:`CausalGraph` +
:class:`Intervention` primitives into an actionable safety check.
It does not model uncertainty — every branch is deterministic.
Monte-Carlo-style uncertainty lives in :mod:`irreversibility` and
can be composed on top of this verifier later.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from .graph import CausalGraph
from .intervention import Intervention

SafetyInvariant = Callable[[Mapping[str, object]], bool]

BranchOutcome = Literal["safe", "unsafe"]


@dataclass(frozen=True)
class CounterfactualBranch:
    """One what-if branch and its verdict."""

    label: str
    intervention: Intervention
    values: Mapping[str, object]
    outcome: BranchOutcome


@dataclass(frozen=True)
class Verdict:
    """Aggregate verdict across every counterfactual branch."""

    total: int
    safe: int
    unsafe_branches: tuple[CounterfactualBranch, ...]

    @property
    def unsafe(self) -> int:
        return self.total - self.safe

    @property
    def safety_rate(self) -> float:
        return self.safe / self.total if self.total else 0.0


class CounterfactualVerifier:
    """Run a set of interventions and grade each against a safety
    invariant.

    Parameters
    ----------
    graph :
        The :class:`CausalGraph` to operate on.
    safety_invariant :
        Callable that receives the post-intervention variable mapping
        and returns ``True`` when the branch is safe.
    """

    def __init__(
        self,
        graph: CausalGraph,
        *,
        safety_invariant: SafetyInvariant,
    ) -> None:
        self._graph = graph
        self._invariant = safety_invariant

    def verify(
        self,
        *,
        inputs: Mapping[str, object],
        branches: Iterable[tuple[str, Intervention]],
    ) -> Verdict:
        """Evaluate each ``(label, intervention)`` branch and
        aggregate the results.

        Raises :class:`ValueError` when the branch list is empty —
        a verdict over zero branches has no operational meaning.
        """
        results: list[CounterfactualBranch] = []
        unsafe_results: list[CounterfactualBranch] = []
        for label, intervention in branches:
            values = intervention.apply(self._graph, inputs)
            outcome: BranchOutcome = "safe" if self._invariant(values) else "unsafe"
            branch = CounterfactualBranch(
                label=label,
                intervention=intervention,
                values=values,
                outcome=outcome,
            )
            results.append(branch)
            if outcome == "unsafe":
                unsafe_results.append(branch)
        if not results:
            raise ValueError("at least one branch is required")
        return Verdict(
            total=len(results),
            safe=sum(1 for b in results if b.outcome == "safe"),
            unsafe_branches=tuple(unsafe_results),
        )
