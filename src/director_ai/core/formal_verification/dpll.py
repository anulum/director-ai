# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DpllSolver

"""Davis-Putnam-Logemann-Loveland SAT solver.

Implementation notes:

* **Unit propagation** — whenever a clause reduces to a single
  literal, assign that literal and propagate.
* **Pure literal elimination** — if a variable appears with only
  one polarity across all remaining clauses, assign that polarity
  without branching.
* **Chronological backtracking** — pick the most-frequent
  unassigned literal as the branch variable and try both
  polarities.

The solver returns a :class:`Solution` with the model on SAT and
no model on UNSAT. It also exposes the number of decisions and
propagations so callers can spot pathological instances.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .cnf import Clause, Literal


@dataclass(frozen=True)
class Solution:
    """Outcome of a solve. ``satisfiable`` is the headline; when
    it is true, ``model`` assigns every variable in the input to
    ``True`` or ``False``.
    """

    satisfiable: bool
    model: dict[str, bool] = field(default_factory=dict)
    decisions: int = 0
    propagations: int = 0


class DpllSolver:
    """Seeded DPLL solver.

    Parameters
    ----------
    max_decisions :
        Upper bound on branch decisions to protect against
        pathological inputs. Default ``100_000``; raises
        :class:`TimeoutError` when exceeded.
    """

    def __init__(self, *, max_decisions: int = 100_000) -> None:
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        self._max_decisions = max_decisions

    def solve(self, clauses: tuple[Clause, ...]) -> Solution:
        self._decisions = 0
        self._propagations = 0
        assignment: dict[str, bool] = {}
        result = self._dpll(list(clauses), assignment)
        if result is None:
            return Solution(
                satisfiable=False,
                decisions=self._decisions,
                propagations=self._propagations,
            )
        return Solution(
            satisfiable=True,
            model=result,
            decisions=self._decisions,
            propagations=self._propagations,
        )

    def _dpll(
        self,
        clauses: list[Clause],
        assignment: dict[str, bool],
    ) -> dict[str, bool] | None:
        # Reduce with the current assignment.
        reduced = _reduce_clauses(clauses, assignment)
        if reduced is None:
            return None  # A clause collapsed to false — conflict.
        if not reduced:
            return dict(assignment)  # Every clause is satisfied.
        # Unit propagation.
        unit_literal = _find_unit(reduced)
        while unit_literal is not None:
            self._propagations += 1
            if (
                unit_literal.name in assignment
                and assignment[unit_literal.name] != unit_literal.positive
            ):
                return None
            assignment[unit_literal.name] = unit_literal.positive
            reduced = _reduce_clauses(reduced, assignment)
            if reduced is None:
                return None
            if not reduced:
                return dict(assignment)
            unit_literal = _find_unit(reduced)
        # Pure literal elimination.
        pure = _find_pure(reduced)
        if pure is not None:
            assignment[pure.name] = pure.positive
            return self._dpll(reduced, assignment)
        # Branch on the most frequent literal.
        branch_var = _pick_branch(reduced, assignment)
        if branch_var is None:
            # No unassigned variables and we did not short-circuit —
            # every clause must have been satisfied already.
            return dict(assignment)
        self._decisions += 1
        if self._decisions > self._max_decisions:
            raise TimeoutError(f"exceeded {self._max_decisions} DPLL decisions")
        for polarity in (True, False):
            branch_assignment = dict(assignment)
            branch_assignment[branch_var] = polarity
            result = self._dpll(reduced, branch_assignment)
            if result is not None:
                return result
        return None


def _reduce_clauses(
    clauses: list[Clause], assignment: dict[str, bool]
) -> list[Clause] | None:
    """Return the simplified clause list under ``assignment``.

    A clause containing a satisfied literal drops out; a clause
    whose every literal is falsified returns ``None`` (conflict).
    Literals whose variable is unassigned stay.
    """
    reduced: list[Clause] = []
    for clause in clauses:
        keep: list[Literal] = []
        satisfied = False
        for literal in clause:
            value = assignment.get(literal.name)
            if value is None:
                keep.append(literal)
            elif value == literal.positive:
                satisfied = True
                break
        if satisfied:
            continue
        if not keep:
            return None
        reduced.append(tuple(keep))
    return reduced


def _find_unit(clauses: list[Clause]) -> Literal | None:
    for clause in clauses:
        if len(clause) == 1:
            return clause[0]
    return None


def _find_pure(clauses: list[Clause]) -> Literal | None:
    polarities: dict[str, set[bool]] = {}
    for clause in clauses:
        for literal in clause:
            polarities.setdefault(literal.name, set()).add(literal.positive)
    for name in sorted(polarities):
        pols = polarities[name]
        if len(pols) == 1:
            return Literal(name=name, positive=next(iter(pols)))
    return None


def _pick_branch(clauses: list[Clause], assignment: dict[str, bool]) -> str | None:
    counter: Counter[str] = Counter()
    for clause in clauses:
        for literal in clause:
            if literal.name in assignment:
                continue
            counter[literal.name] += 1
    if not counter:
        return None
    # ``Counter.most_common`` is insertion-stable; for a
    # reproducible branch choice across Python versions we also
    # break ties by the lexicographically-smallest name.
    top_count = counter.most_common(1)[0][1]
    candidates = [name for name, count in counter.items() if count == top_count]
    return min(candidates)
