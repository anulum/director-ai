# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CnfConverter

"""Convert a :class:`Formula` to conjunctive normal form.

Rewrite sequence:

1. Replace ``A ↔ B`` with ``(A → B) ∧ (B → A)``.
2. Replace ``A → B`` with ``¬A ∨ B``.
3. Push negations inward (De Morgan).
4. Distribute disjunctions over conjunctions.
5. Collapse to a list-of-clauses representation where each
   clause is a disjunction of :class:`Literal` (variable name +
   polarity).

Clause order inside the output is preserved from the distribution
step; variable order inside a clause is deduplicated by name
with the first occurrence's polarity winning.
"""

from __future__ import annotations

from dataclasses import dataclass

from .formula import And, Formula, Iff, Implies, Not, Or, Variable


@dataclass(frozen=True)
class Literal:
    """Variable name + polarity. ``positive=False`` represents
    ``¬name``."""

    name: str
    positive: bool

    def __str__(self) -> str:
        return self.name if self.positive else f"¬{self.name}"

    def negate(self) -> Literal:
        return Literal(name=self.name, positive=not self.positive)


Clause = tuple[Literal, ...]


class CnfConverter:
    """Pure-function CNF converter."""

    def convert(self, formula: Formula) -> tuple[Clause, ...]:
        without_iff = self._remove_iff(formula)
        without_impl = self._remove_implications(without_iff)
        nnf = self._push_negations(without_impl)
        distributed = self._distribute(nnf)
        return self._flatten(distributed)

    def _remove_iff(self, formula: Formula) -> Formula:
        if isinstance(formula, Variable):
            return formula
        if isinstance(formula, Not):
            return Not(self._remove_iff(formula.operand))
        if isinstance(formula, And):
            return And(
                self._remove_iff(formula.left), self._remove_iff(formula.right)
            )
        if isinstance(formula, Or):
            return Or(
                self._remove_iff(formula.left), self._remove_iff(formula.right)
            )
        if isinstance(formula, Implies):
            return Implies(
                self._remove_iff(formula.antecedent),
                self._remove_iff(formula.consequent),
            )
        if isinstance(formula, Iff):
            left = self._remove_iff(formula.left)
            right = self._remove_iff(formula.right)
            return And(Implies(left, right), Implies(right, left))
        raise TypeError(f"unknown node {type(formula).__name__}")  # pragma: no cover

    def _remove_implications(self, formula: Formula) -> Formula:
        if isinstance(formula, Variable):
            return formula
        if isinstance(formula, Not):
            return Not(self._remove_implications(formula.operand))
        if isinstance(formula, And):
            return And(
                self._remove_implications(formula.left),
                self._remove_implications(formula.right),
            )
        if isinstance(formula, Or):
            return Or(
                self._remove_implications(formula.left),
                self._remove_implications(formula.right),
            )
        if isinstance(formula, Implies):
            return Or(
                Not(self._remove_implications(formula.antecedent)),
                self._remove_implications(formula.consequent),
            )
        raise TypeError(
            f"unexpected node {type(formula).__name__}"
        )  # pragma: no cover — Iff already removed

    def _push_negations(self, formula: Formula) -> Formula:
        if isinstance(formula, Variable):
            return formula
        if isinstance(formula, And):
            return And(
                self._push_negations(formula.left),
                self._push_negations(formula.right),
            )
        if isinstance(formula, Or):
            return Or(
                self._push_negations(formula.left),
                self._push_negations(formula.right),
            )
        if isinstance(formula, Not):
            inner = formula.operand
            if isinstance(inner, Variable):
                return formula
            if isinstance(inner, Not):
                return self._push_negations(inner.operand)
            if isinstance(inner, And):
                return Or(
                    self._push_negations(Not(inner.left)),
                    self._push_negations(Not(inner.right)),
                )
            if isinstance(inner, Or):
                return And(
                    self._push_negations(Not(inner.left)),
                    self._push_negations(Not(inner.right)),
                )
        raise TypeError(
            f"unexpected node {type(formula).__name__}"
        )  # pragma: no cover — Implies + Iff already removed

    def _distribute(self, formula: Formula) -> Formula:
        if isinstance(formula, Variable) or (
            isinstance(formula, Not) and isinstance(formula.operand, Variable)
        ):
            return formula
        if isinstance(formula, And):
            return And(self._distribute(formula.left), self._distribute(formula.right))
        if isinstance(formula, Or):
            left = self._distribute(formula.left)
            right = self._distribute(formula.right)
            if isinstance(left, And):
                return And(
                    self._distribute(Or(left.left, right)),
                    self._distribute(Or(left.right, right)),
                )
            if isinstance(right, And):
                return And(
                    self._distribute(Or(left, right.left)),
                    self._distribute(Or(left, right.right)),
                )
            return Or(left, right)
        raise TypeError(
            f"unexpected node {type(formula).__name__}"
        )  # pragma: no cover

    def _flatten(self, formula: Formula) -> tuple[Clause, ...]:
        if isinstance(formula, And):
            return self._flatten(formula.left) + self._flatten(formula.right)
        return (self._flatten_clause(formula),)

    def _flatten_clause(self, formula: Formula) -> Clause:
        if isinstance(formula, Or):
            return self._flatten_clause(formula.left) + self._flatten_clause(
                formula.right
            )
        if isinstance(formula, Variable):
            return (Literal(name=formula.name, positive=True),)
        if isinstance(formula, Not) and isinstance(formula.operand, Variable):
            return (Literal(name=formula.operand.name, positive=False),)
        raise TypeError(
            f"unexpected literal node {type(formula).__name__}"
        )  # pragma: no cover
