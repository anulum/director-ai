# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — propositional Formula AST

"""Immutable AST for propositional logic.

Every node is a ``@dataclass(frozen=True)`` so the nodes compose
naturally and hash cleanly for memoised CNF conversion. Child
ordering is preserved — the AST is structural, not
commutatively normalised.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variable:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Variable.name must be non-empty")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not:
    operand: Formula

    def __str__(self) -> str:
        return f"¬{_wrap(self.operand)}"


@dataclass(frozen=True)
class And:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True)
class Or:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass(frozen=True)
class Implies:
    antecedent: Formula
    consequent: Formula

    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"


@dataclass(frozen=True)
class Iff:
    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} ↔ {self.right})"


Formula = Variable | Not | And | Or | Implies | Iff


def _wrap(f: Formula) -> str:
    if isinstance(f, Variable):
        return str(f)
    return str(f)


def variables(formula: Formula) -> frozenset[str]:
    """Collect every variable name in ``formula``."""
    if isinstance(formula, Variable):
        return frozenset({formula.name})
    if isinstance(formula, Not):
        return variables(formula.operand)
    if isinstance(formula, And | Or):
        return variables(formula.left) | variables(formula.right)
    if isinstance(formula, Implies):
        return variables(formula.antecedent) | variables(formula.consequent)
    if isinstance(formula, Iff):
        return variables(formula.left) | variables(formula.right)
    raise TypeError(f"unknown formula type {type(formula).__name__}")  # pragma: no cover
