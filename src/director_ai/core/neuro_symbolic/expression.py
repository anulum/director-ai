# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Neuro-Symbolic Compliance Expression AST
"""Solver-agnostic typed expression AST for compliance constraints.

A compliance policy is written in this small typed language — boolean, integer,
and real variables with comparisons, arithmetic, and boolean connectives — so the
policy can be authored, serialised, and cross-checked without importing a solver.
The :mod:`director_ai.core.neuro_symbolic.engine` compiles it to Z3 only when a
satisfiability question is actually asked.
"""

from __future__ import annotations

from dataclasses import dataclass

BOOL = "bool"
INT = "int"
REAL = "real"
_SORTS = frozenset({BOOL, INT, REAL})

_COMPARE_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})
_ARITH_OPS = frozenset({"add", "sub", "mul"})

_COMPARE_SYMBOL = {
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}
_ARITH_SYMBOL = {"add": "+", "sub": "-", "mul": "*"}


class Expr:
    """Base type for every expression node."""

    __slots__ = ()


@dataclass(frozen=True)
class Var(Expr):
    """A typed variable; ``sort`` is one of ``bool``, ``int``, ``real``."""

    name: str
    sort: str = REAL

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Var.name must be non-empty")
        if self.sort not in _SORTS:
            raise ValueError(f"unknown sort {self.sort!r}; expected one of {_SORTS}")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Expr):
    """A literal boolean, integer, or real constant."""

    value: bool | int | float

    def __post_init__(self) -> None:
        if not isinstance(self.value, bool | int | float):
            raise ValueError("Const.value must be bool, int, or float")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Not(Expr):
    """Boolean negation."""

    operand: Expr

    def __str__(self) -> str:
        return f"¬{self.operand}"


@dataclass(frozen=True)
class BoolOp(Expr):
    """A binary boolean connective: ``and`` | ``or`` | ``implies``."""

    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in {"and", "or", "implies"}:
            raise ValueError(f"unknown boolean op {self.op!r}")

    def __str__(self) -> str:
        sym = {"and": "∧", "or": "∨", "implies": "→"}[self.op]
        return f"({self.left} {sym} {self.right})"


@dataclass(frozen=True)
class Compare(Expr):
    """A relational comparison between two terms."""

    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in _COMPARE_OPS:
            raise ValueError(f"unknown comparison op {self.op!r}")

    def __str__(self) -> str:
        return f"({self.left} {_COMPARE_SYMBOL[self.op]} {self.right})"


@dataclass(frozen=True)
class Arith(Expr):
    """A binary arithmetic operation."""

    op: str
    left: Expr
    right: Expr

    def __post_init__(self) -> None:
        if self.op not in _ARITH_OPS:
            raise ValueError(f"unknown arithmetic op {self.op!r}")

    def __str__(self) -> str:
        return f"({self.left} {_ARITH_SYMBOL[self.op]} {self.right})"


# ── Constructors ─────────────────────────────────────────────────────────────


def var(name: str, sort: str = REAL) -> Var:
    """Declare a typed variable."""
    return Var(name, sort)


def lit(value: bool | int | float) -> Const:
    """A literal constant."""
    return Const(value)


def not_(operand: Expr) -> Not:
    """Boolean negation."""
    return Not(operand)


def _fold(op: str, operands: tuple[Expr, ...]) -> Expr:
    if not operands:
        raise ValueError(f"{op} requires at least one operand")
    result = operands[0]
    for nxt in operands[1:]:
        result = BoolOp(op, result, nxt)
    return result


def and_(*operands: Expr) -> Expr:
    """Conjunction of one or more expressions."""
    return _fold("and", operands)


def or_(*operands: Expr) -> Expr:
    """Disjunction of one or more expressions."""
    return _fold("or", operands)


def implies(antecedent: Expr, consequent: Expr) -> BoolOp:
    """Material implication."""
    return BoolOp("implies", antecedent, consequent)


def eq(left: Expr, right: Expr) -> Compare:
    """Equality comparison."""
    return Compare("eq", left, right)


def ne(left: Expr, right: Expr) -> Compare:
    """Inequality comparison."""
    return Compare("ne", left, right)


def lt(left: Expr, right: Expr) -> Compare:
    """Strict less-than."""
    return Compare("lt", left, right)


def le(left: Expr, right: Expr) -> Compare:
    """Less-than-or-equal."""
    return Compare("le", left, right)


def gt(left: Expr, right: Expr) -> Compare:
    """Strict greater-than."""
    return Compare("gt", left, right)


def ge(left: Expr, right: Expr) -> Compare:
    """Greater-than-or-equal."""
    return Compare("ge", left, right)


def add(left: Expr, right: Expr) -> Arith:
    """Addition."""
    return Arith("add", left, right)


def sub(left: Expr, right: Expr) -> Arith:
    """Subtraction."""
    return Arith("sub", left, right)


def mul(left: Expr, right: Expr) -> Arith:
    """Multiplication."""
    return Arith("mul", left, right)


def variables(expr: Expr) -> dict[str, str]:
    """Collect ``{name: sort}`` for every variable, rejecting sort conflicts."""
    found: dict[str, str] = {}
    _collect(expr, found)
    return found


def _collect(expr: Expr, found: dict[str, str]) -> None:
    if isinstance(expr, Var):
        existing = found.get(expr.name)
        if existing is not None and existing != expr.sort:
            raise ValueError(
                f"variable {expr.name!r} used as both {existing} and {expr.sort}"
            )
        found[expr.name] = expr.sort
    elif isinstance(expr, Const):
        return
    elif isinstance(expr, Not):
        _collect(expr.operand, found)
    elif isinstance(expr, BoolOp | Compare | Arith):
        _collect(expr.left, found)
        _collect(expr.right, found)
    else:  # pragma: no cover — exhaustive over the sealed Expr union
        raise TypeError(f"unknown expression node: {expr!r}")
