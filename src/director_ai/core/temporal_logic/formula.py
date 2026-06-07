# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Linear Temporal Logic Formula Algebra
"""Linear Temporal Logic (LTL) formula algebra for runtime monitoring.

The formula AST is canonical and self-simplifying: ``And``/``Or`` carry a
``frozenset`` of operands, so the smart constructors flatten nesting, absorb the
``TOP``/``BOTTOM`` constants, and deduplicate operands. That canonicalisation is
what keeps the *progressed* formula bounded as a trace grows — ``progress`` of an
``Always`` collapses back to itself once the per-step obligation is discharged,
instead of accumulating one conjunct per step.

Two evaluation entry points implement runtime LTL over a finite, growing trace:

* :func:`progress` rewrites a formula by one observed state (its Boolean
  derivative). ``BOTTOM`` means definitively violated, ``TOP`` definitively
  satisfied, anything else is still pending on future states.
* :func:`value_at_end` resolves the residual obligation when the trace ends, with
  strong-eventuality finite-trace semantics: ``Always`` is vacuously satisfied,
  but an undischarged ``Eventually``/``Until``/``Next`` is a violation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class Formula:
    """Base type for every LTL node (sealed, only the subclasses below)."""

    __slots__ = ()


@dataclass(frozen=True)
class _Top(Formula):
    """The constant ``true`` (a definitively satisfied obligation)."""

    __slots__ = ()

    def __str__(self) -> str:
        return "⊤"


@dataclass(frozen=True)
class _Bottom(Formula):
    """The constant ``false`` (a definitively violated obligation)."""

    __slots__ = ()

    def __str__(self) -> str:
        return "⊥"


TOP = _Top()
BOTTOM = _Bottom()


@dataclass(frozen=True)
class Atom(Formula):
    """An atomic proposition; true in a state when its name is present."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Not(Formula):
    """Logical negation of ``operand``."""

    operand: Formula

    def __str__(self) -> str:
        return f"¬{self.operand}"


@dataclass(frozen=True)
class And(Formula):
    """Conjunction of a canonical (flattened, deduplicated) operand set."""

    operands: frozenset[Formula]

    def __str__(self) -> str:
        return "(" + " ∧ ".join(sorted(str(o) for o in self.operands)) + ")"


@dataclass(frozen=True)
class Or(Formula):
    """Disjunction of a canonical (flattened, deduplicated) operand set."""

    operands: frozenset[Formula]

    def __str__(self) -> str:
        return "(" + " ∨ ".join(sorted(str(o) for o in self.operands)) + ")"


@dataclass(frozen=True)
class Next(Formula):
    """``X operand`` — ``operand`` must hold in the next state."""

    operand: Formula

    def __str__(self) -> str:
        return f"X {self.operand}"


@dataclass(frozen=True)
class Always(Formula):
    """``G operand`` — ``operand`` must hold in every remaining state."""

    operand: Formula

    def __str__(self) -> str:
        return f"G {self.operand}"


@dataclass(frozen=True)
class Eventually(Formula):
    """``F operand`` — ``operand`` must hold in some remaining state."""

    operand: Formula

    def __str__(self) -> str:
        return f"F {self.operand}"


@dataclass(frozen=True)
class Until(Formula):
    """``left U right`` — ``left`` holds until ``right`` eventually holds."""

    left: Formula
    right: Formula

    def __str__(self) -> str:
        return f"({self.left} U {self.right})"


# ── Smart constructors (always return a canonical, simplified formula) ────────


def atom(name: str) -> Formula:
    """Build an atomic proposition; the name must be non-empty."""
    if not name:
        raise ValueError("atom name must be non-empty")
    return Atom(name)


def not_(formula: Formula) -> Formula:
    """Negate ``formula``, collapsing constants and double negation."""
    if formula is TOP:
        return BOTTOM
    if formula is BOTTOM:
        return TOP
    if isinstance(formula, Not):
        return formula.operand
    return Not(formula)


def _flatten(formula: Formula, cls: type) -> Iterable[Formula]:
    if isinstance(formula, cls):
        operands = formula.operands  # type: ignore[attr-defined]
        for sub in operands:
            yield from _flatten(sub, cls)
    else:
        yield formula


def and_(*formulas: Formula) -> Formula:
    """Conjunction with flattening, dedup, and ⊤/⊥ absorption."""
    operands: set[Formula] = set()
    for formula in formulas:
        for leaf in _flatten(formula, And):
            if leaf is BOTTOM:
                return BOTTOM
            if leaf is TOP:
                continue
            operands.add(leaf)
    if not operands:
        return TOP
    if len(operands) == 1:
        return next(iter(operands))
    return And(frozenset(operands))


def or_(*formulas: Formula) -> Formula:
    """Disjunction with flattening, dedup, and ⊤/⊥ absorption."""
    operands: set[Formula] = set()
    for formula in formulas:
        for leaf in _flatten(formula, Or):
            if leaf is TOP:
                return TOP
            if leaf is BOTTOM:
                continue
            operands.add(leaf)
    if not operands:
        return BOTTOM
    if len(operands) == 1:
        return next(iter(operands))
    return Or(frozenset(operands))


def implies(antecedent: Formula, consequent: Formula) -> Formula:
    """Material implication ``antecedent → consequent``."""
    return or_(not_(antecedent), consequent)


def G(formula: Formula) -> Formula:  # noqa: N802 — LTL operator name
    """``Always`` (globally) ``formula``."""
    if formula is TOP:
        return TOP
    if formula is BOTTOM:
        return BOTTOM
    return Always(formula)


def F(formula: Formula) -> Formula:  # noqa: N802 — LTL operator name
    """``Eventually`` (finally) ``formula``."""
    if formula is TOP:
        return TOP
    if formula is BOTTOM:
        return BOTTOM
    return Eventually(formula)


def X(formula: Formula) -> Formula:  # noqa: N802 — LTL operator name
    """``Next`` ``formula``."""
    return Next(formula)


def U(left: Formula, right: Formula) -> Formula:  # noqa: N802 — LTL operator
    """``left`` until ``right``."""
    if right is TOP:
        return TOP
    return Until(left, right)


# ── Runtime evaluation ───────────────────────────────────────────────────────


def progress(formula: Formula, state: frozenset[str]) -> Formula:
    """Return the Boolean derivative of ``formula`` after observing ``state``.

    ``state`` is the set of atomic propositions true in the current step. The
    result is the obligation that remains for future steps: ``BOTTOM`` if the
    formula is now violated, ``TOP`` if it is now satisfied, otherwise a residual
    formula. Built with the smart constructors so the residual stays canonical
    and bounded.
    """
    if formula is TOP or formula is BOTTOM:
        return formula
    if isinstance(formula, Atom):
        return TOP if formula.name in state else BOTTOM
    if isinstance(formula, Not):
        return not_(progress(formula.operand, state))
    if isinstance(formula, And):
        return and_(*(progress(o, state) for o in formula.operands))
    if isinstance(formula, Or):
        return or_(*(progress(o, state) for o in formula.operands))
    if isinstance(formula, Next):
        # The "next" obligation becomes the wrapped formula for the next step.
        return formula.operand
    if isinstance(formula, Always):
        return and_(progress(formula.operand, state), formula)
    if isinstance(formula, Eventually):
        return or_(progress(formula.operand, state), formula)
    if isinstance(formula, Until):
        return or_(
            progress(formula.right, state),
            and_(progress(formula.left, state), formula),
        )
    raise TypeError(f"unknown formula node: {formula!r}")  # pragma: no cover


def value_at_end(formula: Formula) -> bool:
    """Resolve a residual obligation when the trace ends (no further states).

    Strong-eventuality finite-trace semantics: ``Always`` is vacuously satisfied
    (no remaining state can violate it), while an undischarged ``Eventually``,
    ``Until`` right-hand side, ``Next``, or bare atom is an unmet obligation and
    therefore ``False``.
    """
    if formula is TOP:
        return True
    if formula is BOTTOM:
        return False
    if isinstance(formula, Atom):
        return False
    if isinstance(formula, Not):
        return not value_at_end(formula.operand)
    if isinstance(formula, And):
        return all(value_at_end(o) for o in formula.operands)
    if isinstance(formula, Or):
        return any(value_at_end(o) for o in formula.operands)
    if isinstance(formula, Next):
        return False
    if isinstance(formula, Always):
        return True
    if isinstance(formula, Eventually):
        return False
    if isinstance(formula, Until):
        return False
    raise TypeError(f"unknown formula node: {formula!r}")  # pragma: no cover
