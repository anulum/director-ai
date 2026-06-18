# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — arithmetic consistency for chain-of-thought

"""Verify the explicit arithmetic inside a reasoning chain.

Chain-of-thought answers state their working as equations — ``3 + 4 = 8``,
``12 × 5 = 60``, ``(120 - 20) / 4 = 25`` — and a model that reasons fluently
still gets the sums wrong. ``numeric_verifier`` checks numeric *plausibility*
(percentage changes, date logic, magnitudes); this module checks whether each
asserted equality actually holds.

Equations are extracted with a regex (``left = right`` / ``left equals right`` /
``left is equal to right``, where ``left`` carries at least one operator so a
plain ``x is 5`` is not mistaken for arithmetic), the left side is evaluated, and
the result is compared to the asserted right side within a tolerance.

The evaluator runs through the Rust ``rust_eval_arithmetic`` kernel when the
compiled extension is installed and a pure-Python ``ast`` evaluator otherwise;
the two are bit-for-bit identical (same IEEE-754 doubles, same precedence and
left-associativity), so the dispatch is purely a speed choice.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "ArithmeticCheck",
    "MathConsistencyResult",
    "verify_arithmetic",
]

try:
    from backfire_kernel import rust_eval_arithmetic

    _RUST_MATH = True
except ImportError:  # pragma: no cover - exercised on installs without the kernel
    rust_eval_arithmetic = None
    _RUST_MATH = False

_AST_OPS: dict[type[ast.AST], Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# left = right | left equals [to] right | left is equal to right
# The LHS body and whitespace runs are bounded ({0,N} not *) so the pattern
# cannot backtrack polynomially on long digit/space runs (CodeQL
# py/polynomial-redos); 200 chars far exceeds any real arithmetic expression.
_EQUATION_RE = re.compile(
    r"(?P<lhs>[\d(][\d\s,.+\-*/×÷·()]{0,200}?[\d)])\s{0,4}"
    r"(?:=|equals?(?:\s{1,4}to)?|is\s{1,4}equal\s{1,4}to)\s{0,4}"
    r"(?P<rhs>-?\$?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?))",
    re.IGNORECASE,
)
# A binary operator sitting between two operands (so the left side is arithmetic,
# not a bare number being named).
_HAS_OPERATOR_RE = re.compile(r"[\d)]\s*[-+*/×÷·]\s*[\d(]")
# Characters the Rust tokeniser accepts (after Unicode-operator normalisation).
_DISALLOWED_CHAR_RE = re.compile(r"[^0-9.\s+\-*/()]")


def _python_eval(expr: str) -> float:
    """Evaluate an arithmetic expression with the stdlib ``ast`` (Rust fallback).

    Mirrors ``backfire_core::compute::eval_arithmetic`` exactly: Unicode operators
    normalised, thousands separators dropped, ``+ - * /`` with standard precedence
    over IEEE-754 doubles. Any non-arithmetic input returns ``NaN``.
    """
    normalised = (
        expr.replace("×", "*").replace("·", "*").replace("÷", "/").replace(",", "")
    )
    # Mirror the Rust tokeniser's accepted character set exactly (digits, the four
    # operators, parentheses, decimal point, whitespace). This rejects scientific
    # notation, names, etc. so the two backends agree bit-for-bit.
    if _DISALLOWED_CHAR_RE.search(normalised):
        return math.nan
    try:
        tree = ast.parse(normalised, mode="eval")
    except SyntaxError:
        return math.nan

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.BinOp) and type(node.op) in _AST_OPS:
            return float(_AST_OPS[type(node.op)](_ev(node.left), _ev(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _AST_OPS:
            return float(_AST_OPS[type(node.op)](_ev(node.operand)))
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("unsupported expression")

    try:
        result = float(_ev(tree))
    except (ValueError, ZeroDivisionError, TypeError, OverflowError):
        return math.nan
    # Match the Rust kernel: a non-finite result is reported as NaN.
    return result if math.isfinite(result) else math.nan


def _eval_arithmetic(expr: str) -> float:
    """Evaluate *expr* to a float (Rust fast path or Python), ``NaN`` if invalid."""
    if _RUST_MATH and rust_eval_arithmetic is not None:
        return float(rust_eval_arithmetic(expr))
    return _python_eval(expr)


@dataclass
class ArithmeticCheck:
    """One asserted equation and whether it holds."""

    expression: str
    claimed: float
    computed: float
    correct: bool


@dataclass
class MathConsistencyResult:
    """Arithmetic verification over a text."""

    checks: list[ArithmeticCheck] = field(default_factory=list)
    valid: bool = True

    @property
    def errors(self) -> list[ArithmeticCheck]:
        """Return asserted equations whose computed value disagrees."""
        return [c for c in self.checks if not c.correct]

    @property
    def equations_found(self) -> int:
        """Return the number of recognized equations in the text."""
        return len(self.checks)


def _parse_rhs(raw: str) -> float:
    return float(raw.replace("$", "").replace(",", ""))


def _is_close(
    computed: float, claimed: float, *, rel_tol: float, abs_tol: float
) -> bool:
    return math.isclose(computed, claimed, rel_tol=rel_tol, abs_tol=abs_tol)


def verify_arithmetic(
    text: str,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> MathConsistencyResult:
    """Find ``left = right`` equations in *text* and verify each one.

    Parameters
    ----------
    text:
        The reasoning text to scan.
    rel_tol / abs_tol:
        Relative and absolute tolerance for comparing the computed left side to
        the asserted right side (passed to :func:`math.isclose`).

    Returns
    -------
    MathConsistencyResult
        One :class:`ArithmeticCheck` per recognised equation; ``valid`` is False
        when any equation is wrong.
    """
    checks: list[ArithmeticCheck] = []
    for match in _EQUATION_RE.finditer(text):
        lhs = match.group("lhs").strip()
        if not _HAS_OPERATOR_RE.search(lhs):
            continue
        # ``_eval_arithmetic`` already maps any non-finite result to NaN.
        computed = _eval_arithmetic(lhs)
        if math.isnan(computed):
            continue
        try:
            claimed = _parse_rhs(match.group("rhs"))
        except ValueError:  # pragma: no cover - regex guarantees a numeric rhs
            continue
        correct = _is_close(computed, claimed, rel_tol=rel_tol, abs_tol=abs_tol)
        checks.append(
            ArithmeticCheck(
                expression=match.group(0).strip(),
                claimed=claimed,
                computed=computed,
                correct=correct,
            )
        )
    return MathConsistencyResult(
        checks=checks,
        valid=all(c.correct for c in checks),
    )
