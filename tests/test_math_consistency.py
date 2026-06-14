# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — arithmetic consistency tests

"""Evaluator, equation-extraction, polyglot-parity and reasoning-integration
coverage for chain-of-thought arithmetic verification."""

from __future__ import annotations

import math

import pytest

from director_ai.core.verification import math_consistency as mc
from director_ai.core.verification.math_consistency import (
    ArithmeticCheck,
    MathConsistencyResult,
    verify_arithmetic,
)
from director_ai.core.verification.reasoning_verifier import verify_reasoning_chain

# --------------------------------------------------------------------------- #
# evaluator — Python fallback (forced) and Rust fast path                      #
# --------------------------------------------------------------------------- #


@pytest.fixture
def python_only(monkeypatch):
    monkeypatch.setattr(mc, "_RUST_MATH", False)
    monkeypatch.setattr(mc, "rust_eval_arithmetic", None)


def test_python_eval_precedence_parens_unary(python_only):
    assert mc._eval_arithmetic("3 + 4") == 7.0
    assert mc._eval_arithmetic("2 + 3 * 4") == 14.0
    assert mc._eval_arithmetic("(2 + 3) * 4") == 20.0
    assert mc._eval_arithmetic("-5 + 8") == 3.0
    assert mc._eval_arithmetic("+5") == 5.0


def test_python_eval_unicode_and_commas(python_only):
    assert mc._eval_arithmetic("12 × 5") == 60.0
    assert mc._eval_arithmetic("100 ÷ 4") == 25.0
    assert mc._eval_arithmetic("6 · 7") == 42.0
    assert mc._eval_arithmetic("1,000 + 234") == 1234.0


def test_python_eval_invalid_inputs_are_nan(python_only):
    assert math.isnan(mc._eval_arithmetic("3 +"))
    assert math.isnan(mc._eval_arithmetic("1e308"))  # scientific notation rejected
    assert math.isnan(mc._eval_arithmetic("1 / 0"))  # zero division
    assert math.isnan(mc._eval_arithmetic("(2 + 3"))  # unbalanced
    assert math.isnan(mc._eval_arithmetic("()"))  # parses but unsupported node


def test_python_eval_non_finite_and_overflow_are_nan(python_only):
    big = "1" + "0" * 200  # 1e200, finite
    assert math.isnan(mc._eval_arithmetic(f"{big} * {big}"))  # -> inf -> NaN
    assert math.isnan(mc._eval_arithmetic("1" + "0" * 400))  # float() overflow


def test_rust_and_python_eval_parity():
    cases = ["3 + 4", "2 + 3 * 4", "(120 - 20) / 4", "12 × 5", "10 / 3", "1,000 + 1", "1 / 0"]
    rust = [mc._eval_arithmetic(c) for c in cases]
    saved_flag, saved_fn = mc._RUST_MATH, mc.rust_eval_arithmetic
    mc._RUST_MATH, mc.rust_eval_arithmetic = False, None
    try:
        py = [mc._eval_arithmetic(c) for c in cases]
    finally:
        mc._RUST_MATH, mc.rust_eval_arithmetic = saved_flag, saved_fn
    for r, p in zip(rust, py, strict=True):
        assert (math.isnan(r) and math.isnan(p)) or r == p


# --------------------------------------------------------------------------- #
# verify_arithmetic                                                            #
# --------------------------------------------------------------------------- #


def test_detects_wrong_and_right_equations():
    result = verify_arithmetic("3 + 4 = 8 but 12 * 5 = 60.")
    assert isinstance(result, MathConsistencyResult)
    assert result.equations_found == 2
    assert result.valid is False
    wrong = result.errors
    assert len(wrong) == 1 and wrong[0].claimed == 8.0 and wrong[0].computed == 7.0


def test_equals_word_variants():
    assert verify_arithmetic("100 / 4 equals 25").valid is True
    assert verify_arithmetic("100 / 4 equals to 30").valid is False
    assert verify_arithmetic("2 + 2 is equal to 4").valid is True


def test_prose_without_operator_is_ignored():
    result = verify_arithmetic("The temperature is 25 degrees and the CEO is Sam Altman.")
    assert result.equations_found == 0 and result.valid is True


def test_bare_number_equality_without_operator_is_ignored():
    # "42 = 50" matches the equation shape but the left side carries no operator,
    # so it is a definition/label, not arithmetic to check.
    result = verify_arithmetic("Let x = 5 where 42 = 50 nominally.")
    assert result.equations_found == 0


def test_unevaluable_left_side_is_skipped():
    # The left side has an operator (so it is considered) but evaluates to NaN
    # (division by zero), so it is not counted as a checkable equation.
    result = verify_arithmetic("5 / 0 = 3")
    assert result.equations_found == 0


def test_thousands_separator_and_currency():
    assert verify_arithmetic("1,000 + 1,000 = 2,000").valid is True
    assert verify_arithmetic("1000 * 3 = $3,000").valid is True


def test_tolerance_controls_match():
    assert verify_arithmetic("1 / 3 = 0.3333", abs_tol=1e-3).valid is True
    assert verify_arithmetic("1 / 3 = 0.3333", abs_tol=1e-9, rel_tol=1e-9).valid is False


def test_arithmetic_check_dataclass_fields():
    check = ArithmeticCheck(expression="3 + 4 = 8", claimed=8.0, computed=7.0, correct=False)
    assert check.correct is False and check.computed == 7.0


# --------------------------------------------------------------------------- #
# integration into verify_reasoning_chain                                      #
# --------------------------------------------------------------------------- #


def test_reasoning_chain_flags_math_error():
    text = "Step 1: start with 3. Step 2: 3 + 4 = 8. Step 3: therefore the answer is 8."
    result = verify_reasoning_chain(text)
    assert [c.expression for c in result.math_errors] == ["3 + 4 = 8"]
    assert result.chain_valid is False
    assert result.issues_found >= 1


def test_reasoning_chain_math_can_be_disabled():
    text = "Step 1: 3 + 4 = 8. Step 2: done."
    result = verify_reasoning_chain(text, check_math=False)
    assert result.math_errors == []


def test_reasoning_chain_single_line_still_checks_math():
    # Fewer than two steps: the early-return path must still surface math errors.
    result = verify_reasoning_chain("Quick note: 10 / 4 = 20.")
    assert result.steps_found < 2
    assert len(result.math_errors) == 1
    assert result.chain_valid is False
