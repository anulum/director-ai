# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Neuro-Symbolic Compliance Engine Tests
"""Tests for the typed policy AST and the Z3-backed compliance engine.

The expression/policy layer is solver-agnostic and tested unconditionally; the
engine tests require ``z3`` (the [formal] extra) and skip otherwise.
"""

from __future__ import annotations

import builtins
import importlib.util
from typing import Any

import pytest

from director_ai.core.neuro_symbolic import (
    BOOL,
    INT,
    REAL,
    CompliancePolicy,
    ComplianceVerdict,
    Constraint,
    ConstraintViolation,
    NeuroSymbolicComplianceEngine,
    PolicyFormaliser,
    add,
    and_,
    eq,
    ge,
    gt,
    implies,
    le,
    lit,
    lt,
    mul,
    ne,
    not_,
    or_,
    sub,
    var,
    variables,
)

_HAS_Z3 = importlib.util.find_spec("z3") is not None
_needs_z3 = pytest.mark.skipif(not _HAS_Z3, reason="z3 (the [formal] extra) required")


class TestExpression:
    def test_var_validation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            var("")
        with pytest.raises(ValueError, match="unknown sort"):
            var("x", "complex")

    def test_const_validation(self) -> None:
        with pytest.raises(ValueError, match="bool, int, or float"):
            lit("nope")  # type: ignore[arg-type]

    def test_variables_collects_sorts(self) -> None:
        expr = and_(le(var("a", REAL), lit(5)), eq(var("b", BOOL), lit(True)))
        assert variables(expr) == {"a": REAL, "b": BOOL}

    def test_variables_rejects_sort_conflict(self) -> None:
        expr = and_(le(var("a", REAL), lit(5)), eq(var("a", INT), lit(3)))
        with pytest.raises(ValueError, match="both"):
            variables(expr)

    def test_fold_requires_operand(self) -> None:
        with pytest.raises(ValueError, match="at least one operand"):
            and_()

    def test_str_renderings(self) -> None:
        assert str(var("x")) == "x"
        assert str(not_(var("x", BOOL))) == "¬x"
        assert "∧" in str(and_(var("a", BOOL), var("b", BOOL)))
        assert "∨" in str(or_(var("a", BOOL), var("b", BOOL)))
        assert "→" in str(implies(var("a", BOOL), var("b", BOOL)))
        assert "<=" in str(le(var("x"), lit(5)))
        assert "+" in str(add(var("x"), lit(1)))

    def test_all_comparison_and_arith_constructors(self) -> None:
        x = var("x")
        assert ne(x, lit(1)).op == "ne"
        assert lt(x, lit(1)).op == "lt"
        assert gt(x, lit(1)).op == "gt"
        assert ge(x, lit(1)).op == "ge"
        assert sub(x, lit(1)).op == "sub"
        assert mul(x, lit(2)).op == "mul"

    def test_invalid_ops_rejected(self) -> None:
        from director_ai.core.neuro_symbolic import Arith, BoolOp, Compare

        with pytest.raises(ValueError, match="boolean op"):
            BoolOp("xor", var("a", BOOL), var("b", BOOL))
        with pytest.raises(ValueError, match="comparison op"):
            Compare("approx", var("x"), lit(1))
        with pytest.raises(ValueError, match="arithmetic op"):
            Arith("div", var("x"), lit(1))

    def test_variables_walks_not(self) -> None:
        assert variables(not_(eq(var("x", REAL), lit(1)))) == {"x": REAL}


class TestPolicyModel:
    def test_constraint_requires_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Constraint("", le(var("x"), lit(1)))

    def test_empty_policy_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one constraint"):
            CompliancePolicy([])

    def test_duplicate_names_rejected(self) -> None:
        c = Constraint("dup", le(var("x"), lit(1)))
        with pytest.raises(ValueError, match="unique"):
            CompliancePolicy([c, c])

    def test_cross_constraint_sort_conflict(self) -> None:
        with pytest.raises(ValueError, match="both"):
            CompliancePolicy(
                [
                    Constraint("a", le(var("x", REAL), lit(1))),
                    Constraint("b", eq(var("x", BOOL), lit(True))),
                ]
            )

    def test_variables_union(self) -> None:
        policy = CompliancePolicy(
            [
                Constraint("a", le(var("amount", REAL), lit(10))),
                Constraint("b", eq(var("ok", BOOL), lit(True))),
            ]
        )
        assert policy.variables() == {"amount": REAL, "ok": BOOL}
        assert len(policy.constraints) == 2

    def test_violation_and_verdict_to_dict(self) -> None:
        viol = ConstraintViolation("r", {"x": 5})
        assert viol.to_dict() == {"name": "r", "counterexample": {"x": 5}}
        verdict = ComplianceVerdict(False, (viol,), 1)
        d = verdict.to_dict()
        assert d["compliant"] is False
        assert d["constraints_checked"] == 1
        assert d["violations"][0]["name"] == "r"

    def test_policy_formaliser_is_runtime_checkable(self) -> None:
        class _Stub:
            def formalise(self, natural_language_policy: str) -> CompliancePolicy:
                return CompliancePolicy([Constraint("a", le(var("x"), lit(1)))])

        assert isinstance(_Stub(), PolicyFormaliser)


def test_engine_reports_actionable_error_when_z3_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The formal engine reports the missing optional dependency before solving."""
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,  # noqa: A002
        locals: dict[str, Any] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "z3":
            raise ImportError("synthetic missing z3")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    with pytest.raises(RuntimeError, match=r"requires z3.*\[formal\] extra"):
        NeuroSymbolicComplianceEngine(
            CompliancePolicy([Constraint("limit", le(var("amount"), lit(100)))])
        )


def _finance_policy() -> CompliancePolicy:
    amount = var("amount", REAL)
    approved = var("manager_approved", BOOL)
    return CompliancePolicy(
        [
            Constraint("amount_limit", le(amount, lit(10000))),
            Constraint("approval_required", implies(gt(amount, lit(5000)), approved)),
        ]
    )


@_needs_z3
class TestEngine:
    def test_engine_exposes_policy(self) -> None:
        policy = _finance_policy()
        assert NeuroSymbolicComplianceEngine(policy).policy is policy

    def test_compile_covers_all_node_kinds(self) -> None:
        # One rich constraint exercising not/and/or, ne/ge, and add/sub/mul.
        x = var("x", INT)
        rich = and_(
            ne(x, lit(0)),
            or_(
                ge(add(sub(x, lit(1)), lit(1)), lit(2)),
                not_(eq(mul(x, lit(2)), lit(10))),
            ),
        )
        engine = NeuroSymbolicComplianceEngine(
            CompliancePolicy([Constraint("rich", rich)])
        )
        assert engine.check({"x": 3}).compliant is True

    def test_compliant_output(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        verdict = engine.check({"amount": 3000, "manager_approved": False})
        assert verdict.compliant is True
        assert verdict.violations == ()
        assert verdict.constraints_checked == 2

    def test_non_compliant_reports_each_violated_rule(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        verdict = engine.check({"amount": 15000, "manager_approved": False})
        assert verdict.compliant is False
        names = {v.name for v in verdict.violations}
        assert names == {"amount_limit", "approval_required"}

    def test_counterexample_carries_the_facts(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        verdict = engine.check({"amount": 15000, "manager_approved": False})
        ce = next(v for v in verdict.violations if v.name == "amount_limit")
        assert ce.counterexample["amount"] == 15000.0

    def test_high_amount_with_approval_is_compliant(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        verdict = engine.check({"amount": 8000, "manager_approved": True})
        assert verdict.compliant is True

    def test_unknown_fact_variable_rejected(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        with pytest.raises(ValueError, match="unknown fact variable"):
            engine.check({"nonsense": 1})

    def test_fact_type_validation(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        with pytest.raises(ValueError, match="must be boolean"):
            engine.check({"manager_approved": 1})
        with pytest.raises(ValueError, match="must be numeric"):
            engine.check({"amount": True})

    def test_integer_and_arithmetic_constraint(self) -> None:
        # 2 * quantity must not exceed the cap of 20.
        quantity = var("quantity", INT)
        policy = CompliancePolicy(
            [Constraint("cap", le(mul(lit(2), quantity), lit(20)))]
        )
        engine = NeuroSymbolicComplianceEngine(policy)
        assert engine.check({"quantity": 8}).compliant is True
        bad = engine.check({"quantity": 11})
        assert bad.compliant is False
        assert bad.violations[0].counterexample["quantity"] == 11

    def test_is_consistent(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        assert engine.is_consistent() is True

    def test_inconsistent_policy_detected(self) -> None:
        x = var("x", REAL)
        policy = CompliancePolicy(
            [Constraint("a", gt(x, lit(5))), Constraint("b", lt(x, lit(3)))]
        )
        assert NeuroSymbolicComplianceEngine(policy).is_consistent() is False

    def test_equivalent_formalisations(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        assert engine.equivalent_to(_finance_policy()) is True

    def test_non_equivalent_formalisations(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        amount = var("amount", REAL)
        weaker = CompliancePolicy([Constraint("limit", le(amount, lit(9999)))])
        assert engine.equivalent_to(weaker) is False

    def test_equivalence_sort_conflict_rejected(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        other = CompliancePolicy([Constraint("c", eq(var("amount", INT), lit(1)))])
        with pytest.raises(ValueError, match="conflicting sorts"):
            engine.equivalent_to(other)

    def test_smt_lib_artefact(self) -> None:
        engine = NeuroSymbolicComplianceEngine(_finance_policy())
        smt = engine.to_smt_lib()
        assert "assert" in smt

    def test_two_stage_via_stub_formaliser(self) -> None:
        # Stage 1: a (stubbed) formaliser turns policy text into constraints.
        class _Formaliser:
            def formalise(self, natural_language_policy: str) -> CompliancePolicy:
                amount = var("amount", REAL)
                return CompliancePolicy([Constraint("limit", le(amount, lit(100)))])

        policy = _Formaliser().formalise("amounts must not exceed 100")
        # Stage 2: check an output against the formalised policy.
        engine = NeuroSymbolicComplianceEngine(policy)
        assert engine.check({"amount": 50}).compliant is True
        assert engine.check({"amount": 150}).compliant is False

    def test_boolean_const_and_negation(self) -> None:
        flag = var("flag", BOOL)
        policy = CompliancePolicy([Constraint("must_be_set", eq(flag, lit(True)))])
        engine = NeuroSymbolicComplianceEngine(policy)
        assert engine.check({"flag": True}).compliant is True
        assert engine.check({"flag": False}).compliant is False


class TestGuardWiring:
    @_needs_z3
    def test_guard_compliance_engine(self) -> None:
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        engine = guard.compliance_engine(_finance_policy())
        assert isinstance(engine, NeuroSymbolicComplianceEngine)
        assert engine.check({"amount": 1000, "manager_approved": False}).compliant
