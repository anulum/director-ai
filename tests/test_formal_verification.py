# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — formal verification tests

"""Multi-angle coverage: AST construction, CNF conversion
(iff/implication elimination, De Morgan, distribution), DPLL
solver correctness on SAT and UNSAT instances, unit-propagation
and pure-literal shortcuts, branch-limit protection, reasoning
verifier end-to-end, and protocol compliance for Z3 and Lean
backend adapters."""

from __future__ import annotations

import importlib.util

import pytest

from director_ai.core.formal_verification import (
    And,
    CnfConverter,
    DpllSolver,
    Iff,
    Implies,
    LeanBackend,
    Literal,
    Not,
    Or,
    ReasoningStep,
    ReasoningVerdict,
    ReasoningVerifier,
    Solution,
    Variable,
    VerifierBackend,
    Z3Backend,
)
from director_ai.core.formal_verification.formula import variables

# --- Formula AST ---------------------------------------------------


class TestFormula:
    def test_variable_rejects_empty(self):
        with pytest.raises(ValueError, match="Variable"):
            Variable(name="")

    def test_str_representation(self):
        f = Implies(Variable("p"), Variable("q"))
        assert "→" in str(f)

    def test_variables_collects_all(self):
        f = And(
            Or(Variable("p"), Not(Variable("q"))), Iff(Variable("r"), Variable("s"))
        )
        assert variables(f) == frozenset({"p", "q", "r", "s"})


# --- CnfConverter --------------------------------------------------


def _names(clauses: tuple[tuple[Literal, ...], ...]) -> list[set[tuple[str, bool]]]:
    return [{(lit.name, lit.positive) for lit in clause} for clause in clauses]


class TestCnf:
    def test_variable_becomes_singleton(self):
        clauses = CnfConverter().convert(Variable("p"))
        assert _names(clauses) == [{("p", True)}]

    def test_implication_eliminated(self):
        """p → q becomes ¬p ∨ q."""
        clauses = CnfConverter().convert(Implies(Variable("p"), Variable("q")))
        assert _names(clauses) == [{("p", False), ("q", True)}]

    def test_iff_expansion(self):
        """p ↔ q becomes (p → q) ∧ (q → p)."""
        clauses = CnfConverter().convert(Iff(Variable("p"), Variable("q")))
        clause_sets = _names(clauses)
        assert {("p", False), ("q", True)} in clause_sets
        assert {("q", False), ("p", True)} in clause_sets

    def test_de_morgan_or(self):
        """¬(p ∨ q) becomes ¬p ∧ ¬q."""
        clauses = CnfConverter().convert(Not(Or(Variable("p"), Variable("q"))))
        clause_sets = _names(clauses)
        assert {("p", False)} in clause_sets
        assert {("q", False)} in clause_sets

    def test_de_morgan_and(self):
        """¬(p ∧ q) becomes ¬p ∨ ¬q."""
        clauses = CnfConverter().convert(Not(And(Variable("p"), Variable("q"))))
        assert _names(clauses) == [{("p", False), ("q", False)}]

    def test_double_negation(self):
        clauses = CnfConverter().convert(Not(Not(Variable("p"))))
        assert _names(clauses) == [{("p", True)}]

    def test_distribution(self):
        """p ∨ (q ∧ r) becomes (p ∨ q) ∧ (p ∨ r)."""
        clauses = CnfConverter().convert(
            Or(Variable("p"), And(Variable("q"), Variable("r")))
        )
        clause_sets = _names(clauses)
        assert {("p", True), ("q", True)} in clause_sets
        assert {("p", True), ("r", True)} in clause_sets


# --- DpllSolver ----------------------------------------------------


class TestDpll:
    def test_empty_clause_set_is_sat(self):
        result = DpllSolver().solve(())
        assert result.satisfiable
        assert result.model == {}

    def test_single_unit_clause(self):
        clauses = ((Literal(name="p", positive=True),),)
        result = DpllSolver().solve(clauses)
        assert result.satisfiable
        assert result.model == {"p": True}

    def test_trivial_contradiction(self):
        clauses = (
            (Literal(name="p", positive=True),),
            (Literal(name="p", positive=False),),
        )
        result = DpllSolver().solve(clauses)
        assert not result.satisfiable

    def test_unit_propagation_counts(self):
        """(p) ∧ (¬p ∨ q) — should propagate p=True, then q=True."""
        clauses = (
            (Literal("p", True),),
            (Literal("p", False), Literal("q", True)),
        )
        result = DpllSolver().solve(clauses)
        assert result.satisfiable
        assert result.model == {"p": True, "q": True}
        assert result.propagations >= 1

    def test_pure_literal(self):
        """p appears only positive — assign it True without branching."""
        clauses = (
            (Literal("p", True), Literal("q", True)),
            (Literal("p", True), Literal("r", False)),
        )
        result = DpllSolver().solve(clauses)
        assert result.satisfiable
        assert result.model.get("p") is True

    def test_multi_branch_sat(self):
        """(p ∨ q) ∧ (¬p ∨ q) — q=True satisfies both."""
        clauses = (
            (Literal("p", True), Literal("q", True)),
            (Literal("p", False), Literal("q", True)),
        )
        result = DpllSolver().solve(clauses)
        assert result.satisfiable
        assert result.model.get("q") is True

    def test_pigeonhole_unsat(self):
        """Encode a tiny pigeonhole principle (2 pigeons, 1 hole) —
        UNSAT: at least one pigeon must sit without a hole."""
        # P_ij means pigeon i sits in hole j. Two pigeons, one hole
        # (hole=0). Constraints:
        #   at-least-one: P_10 ∨ (nothing; only one hole) → P_10 must be true.
        #   at-least-one: P_20 → P_20 must be true.
        #   at-most-one-per-hole: ¬(P_10 ∧ P_20) = ¬P_10 ∨ ¬P_20.
        clauses = (
            (Literal("P10", True),),
            (Literal("P20", True),),
            (Literal("P10", False), Literal("P20", False)),
        )
        result = DpllSolver().solve(clauses)
        assert not result.satisfiable

    def test_decision_cap(self):
        """Pigeonhole PHP(n+1, n) is exponentially hard for basic
        DPLL (no CDCL). PHP(5, 4) is a reliable trigger at a
        tiny max_decisions budget."""
        solver = DpllSolver(max_decisions=2)

        def pigeonhole(n_pigeons: int, n_holes: int) -> tuple[tuple[Literal, ...], ...]:
            clauses: list[tuple[Literal, ...]] = []
            # At least one hole per pigeon.
            for pigeon in range(n_pigeons):
                clauses.append(
                    tuple(
                        Literal(name=f"P{pigeon}_{hole}", positive=True)
                        for hole in range(n_holes)
                    )
                )
            # At most one pigeon per hole.
            for hole in range(n_holes):
                for a in range(n_pigeons):
                    for b in range(a + 1, n_pigeons):
                        clauses.append(
                            (
                                Literal(name=f"P{a}_{hole}", positive=False),
                                Literal(name=f"P{b}_{hole}", positive=False),
                            )
                        )
            return tuple(clauses)

        with pytest.raises(TimeoutError):
            solver.solve(pigeonhole(5, 4))

    def test_bad_max_decisions(self):
        with pytest.raises(ValueError, match="max_decisions"):
            DpllSolver(max_decisions=0)

    def test_solution_dataclass(self):
        sol = Solution(satisfiable=True, model={"p": True})
        assert sol.satisfiable
        assert sol.decisions == 0


# --- ReasoningVerifier --------------------------------------------


class TestReasoningVerifier:
    def test_consistent_chain(self):
        verifier = ReasoningVerifier()
        # (p → q), p, and ¬¬q are all simultaneously satisfiable.
        steps = [
            ReasoningStep(
                label="premise-1", formula=Implies(Variable("p"), Variable("q"))
            ),
            ReasoningStep(label="premise-2", formula=Variable("p")),
            ReasoningStep(
                label="conclusion",
                formula=Not(Not(Variable("q"))),
            ),
        ]
        verdict = verifier.verify(steps)
        assert isinstance(verdict, ReasoningVerdict)
        assert verdict.consistent
        assert verdict.backend == "dpll"
        assert verdict.step_count == 3

    def test_contradictory_chain(self):
        verifier = ReasoningVerifier()
        steps = [
            ReasoningStep(label="premise", formula=Variable("p")),
            ReasoningStep(label="negation", formula=Not(Variable("p"))),
        ]
        verdict = verifier.verify(steps)
        assert verdict.contradictory

    def test_empty_steps_rejected(self):
        verifier = ReasoningVerifier()
        with pytest.raises(ValueError, match="non-empty"):
            verifier.verify([])

    def test_step_label_required(self):
        with pytest.raises(ValueError, match="label"):
            ReasoningStep(label="", formula=Variable("p"))

    def test_custom_backend_plugs_in(self):
        class _AlwaysSat:
            name = "always-sat"

            def solve(self, formula) -> Solution:
                return Solution(satisfiable=True, model={"marker": True})

        backend = _AlwaysSat()
        assert isinstance(backend, VerifierBackend)
        verifier = ReasoningVerifier(backend=backend)
        verdict = verifier.verify([ReasoningStep(label="x", formula=Variable("p"))])
        assert verdict.backend == "always-sat"
        assert verdict.consistent
        assert verdict.model == {"marker": True}


# --- External backend adapters ------------------------------------


class TestZ3Backend:
    def test_missing_runner_rejected(self):
        with pytest.raises(ValueError, match="z3_solver"):
            Z3Backend(z3_solver=None)

    def test_from_z3_without_package(self):
        if importlib.util.find_spec("z3") is not None:
            pytest.skip("z3 installed — ImportError branch cannot fire")
        with pytest.raises(ImportError, match="formal"):
            Z3Backend.from_z3()


class TestLeanBackend:
    def test_runner_required(self):
        with pytest.raises(ValueError, match="runner"):
            LeanBackend(runner=None)

    def test_runner_must_be_callable(self):
        with pytest.raises(ValueError, match="callable"):
            LeanBackend(runner=42)

    def test_runner_returns_solution(self):
        def fake_runner(source: str) -> dict:
            assert "def target" in source
            return {"sat": True, "model": {"p": 1, "q": 0}}

        backend = LeanBackend(runner=fake_runner)
        solution = backend.solve(And(Variable("p"), Not(Variable("q"))))
        assert solution.satisfiable
        assert solution.model == {"p": True, "q": False}

    def test_runner_must_return_dict(self):
        def bad_runner(source: str):
            return "bogus"

        backend = LeanBackend(runner=bad_runner)
        with pytest.raises(ValueError, match="dict"):
            backend.solve(Variable("p"))

    def test_runner_model_must_be_dict(self):
        def bad_model(source: str):
            return {"sat": True, "model": "not-a-dict"}

        backend = LeanBackend(runner=bad_model)
        with pytest.raises(ValueError, match="non-dict model"):
            backend.solve(Variable("p"))

    def test_lean_backend_is_verifier(self):
        def fake(source: str):
            return {"sat": True, "model": {}}

        backend = LeanBackend(runner=fake)
        assert isinstance(backend, VerifierBackend)
