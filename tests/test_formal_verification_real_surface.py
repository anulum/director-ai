# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - formal verification real-surface tests
"""Real public-surface coverage for formal verification model reporting."""

from __future__ import annotations

from director_ai.core.formal_verification import (
    DpllSolver,
    Literal,
    Or,
    ReasoningStep,
    ReasoningVerifier,
    Variable,
)


def test_dpll_solver_sat_solution_reports_total_model() -> None:
    """SAT solutions should assign every variable present in public clauses."""
    clauses = ((Literal("approved", True), Literal("manual_review", True)),)

    solution = DpllSolver().solve(clauses)

    assert solution.satisfiable is True
    assert set(solution.model) == {"approved", "manual_review"}
    assert any(solution.model.values())


def test_reasoning_verifier_sat_verdict_reports_total_model() -> None:
    """Verifier verdicts should expose all variables from public formulas."""
    verifier = ReasoningVerifier()

    verdict = verifier.verify(
        (
            ReasoningStep(
                label="policy-branch",
                formula=Or(Variable("approved"), Variable("manual_review")),
            ),
        )
    )

    assert verdict.consistent is True
    assert verdict.backend == "dpll"
    assert verdict.step_count == 1
    assert set(verdict.model) == {"approved", "manual_review"}
    assert any(verdict.model.values())
