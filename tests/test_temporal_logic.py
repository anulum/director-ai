# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Temporal Logic Monitor Tests
"""Multi-angle tests for LTL formula algebra, progression, and agent specs."""

from __future__ import annotations

import pytest

from director_ai.core.temporal_logic import (
    BOTTOM,
    TOP,
    F,
    G,
    LTLMonitor,
    StepObservation,
    TrajectorySafetyMonitor,
    U,
    Verdict,
    X,
    and_,
    atom,
    implies,
    not_,
    or_,
    progress,
    value_at_end,
)
from director_ai.core.temporal_logic.formula import Always, Atom


class TestFormulaAlgebra:
    def test_and_absorbs_bottom(self):
        assert and_(atom("a"), BOTTOM) is BOTTOM

    def test_and_drops_top(self):
        assert and_(atom("a"), TOP) == atom("a")

    def test_and_empty_is_top(self):
        assert and_() is TOP

    def test_or_absorbs_top(self):
        assert or_(atom("a"), TOP) is TOP

    def test_or_drops_bottom(self):
        assert or_(atom("a"), BOTTOM) == atom("a")

    def test_or_empty_is_bottom(self):
        assert or_() is BOTTOM

    def test_and_flattens_and_dedups(self):
        # Nested And with a duplicate operand canonicalises to two operands.
        formula = and_(and_(atom("a"), atom("b")), atom("a"))
        assert isinstance(formula, type(and_(atom("a"), atom("b"))))
        assert formula == and_(atom("a"), atom("b"))

    def test_not_double_negation(self):
        assert not_(not_(atom("a"))) == atom("a")

    def test_not_of_constants(self):
        assert not_(TOP) is BOTTOM
        assert not_(BOTTOM) is TOP

    def test_implies_desugars_to_or(self):
        assert implies(atom("a"), atom("b")) == or_(not_(atom("a")), atom("b"))

    def test_g_of_constants(self):
        assert G(TOP) is TOP
        assert G(BOTTOM) is BOTTOM

    def test_f_of_constants(self):
        assert F(TOP) is TOP
        assert F(BOTTOM) is BOTTOM

    def test_u_with_true_right_is_top(self):
        assert U(atom("a"), TOP) is TOP

    def test_atom_requires_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            atom("")

    def test_str_renderings(self):
        assert str(TOP) == "⊤"
        assert str(BOTTOM) == "⊥"
        assert str(atom("p")) == "p"
        assert str(G(atom("p"))) == "G p"
        assert str(F(atom("p"))) == "F p"
        assert str(X(atom("p"))) == "X p"
        assert str(not_(atom("p"))) == "¬p"
        assert str(U(atom("p"), atom("q"))) == "(p U q)"
        assert " ∧ " in str(and_(atom("p"), atom("q")))
        assert " ∨ " in str(or_(atom("p"), atom("q")))


class TestProgression:
    def test_atom_true_and_false(self):
        assert progress(atom("a"), frozenset({"a"})) is TOP
        assert progress(atom("a"), frozenset()) is BOTTOM

    def test_constants_unchanged(self):
        assert progress(TOP, frozenset()) is TOP
        assert progress(BOTTOM, frozenset()) is BOTTOM

    def test_next_defers_operand(self):
        assert progress(X(atom("a")), frozenset()) == atom("a")

    def test_always_holds_then_keeps_obligation(self):
        # G a after a holds collapses back to G a (bounded residual).
        residual = progress(G(atom("a")), frozenset({"a"}))
        assert residual == G(atom("a"))

    def test_always_violated_when_operand_fails(self):
        assert progress(G(atom("a")), frozenset()) is BOTTOM

    def test_eventually_satisfied_now(self):
        assert progress(F(atom("a")), frozenset({"a"})) is TOP

    def test_eventually_pending(self):
        assert progress(F(atom("a")), frozenset()) == F(atom("a"))

    def test_until_right_satisfies(self):
        assert progress(U(atom("a"), atom("b")), frozenset({"b"})) is TOP

    def test_until_left_holds_keeps_obligation(self):
        residual = progress(U(atom("a"), atom("b")), frozenset({"a"}))
        assert residual == U(atom("a"), atom("b"))

    def test_until_violated_when_neither_holds(self):
        assert progress(U(atom("a"), atom("b")), frozenset()) is BOTTOM

    def test_residual_stays_bounded_over_many_steps(self):
        formula = G(implies(atom("a"), F(atom("b"))))
        residual = formula
        for _ in range(200):
            residual = progress(residual, frozenset({"b"}))  # no trigger, b free
        # Canonicalisation keeps the residual from accumulating per-step conjuncts.
        assert len(str(residual)) < 80


class TestValueAtEnd:
    def test_constants(self):
        assert value_at_end(TOP) is True
        assert value_at_end(BOTTOM) is False

    def test_always_vacuously_true(self):
        assert value_at_end(G(atom("a"))) is True

    def test_eventually_unmet_is_false(self):
        assert value_at_end(F(atom("a"))) is False

    def test_until_unmet_is_false(self):
        assert value_at_end(U(atom("a"), atom("b"))) is False

    def test_next_unmet_is_false(self):
        assert value_at_end(X(atom("a"))) is False

    def test_bare_atom_is_false(self):
        assert value_at_end(atom("a")) is False

    def test_boolean_composition(self):
        assert value_at_end(or_(G(atom("a")), F(atom("b")))) is True
        assert value_at_end(and_(G(atom("a")), F(atom("b")))) is False
        assert value_at_end(not_(F(atom("a")))) is True


class TestLTLMonitor:
    def test_safety_violation_latches(self):
        mon = LTLMonitor(G(atom("safe")))
        assert mon.push({"safe"}) is Verdict.INCONCLUSIVE
        assert mon.push(set()) is Verdict.VIOLATED
        # Latched: a later good state does not flip it back.
        assert mon.push({"safe"}) is Verdict.VIOLATED
        assert mon.is_definitive is True

    def test_eventually_satisfied_latches(self):
        mon = LTLMonitor(F(atom("done")))
        assert mon.push(set()) is Verdict.INCONCLUSIVE
        assert mon.push({"done"}) is Verdict.SATISFIED
        assert mon.is_definitive is True

    def test_finalize_pending_eventually_is_violation(self):
        mon = LTLMonitor(F(atom("done")))
        mon.push(set())
        assert mon.finalize() is Verdict.VIOLATED

    def test_finalize_clean_always_is_satisfied(self):
        mon = LTLMonitor(G(atom("safe")))
        mon.push({"safe"})
        mon.push({"safe"})
        assert mon.finalize() is Verdict.SATISFIED

    def test_steps_and_initial(self):
        mon = LTLMonitor(G(atom("a")), name="spec")
        mon.push({"a"})
        mon.push({"a"})
        assert mon.steps == 2
        assert mon.name == "spec"
        assert str(mon.initial) == "G a"

    def test_finalize_returns_definitive_unchanged(self):
        mon = LTLMonitor(G(atom("a")))
        mon.push(set())  # violated
        assert mon.finalize() is Verdict.VIOLATED


class TestStepObservation:
    def test_propositions_only_true_flags(self):
        obs = StepObservation(tool_call=True, evidence_retrieved=True)
        assert obs.propositions() == frozenset({"tool_call", "evidence_retrieved"})

    def test_empty_observation(self):
        assert StepObservation().propositions() == frozenset()


class TestTrajectorySafetyMonitor:
    def test_injection_then_output_violates(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(injection_detected=True, output_emitted=True))
        report = mon.report()
        assert report["verdict"] == "violated"
        assert "no_output_after_injection" in report["violations"]

    def test_injection_without_output_is_safe(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(injection_detected=True))
        mon.observe(StepObservation(output_emitted=True))
        report = mon.finalize()
        assert "no_output_after_injection" not in report["violations"]

    def test_tool_call_without_verification_violates_on_finalize(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(tool_call=True))
        mon.observe(StepObservation())
        assert mon.report()["verdict"] == "inconclusive"
        final = mon.finalize()
        assert "tool_calls_are_verified" in final["violations"]

    def test_tool_call_then_verified_is_satisfied(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(tool_call=True))
        mon.observe(StepObservation(verification_passed=True))
        assert mon.finalize()["verdict"] == "satisfied"

    def test_handoff_next_step_coherence_check_satisfied(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(handoff=True))
        mon.observe(StepObservation(coherence_check=True))
        final = mon.finalize()
        assert "handoff_is_coherence_checked" not in final["violations"]

    def test_handoff_without_next_coherence_check_violates(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(handoff=True))
        mon.observe(StepObservation())  # no coherence_check in the next step
        assert "handoff_is_coherence_checked" in mon.report()["violations"]

    def test_fact_claim_eventually_grounded(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(fact_claim=True))
        mon.observe(StepObservation())
        mon.observe(StepObservation(evidence_retrieved=True))
        assert "fact_claims_are_grounded" not in mon.finalize()["violations"]

    def test_violated_at_step_recorded(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation())
        mon.observe(StepObservation(injection_detected=True, output_emitted=True))
        report = mon.report()
        status = next(
            s for s in report["specs"] if s["name"] == "no_output_after_injection"
        )
        assert status["violated_at_step"] == 2

    def test_report_is_tenant_safe_structure(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(tool_call=True))
        report = mon.report()
        assert set(report) == {"verdict", "steps", "specs", "violations"}
        for status in report["specs"]:
            assert set(status) == {
                "name",
                "verdict",
                "formula",
                "violated_at_step",
            }

    def test_observe_accepts_raw_proposition_set(self):
        mon = TrajectorySafetyMonitor()
        mon.observe({"injection_detected", "output_emitted"})
        assert mon.report()["verdict"] == "violated"

    def test_overall_verdict_precedence(self):
        # One pending (inconclusive) + one violated -> overall violated.
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation(tool_call=True))  # pending liveness
        mon.observe(StepObservation(injection_detected=True, output_emitted=True))
        assert mon.report()["verdict"] == "violated"

    def test_custom_specs(self):
        specs = {"never_p": G(not_(atom("p")))}
        mon = TrajectorySafetyMonitor(specs)
        mon.observe({"p"})
        assert mon.report()["violations"] == ["never_p"]

    def test_empty_specs_rejected(self):
        with pytest.raises(ValueError, match="at least one specification"):
            TrajectorySafetyMonitor({})

    def test_steps_counter(self):
        mon = TrajectorySafetyMonitor()
        mon.observe(StepObservation())
        mon.observe(StepObservation())
        assert mon.steps == 2


class TestProductionGuardWiring:
    def test_guard_exposes_trajectory_monitor(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        mon = guard.trajectory_monitor()
        mon.observe(StepObservation(injection_detected=True, output_emitted=True))
        assert mon.report()["verdict"] == "violated"
        # Each call returns an independent monitor (no shared state).
        assert guard.trajectory_monitor() is not mon


def test_always_node_is_used_in_default_specs():
    # The built-in specs are all safety envelopes (Always-rooted).
    from director_ai.core.temporal_logic import default_agent_safety_specs

    for formula in default_agent_safety_specs().values():
        assert isinstance(formula, Always)


def test_atom_type_exposed():
    assert isinstance(atom("x"), Atom)
