# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Execution-ring authorisation tests

"""Multi-angle tests for graduated human authorisation of agent actions.

Covers ring classification (per ring, compound highest-wins, fail-closed
default, tokenisation edges), the authorisation-factor reduction (two-person
rule, cooling-window threshold, independent CISO notification), and the gate
decision for each ring including missing-factor reporting and ProductionGuard
wiring.
"""

from __future__ import annotations

import pytest

from director_ai.core.execution_rings import (
    RING_REQUIRED_FACTORS,
    AuthorizationEvidence,
    AuthorizationFactor,
    ExecutionRing,
    ExecutionRingGate,
    RingDecision,
    classify_operation,
    satisfied_factors,
)


class TestClassification:
    @pytest.mark.parametrize(
        ("operation", "ring"),
        [
            ("read user record", ExecutionRing.READ),
            ("list files", ExecutionRing.READ),
            ("update the profile", ExecutionRing.WRITE),
            ("create invoice", ExecutionRing.WRITE),
            ("delete account", ExecutionRing.DELETE),
            ("drop table users", ExecutionRing.DELETE),
            ("run shell script", ExecutionRing.EXECUTE),
            ("invoke the tool", ExecutionRing.EXECUTE),
            ("export all data", ExecutionRing.EXFILTRATE),
            ("email the report", ExecutionRing.EXFILTRATE),
        ],
    )
    def test_each_ring(self, operation, ring):
        assert classify_operation(operation) == ring

    def test_compound_takes_highest_ring(self):
        # contains both 'update' (write) and 'export' (exfiltrate) -> exfiltrate.
        assert classify_operation("update then export") == ExecutionRing.EXFILTRATE

    def test_unknown_operation_fails_closed_to_execute(self):
        assert classify_operation("frobnicate widget") == ExecutionRing.EXECUTE

    def test_punctuation_tokenisation(self):
        assert classify_operation("delete,account!") == ExecutionRing.DELETE

    def test_empty_operation_defaults_execute(self):
        assert classify_operation("") == ExecutionRing.EXECUTE

    def test_leading_and_repeated_separators(self):
        # Leading/duplicated separators exercise the empty-buffer tokenise path.
        assert classify_operation("   delete   account  ") == ExecutionRing.DELETE
        assert classify_operation("!!!") == ExecutionRing.EXECUTE

    def test_ring_ordering(self):
        assert (
            ExecutionRing.READ
            < ExecutionRing.WRITE
            < ExecutionRing.DELETE
            < ExecutionRing.EXECUTE
            < ExecutionRing.EXFILTRATE
        )

    def test_required_factor_sets_are_cumulative(self):
        for lower, higher in (
            (ExecutionRing.READ, ExecutionRing.WRITE),
            (ExecutionRing.WRITE, ExecutionRing.DELETE),
            (ExecutionRing.DELETE, ExecutionRing.EXECUTE),
            (ExecutionRing.EXECUTE, ExecutionRing.EXFILTRATE),
        ):
            assert RING_REQUIRED_FACTORS[lower] < RING_REQUIRED_FACTORS[higher]


class TestAuthorizationEvidence:
    def test_negative_cooling_rejected(self):
        with pytest.raises(ValueError, match="cooling_elapsed_seconds"):
            AuthorizationEvidence(cooling_elapsed_seconds=-1.0)

    def test_default_is_empty(self):
        held = satisfied_factors(
            AuthorizationEvidence(), cooling_period_seconds=86_400.0
        )
        assert held == frozenset()


class TestSatisfiedFactors:
    def test_operator_approval(self):
        held = satisfied_factors(
            AuthorizationEvidence(operator_approval=True),
            cooling_period_seconds=86_400.0,
        )
        assert held == frozenset({AuthorizationFactor.OPERATOR_APPROVAL})

    def test_cooling_only_counts_with_approval_and_elapsed(self):
        # Elapsed but not approved -> nothing (clock has no meaning without approval).
        assert (
            satisfied_factors(
                AuthorizationEvidence(cooling_elapsed_seconds=99_999.0),
                cooling_period_seconds=86_400.0,
            )
            == frozenset()
        )
        held = satisfied_factors(
            AuthorizationEvidence(
                operator_approval=True, cooling_elapsed_seconds=86_400.0
            ),
            cooling_period_seconds=86_400.0,
        )
        assert AuthorizationFactor.COOLING_PERIOD in held

    def test_cooling_below_window_not_counted(self):
        held = satisfied_factors(
            AuthorizationEvidence(operator_approval=True, cooling_elapsed_seconds=10.0),
            cooling_period_seconds=86_400.0,
        )
        assert AuthorizationFactor.COOLING_PERIOD not in held

    def test_two_person_rule_requires_first_operator(self):
        # Second approver without a first counts for nothing.
        assert (
            satisfied_factors(
                AuthorizationEvidence(second_operator_approval=True),
                cooling_period_seconds=86_400.0,
            )
            == frozenset()
        )
        held = satisfied_factors(
            AuthorizationEvidence(
                operator_approval=True, second_operator_approval=True
            ),
            cooling_period_seconds=86_400.0,
        )
        assert AuthorizationFactor.SECOND_OPERATOR in held

    def test_ciso_notification_is_independent(self):
        held = satisfied_factors(
            AuthorizationEvidence(ciso_notification=True),
            cooling_period_seconds=86_400.0,
        )
        assert held == frozenset({AuthorizationFactor.CISO_NOTIFICATION})


class TestGate:
    def test_negative_cooling_period_rejected(self):
        with pytest.raises(ValueError, match="cooling_period_seconds"):
            ExecutionRingGate(cooling_period_seconds=-1.0)

    def test_cooling_period_property(self):
        assert (
            ExecutionRingGate(cooling_period_seconds=60.0).cooling_period_seconds
            == 60.0
        )

    def test_read_always_allowed(self):
        decision = ExecutionRingGate().evaluate(ExecutionRing.READ)
        assert decision.allowed is True
        assert decision.missing == frozenset()

    def test_write_needs_approval(self):
        gate = ExecutionRingGate()
        assert gate.evaluate(ExecutionRing.WRITE).allowed is False
        ok = gate.evaluate(
            ExecutionRing.WRITE, AuthorizationEvidence(operator_approval=True)
        )
        assert ok.allowed is True

    def test_delete_waits_for_cooling(self):
        gate = ExecutionRingGate(cooling_period_seconds=100.0)
        early = gate.evaluate(
            ExecutionRing.DELETE,
            AuthorizationEvidence(operator_approval=True, cooling_elapsed_seconds=50.0),
        )
        assert early.allowed is False
        assert AuthorizationFactor.COOLING_PERIOD in early.missing
        late = gate.evaluate(
            ExecutionRing.DELETE,
            AuthorizationEvidence(
                operator_approval=True, cooling_elapsed_seconds=150.0
            ),
        )
        assert late.allowed is True

    def test_exfiltrate_needs_all_four(self):
        gate = ExecutionRingGate(cooling_period_seconds=10.0)
        evidence = AuthorizationEvidence(
            operator_approval=True,
            second_operator_approval=True,
            ciso_notification=True,
            cooling_elapsed_seconds=20.0,
        )
        decision = gate.evaluate(ExecutionRing.EXFILTRATE, evidence)
        assert decision.allowed is True
        assert decision.missing == frozenset()
        assert decision.satisfied == RING_REQUIRED_FACTORS[ExecutionRing.EXFILTRATE]

    def test_satisfied_is_clipped_to_required(self):
        # Extra CISO notification on a WRITE is not reported as a required match.
        gate = ExecutionRingGate()
        decision = gate.evaluate(
            ExecutionRing.WRITE,
            AuthorizationEvidence(operator_approval=True, ciso_notification=True),
        )
        assert decision.satisfied == frozenset({AuthorizationFactor.OPERATOR_APPROVAL})

    def test_authorize_classifies_then_evaluates(self):
        gate = ExecutionRingGate(cooling_period_seconds=10.0)
        decision = gate.authorize(
            "export the dataset",
            AuthorizationEvidence(
                operator_approval=True,
                second_operator_approval=True,
                ciso_notification=True,
                cooling_elapsed_seconds=20.0,
            ),
        )
        assert decision.ring == ExecutionRing.EXFILTRATE
        assert decision.allowed is True

    def test_evaluate_with_no_evidence_uses_empty(self):
        decision = ExecutionRingGate().evaluate(ExecutionRing.WRITE)
        assert decision.allowed is False


class TestRingDecisionSerialisation:
    def test_to_dict_is_tenant_safe(self):
        decision = ExecutionRingGate().evaluate(
            ExecutionRing.DELETE, AuthorizationEvidence(operator_approval=True)
        )
        d = decision.to_dict()
        assert set(d) == {"ring", "allowed", "required", "satisfied", "missing"}
        assert d["ring"] == "delete"
        assert d["missing"] == ["cooling_period"]

    def test_default_missing_is_empty(self):
        decision = RingDecision(
            ring=ExecutionRing.READ,
            allowed=True,
            required=frozenset(),
            satisfied=frozenset(),
        )
        assert decision.missing == frozenset()


class TestGuardWiring:
    def test_production_guard_exposes_execution_rings(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        gate = guard.execution_rings()
        assert isinstance(gate, ExecutionRingGate)
        # Cached across calls.
        assert guard.execution_rings() is gate
        decision = gate.authorize(
            "delete record", AuthorizationEvidence(operator_approval=True)
        )
        assert decision.ring == ExecutionRing.DELETE
        assert decision.allowed is False  # cooling period still outstanding
