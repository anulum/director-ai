# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Consilium Unit Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Unit tests for research/consilium/director_core.py:
EthicalFunctional, SystemState, ConsiliumAgent.
"""

import pytest

from director_ai.research.consilium import (
    ConsiliumAgent,
    EthicalFunctional,
    SystemState,
)


class TestSystemState:
    def test_dataclass_fields(self):
        state = SystemState(
            error_count=0,
            test_failure_count=0,
            code_complexity_score=5.0,
            knowledge_graph_density=0.8,
            code_coverage_percent=85.0,
            rag_concept_entropy=1.2,
        )
        assert state.error_count == 0
        assert state.code_coverage_percent == 85.0

    def test_default_timestamp(self):
        state = SystemState(
            error_count=1,
            test_failure_count=2,
            code_complexity_score=3.0,
            knowledge_graph_density=0.5,
            code_coverage_percent=60.0,
            rag_concept_entropy=2.0,
        )
        assert state.timestamp is not None


class TestEthicalFunctional:
    @pytest.fixture
    def functional(self):
        return EthicalFunctional()

    def test_evaluate_returns_float(self, functional):
        state = SystemState(
            error_count=5,
            test_failure_count=2,
            code_complexity_score=8.0,
            knowledge_graph_density=0.7,
            code_coverage_percent=75.0,
            rag_concept_entropy=1.5,
        )
        result = functional.evaluate(state)
        assert isinstance(result, float)

    def test_perfect_state_low_energy(self, functional):
        """A near-perfect system state should have low ethical energy."""
        state = SystemState(
            error_count=0,
            test_failure_count=0,
            code_complexity_score=1.0,
            knowledge_graph_density=0.95,
            code_coverage_percent=99.0,
            rag_concept_entropy=0.1,
        )
        energy = functional.evaluate(state)
        assert energy < 50.0

    def test_bad_state_high_energy(self, functional):
        """A degraded system state should have higher ethical energy."""
        state = SystemState(
            error_count=100,
            test_failure_count=50,
            code_complexity_score=20.0,
            knowledge_graph_density=0.1,
            code_coverage_percent=10.0,
            rag_concept_entropy=5.0,
        )
        energy = functional.evaluate(state)
        # Energy should be noticeably higher than the perfect case
        assert energy > 50.0

    def test_energy_monotonicity_with_errors(self, functional):
        """More errors → higher ethical energy."""
        base = dict(
            test_failure_count=0,
            code_complexity_score=5.0,
            knowledge_graph_density=0.7,
            code_coverage_percent=80.0,
            rag_concept_entropy=1.0,
        )
        e_low = functional.evaluate(SystemState(error_count=0, **base))
        e_high = functional.evaluate(SystemState(error_count=50, **base))
        assert e_high > e_low


class TestConsiliumAgent:
    def test_instantiation(self):
        agent = ConsiliumAgent()
        assert agent is not None

    def test_decide_returns_string(self):
        agent = ConsiliumAgent()
        decision = agent.decide()
        assert isinstance(decision, str)
        assert len(decision) > 0

    def test_decide_is_known_action(self):
        agent = ConsiliumAgent()
        decision = agent.decide()
        known_actions = [
            "REFACTOR_CORE",
            "EXPAND_KNOWLEDGE",
            "STABILIZE_TESTS",
            "DO_NOTHING",
        ]
        assert decision in known_actions
