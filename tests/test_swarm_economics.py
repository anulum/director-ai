# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — swarm economics tests

"""Multi-angle coverage: ResourcePool regeneration + ledger,
AgentEconomicState validation, NashBargainingSolver on symmetric
and asymmetric valuations, fairness_gap semantics,
TragedyDetector grace-window behaviour, EconomicRiskScorer
composite under hot/cold scenarios."""

from __future__ import annotations

import contextlib
import threading

import pytest

from director_ai.core.swarm_economics import (
    AgentEconomicState,
    BargainingSolution,
    ConsumptionRecord,
    DisagreementPoint,
    EconomicRiskScorer,
    EconomicVerdict,
    NashBargainingSolver,
    PoolError,
    ResourcePool,
    TragedyDetector,
    TragedySignal,
)

# --- ResourcePool --------------------------------------------------


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


class TestResourcePool:
    def test_initial_balance_defaults_to_capacity(self):
        pool = ResourcePool(capacity=100.0, clock=_FakeClock())
        assert pool.balance() == 100.0

    def test_consume_decreases_balance(self):
        pool = ResourcePool(capacity=100.0, clock=_FakeClock())
        record = pool.consume(agent_id="a", amount=25.0)
        assert pool.balance() == 75.0
        assert isinstance(record, ConsumptionRecord)
        assert record.agent_id == "a"

    def test_insufficient_balance_raises(self):
        pool = ResourcePool(capacity=10.0, clock=_FakeClock())
        with pytest.raises(PoolError, match="insufficient"):
            pool.consume(agent_id="a", amount=25.0)

    def test_regeneration(self):
        clock = _FakeClock(start=0.0)
        pool = ResourcePool(
            capacity=100.0,
            regeneration_rate=5.0,
            initial_balance=50.0,
            clock=clock,
        )
        clock.now = 4.0
        # 4 seconds × 5 units/s = +20 → 70
        assert pool.balance() == 70.0

    def test_regeneration_capped_at_capacity(self):
        clock = _FakeClock(start=0.0)
        pool = ResourcePool(
            capacity=100.0,
            regeneration_rate=50.0,
            initial_balance=50.0,
            clock=clock,
        )
        clock.now = 100.0
        assert pool.balance() == 100.0

    def test_zero_regeneration_stays_fixed(self):
        clock = _FakeClock(start=0.0)
        pool = ResourcePool(capacity=100.0, initial_balance=50.0, clock=clock)
        clock.now = 1e6
        assert pool.balance() == 50.0

    def test_ledger_records_consumption(self):
        pool = ResourcePool(capacity=100.0, clock=_FakeClock())
        pool.consume(agent_id="a", amount=10.0)
        pool.consume(agent_id="b", amount=5.0)
        assert len(pool.ledger()) == 2

    def test_recent_window(self):
        clock = _FakeClock(start=100.0)
        pool = ResourcePool(capacity=100.0, clock=clock)
        pool.consume(agent_id="a", amount=5.0)
        clock.now = 160.0
        pool.consume(agent_id="b", amount=5.0)
        clock.now = 170.0
        recent = pool.recent(since_seconds=30.0)
        assert len(recent) == 1
        assert recent[0].agent_id == "b"

    def test_recent_bad_window(self):
        pool = ResourcePool(capacity=100.0, clock=_FakeClock())
        with pytest.raises(PoolError, match="since_seconds"):
            pool.recent(since_seconds=0)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"capacity": 0.0}, "capacity"),
            ({"capacity": 10.0, "regeneration_rate": -1.0}, "regeneration"),
            ({"capacity": 10.0, "ledger_size": 0}, "ledger_size"),
            ({"capacity": 10.0, "initial_balance": 100.0}, "initial_balance"),
            ({"capacity": 10.0, "initial_balance": -1.0}, "initial_balance"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(PoolError, match=match):
            ResourcePool(**kwargs)

    def test_consume_validation(self):
        pool = ResourcePool(capacity=10.0, clock=_FakeClock())
        with pytest.raises(PoolError, match="amount"):
            pool.consume(agent_id="a", amount=0)
        with pytest.raises(PoolError, match="agent_id"):
            ConsumptionRecord(agent_id="", amount=1.0, timestamp=0.0)

    def test_reset(self):
        pool = ResourcePool(capacity=100.0, clock=_FakeClock())
        pool.consume(agent_id="a", amount=30.0)
        pool.reset()
        assert pool.balance() == 100.0
        assert pool.ledger() == ()

    def test_reset_bad_balance(self):
        pool = ResourcePool(capacity=100.0, clock=_FakeClock())
        with pytest.raises(PoolError, match="balance"):
            pool.reset(balance=200.0)

    def test_concurrent_consume(self):
        pool = ResourcePool(capacity=1000.0, clock=_FakeClock())

        def writer(tag: str) -> None:
            for _ in range(100):
                with contextlib.suppress(PoolError):
                    pool.consume(agent_id=tag, amount=1.0)

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert pool.balance() >= 0.0
        assert len(pool.ledger()) <= 800


# --- AgentEconomicState -------------------------------------------


class TestAgentState:
    def test_valid(self):
        state = AgentEconomicState(agent_id="a", credit_balance=5.0, valuation=2.0)
        assert state.valuation == 2.0

    def test_negative_credit(self):
        with pytest.raises(PoolError, match="credit_balance"):
            AgentEconomicState(agent_id="a", credit_balance=-1.0, valuation=1.0)

    def test_empty_agent(self):
        with pytest.raises(PoolError, match="agent_id"):
            AgentEconomicState(agent_id="", credit_balance=0.0, valuation=1.0)


# --- NashBargainingSolver -----------------------------------------


def _agents(valuations: dict[str, float]) -> tuple[AgentEconomicState, ...]:
    return tuple(
        AgentEconomicState(agent_id=a, credit_balance=100.0, valuation=v)
        for a, v in valuations.items()
    )


class TestBargaining:
    def test_symmetric_splits_evenly(self):
        solver = NashBargainingSolver(step=0.1)
        agents = _agents({"a": 1.0, "b": 1.0})
        solution = solver.solve(agents=agents, budget=1.0)
        a_share = solution.allocation["a"]
        b_share = solution.allocation["b"]
        assert a_share == pytest.approx(0.5, abs=0.11)
        assert b_share == pytest.approx(0.5, abs=0.11)

    def test_equal_allocation_under_equal_disagreement(self):
        """Nash bargaining with zero disagreement points and a
        budget constraint gives every agent an equal share of
        the resource, regardless of valuation. Utilities then
        differ because u_i = v_i * x_i."""
        solver = NashBargainingSolver(step=0.1)
        agents = _agents({"low": 1.0, "high": 3.0})
        solution = solver.solve(agents=agents, budget=1.0)
        assert solution.allocation["low"] == pytest.approx(
            solution.allocation["high"], abs=0.11
        )
        # Utilities: low = 0.5, high = 1.5 → fairness_gap ≈ 0.667.
        assert solution.fairness_gap == pytest.approx(0.667, abs=0.1)

    def test_disagreement_point_affects_allocation(self):
        solver = NashBargainingSolver(step=0.1)
        agents = _agents({"a": 1.0, "b": 1.0})
        # b has a high disagreement utility — the solution must
        # give b more to keep the surplus positive.
        disagreement = (DisagreementPoint(agent_id="b", utility=0.5),)
        solution = solver.solve(agents=agents, budget=1.0, disagreement=disagreement)
        assert solution.allocation["b"] >= solution.allocation["a"]

    def test_fairness_gap_zero_when_equal(self):
        solver = NashBargainingSolver(step=0.1)
        agents = _agents({"a": 1.0, "b": 1.0})
        solution = solver.solve(agents=agents, budget=1.0)
        assert solution.fairness_gap < 0.3

    def test_duplicate_agent_rejected(self):
        solver = NashBargainingSolver(step=0.1)
        agents = (
            AgentEconomicState(agent_id="a", credit_balance=0.0, valuation=1.0),
            AgentEconomicState(agent_id="a", credit_balance=0.0, valuation=1.0),
        )
        with pytest.raises(ValueError, match="duplicate"):
            solver.solve(agents=agents, budget=1.0)

    def test_single_agent_rejected(self):
        solver = NashBargainingSolver(step=0.1)
        agents = _agents({"a": 1.0})
        with pytest.raises(ValueError, match="at least two"):
            solver.solve(agents=agents, budget=1.0)

    def test_bad_budget(self):
        solver = NashBargainingSolver(step=0.1)
        agents = _agents({"a": 1.0, "b": 1.0})
        with pytest.raises(ValueError, match="budget"):
            solver.solve(agents=agents, budget=0.0)

    def test_unknown_disagreement_agent(self):
        solver = NashBargainingSolver(step=0.1)
        agents = _agents({"a": 1.0, "b": 1.0})
        disagreement = (DisagreementPoint(agent_id="ghost", utility=0.1),)
        with pytest.raises(ValueError, match="unknown agents"):
            solver.solve(agents=agents, budget=1.0, disagreement=disagreement)

    def test_step_larger_than_budget(self):
        solver = NashBargainingSolver(step=2.0)
        agents = _agents({"a": 1.0, "b": 1.0})
        with pytest.raises(ValueError, match="grid points"):
            solver.solve(agents=agents, budget=1.0)

    def test_disagreement_validation(self):
        with pytest.raises(ValueError, match="utility"):
            DisagreementPoint(agent_id="a", utility=-1.0)
        with pytest.raises(ValueError, match="agent_id"):
            DisagreementPoint(agent_id="", utility=0.0)

    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="step"):
            NashBargainingSolver(step=0.0)
        with pytest.raises(ValueError, match="epsilon"):
            NashBargainingSolver(epsilon=0.0)

    def test_solution_is_dataclass(self):
        solver = NashBargainingSolver(step=0.1)
        agents = _agents({"a": 1.0, "b": 1.0})
        solution = solver.solve(agents=agents, budget=1.0)
        assert isinstance(solution, BargainingSolution)
        assert solution.total_allocated == pytest.approx(1.0, abs=0.1)


# --- TragedyDetector ----------------------------------------------


class TestTragedyDetector:
    def test_quiet_pool_returns_no_alarm(self):
        clock = _FakeClock(start=100.0)
        pool = ResourcePool(capacity=100.0, regeneration_rate=10.0, clock=clock)
        detector = TragedyDetector(
            pool=pool,
            window_seconds=60.0,
            grace_seconds=30.0,
            grace_factor=1.5,
            clock=clock,
        )
        signal = detector.check()
        assert isinstance(signal, TragedySignal)
        assert not signal.firing

    def test_over_consumption_fires_after_grace(self):
        clock = _FakeClock(start=0.0)
        pool = ResourcePool(capacity=10_000.0, regeneration_rate=1.0, clock=clock)
        detector = TragedyDetector(
            pool=pool,
            window_seconds=60.0,
            grace_seconds=10.0,
            grace_factor=1.0,
            clock=clock,
        )
        # Burn 10 units per second over 60 s → observed_rate = 10,
        # sustainable = 1, way above threshold.
        for tick in range(60):
            clock.now = float(tick)
            pool.consume(agent_id="greedy", amount=10.0)
        # First check — detector starts the grace clock.
        clock.now = 60.0
        first_signal = detector.check()
        assert not first_signal.firing
        # Advance past the grace window.
        clock.now = 80.0
        second_signal = detector.check()
        assert second_signal.firing
        assert second_signal.pressure > 0.5

    def test_non_regenerating_pool_pressure(self):
        clock = _FakeClock()
        pool = ResourcePool(capacity=100.0, clock=clock)
        detector = TragedyDetector(pool=pool, clock=clock)
        pool.consume(agent_id="a", amount=1.0)
        signal = detector.check()
        # With regeneration_rate == 0 any draw yields pressure 1.0.
        assert signal.pressure == 1.0

    def test_reset_clears_grace_clock(self):
        clock = _FakeClock(start=0.0)
        pool = ResourcePool(capacity=1000.0, regeneration_rate=1.0, clock=clock)
        detector = TragedyDetector(
            pool=pool,
            window_seconds=10.0,
            grace_seconds=5.0,
            grace_factor=1.0,
            clock=clock,
        )
        for i in range(10):
            clock.now = float(i)
            pool.consume(agent_id="a", amount=5.0)
        clock.now = 10.0
        detector.check()
        detector.reset()
        # Drop traffic so observed_rate normalises.
        clock.now = 100.0
        signal = detector.check()
        assert not signal.firing

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"window_seconds": 0.0}, "window_seconds"),
            ({"grace_seconds": -1.0}, "grace_seconds"),
            ({"grace_factor": 0.5}, "grace_factor"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        pool = ResourcePool(capacity=10.0, clock=_FakeClock())
        with pytest.raises(ValueError, match=match):
            TragedyDetector(pool=pool, **kwargs)


# --- EconomicRiskScorer -------------------------------------------


class TestEconomicRiskScorer:
    def test_empty_pool_has_low_risk(self):
        clock = _FakeClock()
        pool = ResourcePool(capacity=100.0, regeneration_rate=5.0, clock=clock)
        detector = TragedyDetector(pool=pool, clock=clock)
        scorer = EconomicRiskScorer(pool=pool, detector=detector)
        verdict = scorer.score()
        assert isinstance(verdict, EconomicVerdict)
        assert verdict.safe

    def test_exhausted_pool_raises_risk(self):
        clock = _FakeClock()
        pool = ResourcePool(capacity=100.0, initial_balance=5.0, clock=clock)
        detector = TragedyDetector(pool=pool, clock=clock)
        scorer = EconomicRiskScorer(
            pool=pool,
            detector=detector,
            weight_exhaustion=1.0,
            weight_fairness=0.0,
            weight_tragedy=0.0,
        )
        verdict = scorer.score()
        assert verdict.exhaustion_headroom > 0.9
        assert verdict.risk > 0.9

    def test_bargaining_result_feeds_fairness(self):
        clock = _FakeClock()
        pool = ResourcePool(capacity=100.0, regeneration_rate=10.0, clock=clock)
        detector = TragedyDetector(pool=pool, clock=clock)
        scorer = EconomicRiskScorer(
            pool=pool,
            detector=detector,
            weight_exhaustion=0.0,
            weight_fairness=1.0,
            weight_tragedy=0.0,
        )
        # Force a lopsided fairness_gap via a pre-built solution
        # dataclass (the scorer doesn't run bargaining itself).
        from director_ai.core.swarm_economics.bargaining import (
            BargainingSolution,
        )

        solution = BargainingSolution(
            allocation={"a": 0.9, "b": 0.1},
            nash_product=0.0,
            utilities={"a": 1.0, "b": 0.1},
            total_allocated=1.0,
        )
        verdict = scorer.score(bargaining=solution)
        assert verdict.fairness_gap == pytest.approx(0.9)
        assert verdict.risk == pytest.approx(0.9)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            (
                {
                    "weight_exhaustion": -0.1,
                    "weight_fairness": 0.5,
                    "weight_tragedy": 0.6,
                },
                "non-negative",
            ),
            (
                {
                    "weight_exhaustion": 0.4,
                    "weight_fairness": 0.4,
                    "weight_tragedy": 0.4,
                },
                "sum to 1",
            ),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        pool = ResourcePool(capacity=10.0, clock=_FakeClock())
        detector = TragedyDetector(pool=pool, clock=_FakeClock())
        with pytest.raises(ValueError, match=match):
            EconomicRiskScorer(pool=pool, detector=detector, **kwargs)
