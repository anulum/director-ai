# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TrajectorySimulator tests

"""Multi-angle coverage for the trajectory simulator: deterministic
replay, halt-rate aggregation, proceed / warn / halt bands,
threshold validation, per-trajectory callback, empty-sample guards,
and credible-interval bounds."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from director_ai.core.trajectory import (
    PreflightVerdict,
    TrajectoryResult,
    TrajectorySimulator,
)

# --- Test doubles ----------------------------------------------------


class _FixedActor:
    """Returns a fixed token list; ignores seed so the halt rate is
    fully controlled by the scorer."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    def sample(self, _prompt: str, _seed: int) -> list[str]:
        return list(self._tokens)


class _SeededActor:
    """Token list depends on seed; used to verify deterministic
    replay."""

    def sample(self, prompt: str, seed: int) -> list[str]:
        rng = random.Random(seed)
        words = prompt.split()
        return [rng.choice(words) for _ in range(5)] if words else [f"tok{seed}"]


@dataclass
class _Score:
    score: float


class _FixedScorer:
    """Always returns the same verdict."""

    def __init__(self, approved: bool, coherence: float) -> None:
        self._approved = approved
        self._coherence = coherence
        self.calls = 0

    def review(
        self, prompt: str, action: str, tenant_id: str = ""
    ) -> tuple[bool, _Score]:
        self.calls += 1
        return self._approved, _Score(score=self._coherence)


class _SequenceScorer:
    """Cycles through a sequence of verdicts — useful for testing
    halt-rate aggregation with a known mix."""

    def __init__(self, sequence: list[tuple[bool, float]]) -> None:
        self._seq = sequence
        self._i = 0

    def review(
        self, prompt: str, action: str, tenant_id: str = ""
    ) -> tuple[bool, _Score]:
        approved, coh = self._seq[self._i % len(self._seq)]
        self._i += 1
        return approved, _Score(score=coh)


# --- Tests -----------------------------------------------------------


class TestConstruction:
    def test_default_ok(self):
        TrajectorySimulator(
            actor=_FixedActor(["a", "b"]),
            scorer=_FixedScorer(True, 0.9),
        )

    def test_n_simulations_must_be_positive(self):
        with pytest.raises(ValueError, match="n_simulations"):
            TrajectorySimulator(
                actor=_FixedActor(["a"]),
                scorer=_FixedScorer(True, 0.9),
                n_simulations=0,
            )

    def test_threshold_ordering_validated(self):
        with pytest.raises(ValueError, match="warn < halt"):
            TrajectorySimulator(
                actor=_FixedActor(["a"]),
                scorer=_FixedScorer(True, 0.9),
                halt_rate_warn=0.6,
                halt_rate_halt=0.3,
            )

    def test_ci_level_validated(self):
        with pytest.raises(ValueError, match="ci_level"):
            TrajectorySimulator(
                actor=_FixedActor(["a"]),
                scorer=_FixedScorer(True, 0.9),
                ci_level=1.1,
            )


class TestProceedBand:
    def test_all_approved_proceeds(self):
        sim = TrajectorySimulator(
            actor=_FixedActor(["clean"]),
            scorer=_FixedScorer(True, 0.9),
            n_simulations=4,
        )
        verdict = sim.preflight("Hello.")
        assert verdict.recommended == "proceed"
        assert verdict.halt_rate == 0.0
        assert verdict.mean_coherence == pytest.approx(0.9)
        assert verdict.min_coherence == pytest.approx(0.9)
        assert verdict.max_coherence == pytest.approx(0.9)
        assert "halt_rate=" in verdict.reason


class TestWarnBand:
    def test_partial_halts_warns(self):
        seq = [(True, 0.8), (False, 0.3), (True, 0.75), (True, 0.78)]
        sim = TrajectorySimulator(
            actor=_FixedActor(["a"]),
            scorer=_SequenceScorer(seq),
            n_simulations=4,
            halt_rate_warn=0.2,
            halt_rate_halt=0.5,
        )
        verdict = sim.preflight("hello")
        assert verdict.recommended == "warn"
        assert verdict.halt_rate == pytest.approx(0.25)
        assert "warn_threshold" in verdict.reason


class TestHaltBand:
    def test_most_halts_triggers_halt(self):
        seq = [(False, 0.1), (False, 0.2), (True, 0.8), (False, 0.15)]
        sim = TrajectorySimulator(
            actor=_FixedActor(["a"]),
            scorer=_SequenceScorer(seq),
            n_simulations=4,
        )
        verdict = sim.preflight("hello")
        assert verdict.recommended == "halt"
        assert verdict.halt_rate == pytest.approx(0.75)
        assert "halt_threshold" in verdict.reason


class TestDeterminism:
    def test_seeded_actor_produces_identical_trajectories(self):
        sim = TrajectorySimulator(
            actor=_SeededActor(),
            scorer=_FixedScorer(True, 0.9),
            n_simulations=4,
            base_seed=42,
        )
        a = sim.preflight("quick brown fox jumps over the lazy dog")
        b = sim.preflight("quick brown fox jumps over the lazy dog")
        assert [t.tokens for t in a.trajectories] == [t.tokens for t in b.trajectories]
        assert [t.seed for t in a.trajectories] == [t.seed for t in b.trajectories]

    def test_different_base_seeds_produce_different_draws(self):
        sim_a = TrajectorySimulator(
            actor=_SeededActor(),
            scorer=_FixedScorer(True, 0.9),
            n_simulations=4,
            base_seed=10,
        )
        sim_b = TrajectorySimulator(
            actor=_SeededActor(),
            scorer=_FixedScorer(True, 0.9),
            n_simulations=4,
            base_seed=99,
        )
        a = sim_a.preflight("quick brown fox jumps over lazy dog alpha beta")
        b = sim_b.preflight("quick brown fox jumps over lazy dog alpha beta")
        assert [t.tokens for t in a.trajectories] != [t.tokens for t in b.trajectories]


class TestCallback:
    def test_on_trajectory_fires_for_each_draw(self):
        calls: list[TrajectoryResult] = []
        sim = TrajectorySimulator(
            actor=_FixedActor(["a"]),
            scorer=_FixedScorer(True, 0.9),
            n_simulations=3,
        )
        sim.preflight("x", on_trajectory=calls.append)
        assert len(calls) == 3
        assert [c.trajectory_id for c in calls] == [0, 1, 2]

    def test_callback_failure_does_not_abort(self):
        def boom(_t: TrajectoryResult) -> None:
            raise RuntimeError("sink broken")

        sim = TrajectorySimulator(
            actor=_FixedActor(["a"]),
            scorer=_FixedScorer(True, 0.9),
            n_simulations=2,
        )
        verdict = sim.preflight("x", on_trajectory=boom)
        assert verdict.n_simulations == 2
        assert verdict.halt_rate == 0.0


class TestVerdictShape:
    def test_trajectories_preserved(self):
        sim = TrajectorySimulator(
            actor=_FixedActor(["Paris"]),
            scorer=_FixedScorer(True, 0.88),
            n_simulations=2,
        )
        verdict = sim.preflight("capital?")
        assert verdict.n_simulations == 2
        assert len(verdict.trajectories) == 2
        for t in verdict.trajectories:
            assert t.text == "Paris"

    def test_empty_verdict_on_zero_simulations_raises_at_construction(self):
        with pytest.raises(ValueError):
            TrajectorySimulator(
                actor=_FixedActor(["x"]),
                scorer=_FixedScorer(True, 0.9),
                n_simulations=0,
            )

    def test_ci_bounds_include_mean(self):
        seq = [(True, c) for c in (0.1, 0.3, 0.5, 0.7, 0.9)]
        sim = TrajectorySimulator(
            actor=_FixedActor(["a"]),
            scorer=_SequenceScorer(seq),
            n_simulations=5,
        )
        verdict = sim.preflight("x")
        assert verdict.ci_low <= verdict.mean_coherence <= verdict.ci_high


class TestAggregateMetrics:
    def test_min_max_match_trajectories(self):
        seq = [(True, c) for c in (0.2, 0.5, 0.8)]
        sim = TrajectorySimulator(
            actor=_FixedActor(["a"]),
            scorer=_SequenceScorer(seq),
            n_simulations=3,
        )
        verdict = sim.preflight("x")
        assert verdict.min_coherence == pytest.approx(0.2)
        assert verdict.max_coherence == pytest.approx(0.8)

    def test_std_zero_for_single_value(self):
        sim = TrajectorySimulator(
            actor=_FixedActor(["a"]),
            scorer=_FixedScorer(True, 0.9),
            n_simulations=1,
        )
        verdict = sim.preflight("x")
        assert verdict.std_coherence == 0.0

    def test_isinstance_pre_flight_verdict(self):
        sim = TrajectorySimulator(
            actor=_FixedActor(["a"]),
            scorer=_FixedScorer(True, 0.9),
            n_simulations=1,
        )
        assert isinstance(sim.preflight("x"), PreflightVerdict)
