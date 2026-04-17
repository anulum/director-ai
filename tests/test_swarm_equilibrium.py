# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — swarm equilibrium tests

"""Multi-angle coverage: StrategyProfile + NormalFormGame
validation, NashSolver on canonical games (prisoner's dilemma,
matching pennies, stag hunt, battle of sexes), analytical 2x2
mixed equilibrium, deviation_gain behaviour, Stackelberg
backward induction, scorer orchestration."""

from __future__ import annotations

import math

import pytest

from director_ai.core.swarm_equilibrium import (
    NashSolver,
    NormalFormGame,
    PayoffError,
    StabilityReport,
    StackelbergSolver,
    StrategyProfile,
    SwarmEquilibriumScorer,
)

# --- StrategyProfile -----------------------------------------------


class TestStrategyProfile:
    def test_valid(self):
        sp = StrategyProfile(choices=("a", "b"))
        assert sp.choices == ("a", "b")

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            StrategyProfile(choices=())

    def test_empty_choice_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            StrategyProfile(choices=("a", ""))


# --- NormalFormGame ------------------------------------------------


def _prisoners_dilemma() -> NormalFormGame:
    """Classic PD. Row player is ``row``, column player is ``col``.

    Cooperate-Cooperate → (3, 3); Cooperate-Defect → (0, 5);
    Defect-Cooperate → (5, 0); Defect-Defect → (1, 1).

    Unique Nash equilibrium: (Defect, Defect) with payoffs (1, 1).
    """
    players = ("row", "col")
    strategies = {p: ("C", "D") for p in players}
    payoffs = {
        StrategyProfile(("C", "C")): (3.0, 3.0),
        StrategyProfile(("C", "D")): (0.0, 5.0),
        StrategyProfile(("D", "C")): (5.0, 0.0),
        StrategyProfile(("D", "D")): (1.0, 1.0),
    }
    return NormalFormGame(players=players, strategies=strategies, payoffs=payoffs)


def _matching_pennies() -> NormalFormGame:
    """Zero-sum game with a unique mixed Nash at (1/2, 1/2)."""
    players = ("row", "col")
    strategies = {p: ("H", "T") for p in players}
    payoffs = {
        StrategyProfile(("H", "H")): (1.0, -1.0),
        StrategyProfile(("H", "T")): (-1.0, 1.0),
        StrategyProfile(("T", "H")): (-1.0, 1.0),
        StrategyProfile(("T", "T")): (1.0, -1.0),
    }
    return NormalFormGame(players=players, strategies=strategies, payoffs=payoffs)


def _stag_hunt() -> NormalFormGame:
    """Stag hunt: two pure Nash equilibria (Stag, Stag) and
    (Hare, Hare), one mixed Nash."""
    players = ("row", "col")
    strategies = {p: ("Stag", "Hare") for p in players}
    payoffs = {
        StrategyProfile(("Stag", "Stag")): (4.0, 4.0),
        StrategyProfile(("Stag", "Hare")): (0.0, 2.0),
        StrategyProfile(("Hare", "Stag")): (2.0, 0.0),
        StrategyProfile(("Hare", "Hare")): (2.0, 2.0),
    }
    return NormalFormGame(players=players, strategies=strategies, payoffs=payoffs)


class TestNormalFormGame:
    def test_valid_construction(self):
        game = _prisoners_dilemma()
        assert game.players == ("row", "col")
        assert len(game.payoffs) == 4

    def test_missing_payoff_rejected(self):
        with pytest.raises(PayoffError, match="profiles"):
            NormalFormGame(
                players=("a", "b"),
                strategies={"a": ("x", "y"), "b": ("p", "q")},
                payoffs={StrategyProfile(("x", "p")): (1.0, 1.0)},
            )

    def test_unknown_choice_rejected(self):
        with pytest.raises(PayoffError, match="not in player"):
            NormalFormGame(
                players=("a", "b"),
                strategies={"a": ("x",), "b": ("p",)},
                payoffs={StrategyProfile(("x", "bogus")): (1.0, 1.0)},
            )

    def test_non_finite_payoff_rejected(self):
        with pytest.raises(PayoffError, match="finite"):
            NormalFormGame(
                players=("a", "b"),
                strategies={"a": ("x",), "b": ("p",)},
                payoffs={StrategyProfile(("x", "p")): (float("inf"), 0.0)},
            )

    def test_payoff_tuple_length_enforced(self):
        with pytest.raises(PayoffError, match="payoff tuple length"):
            NormalFormGame(
                players=("a", "b"),
                strategies={"a": ("x",), "b": ("p",)},
                payoffs={StrategyProfile(("x", "p")): (1.0, 2.0, 3.0)},
            )

    def test_single_player_rejected(self):
        with pytest.raises(ValueError, match="at least two"):
            NormalFormGame(
                players=("a",),
                strategies={"a": ("x",)},
                payoffs={StrategyProfile(("x",)): (1.0,)},
            )

    def test_duplicate_players_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            NormalFormGame(
                players=("a", "a"),
                strategies={"a": ("x",)},
                payoffs={},
            )

    def test_empty_strategies_rejected(self):
        with pytest.raises(PayoffError, match="empty strategy"):
            NormalFormGame(
                players=("a", "b"),
                strategies={"a": (), "b": ("x",)},
                payoffs={},
            )

    def test_missing_strategy_spec_rejected(self):
        with pytest.raises(PayoffError, match="no strategies"):
            NormalFormGame(
                players=("a", "b"),
                strategies={"a": ("x",)},
                payoffs={},
            )

    def test_best_responses(self):
        game = _prisoners_dilemma()
        # Row best-responds to col playing C.
        assert game.best_responses(StrategyProfile(("D", "C")), "row") == ("D",)
        # Row best-responds to col playing D.
        assert game.best_responses(StrategyProfile(("D", "D")), "row") == ("D",)

    def test_unknown_player_rejected_in_best_responses(self):
        game = _prisoners_dilemma()
        with pytest.raises(ValueError, match="unknown player"):
            game.best_responses(StrategyProfile(("C", "C")), "bogus")

    def test_payoff_lookup_for_unknown_profile(self):
        game = _prisoners_dilemma()
        with pytest.raises(KeyError, match="unknown profile"):
            game.payoff(StrategyProfile(("X", "Y")), "row")


# --- NashSolver ----------------------------------------------------


class TestNashSolver:
    def test_prisoners_dilemma_single_nash(self):
        solver = NashSolver()
        pures = solver.pure_equilibria(_prisoners_dilemma())
        assert len(pures) == 1
        assert pures[0].profile == StrategyProfile(("D", "D"))
        assert pures[0].expected_payoffs == (1.0, 1.0)

    def test_matching_pennies_no_pure(self):
        solver = NashSolver()
        pures = solver.pure_equilibria(_matching_pennies())
        assert pures == ()

    def test_matching_pennies_mixed_equilibrium(self):
        solver = NashSolver()
        mixed = solver.mixed_equilibrium_2x2(_matching_pennies())
        assert mixed is not None
        row_probs = mixed.mixed_strategies["row"]
        col_probs = mixed.mixed_strategies["col"]
        assert row_probs["H"] == pytest.approx(0.5)
        assert row_probs["T"] == pytest.approx(0.5)
        assert col_probs["H"] == pytest.approx(0.5)
        assert col_probs["T"] == pytest.approx(0.5)
        # Expected payoffs of mixed equilibrium in matching pennies = 0.
        assert all(abs(p) < 1e-9 for p in mixed.expected_payoffs)

    def test_stag_hunt_multiple_pure(self):
        solver = NashSolver()
        pures = solver.pure_equilibria(_stag_hunt())
        profile_set = {eq.profile for eq in pures}
        assert StrategyProfile(("Stag", "Stag")) in profile_set
        assert StrategyProfile(("Hare", "Hare")) in profile_set

    def test_mixed_falls_back_to_none_for_pure_games(self):
        solver = NashSolver()
        # The PD has a strict pure equilibrium at (D, D); the
        # analytical mixed formula must land outside (0, 1) or
        # otherwise return None.
        mixed = solver.mixed_equilibrium_2x2(_prisoners_dilemma())
        assert mixed is None

    def test_equilibria_combines_pure_and_mixed(self):
        solver = NashSolver()
        eqs = solver.equilibria(_matching_pennies())
        # No pure equilibria; one mixed.
        assert len(eqs) == 1
        assert eqs[0].kind == "mixed"

    def test_deviation_gain_zero_on_equilibrium(self):
        solver = NashSolver()
        gain = solver.deviation_gain(
            _prisoners_dilemma(), StrategyProfile(("D", "D"))
        )
        assert gain == pytest.approx(0.0)

    def test_deviation_gain_positive_off_equilibrium(self):
        solver = NashSolver()
        gain = solver.deviation_gain(
            _prisoners_dilemma(), StrategyProfile(("C", "C"))
        )
        # Either player can unilaterally defect for a +2 payoff gain
        # (3 → 5).
        assert gain == pytest.approx(2.0)

    def test_three_player_game(self):
        """Three-player coordination game — pure Nash only when all
        three pick the same strategy."""
        players = ("a", "b", "c")
        strategies = {p: ("0", "1") for p in players}
        payoffs: dict[StrategyProfile, tuple[float, ...]] = {}
        for x in ("0", "1"):
            for y in ("0", "1"):
                for z in ("0", "1"):
                    reward = 5.0 if x == y == z else 0.0
                    payoffs[StrategyProfile((x, y, z))] = (reward, reward, reward)
        game = NormalFormGame(
            players=players, strategies=strategies, payoffs=payoffs
        )
        pures = NashSolver().pure_equilibria(game)
        assert len(pures) == 2
        profile_set = {eq.profile for eq in pures}
        assert StrategyProfile(("0", "0", "0")) in profile_set
        assert StrategyProfile(("1", "1", "1")) in profile_set

    def test_negative_eps_rejected(self):
        with pytest.raises(ValueError, match="eps"):
            NashSolver(eps=-1e-6)


# --- StackelbergSolver ---------------------------------------------


class TestStackelbergSolver:
    def test_row_as_leader_defects(self):
        """PD with row leading: row commits to D, col best-responds
        with D, leader gets 1."""
        solver = StackelbergSolver()
        eq = solver.solve(_prisoners_dilemma(), leader="row")
        assert eq.profile == StrategyProfile(("D", "D"))
        assert eq.expected_payoffs == (1.0, 1.0)

    def test_col_as_leader(self):
        solver = StackelbergSolver()
        eq = solver.solve(_prisoners_dilemma(), leader="col")
        assert eq.profile == StrategyProfile(("D", "D"))

    def test_unknown_leader_rejected(self):
        solver = StackelbergSolver()
        with pytest.raises(ValueError, match="not in the player list"):
            solver.solve(_prisoners_dilemma(), leader="bogus")

    def test_non_two_player_rejected(self):
        players = ("a", "b", "c")
        strategies = {p: ("0",) for p in players}
        payoffs = {StrategyProfile(("0", "0", "0")): (0.0, 0.0, 0.0)}
        game = NormalFormGame(
            players=players, strategies=strategies, payoffs=payoffs
        )
        solver = StackelbergSolver()
        with pytest.raises(ValueError, match="two-player"):
            solver.solve(game, leader="a")

    def test_stag_hunt_leader_picks_stag(self):
        """With commitment power, the row leader picks Stag —
        follower best-responds with Stag for the (4, 4) outcome."""
        solver = StackelbergSolver()
        eq = solver.solve(_stag_hunt(), leader="row")
        assert eq.profile == StrategyProfile(("Stag", "Stag"))


# --- SwarmEquilibriumScorer ---------------------------------------


class TestScorer:
    def test_nash_profile_has_zero_deviation(self):
        scorer = SwarmEquilibriumScorer(payoff_scale=4.0)
        report = scorer.score(
            game=_prisoners_dilemma(),
            observed=StrategyProfile(("D", "D")),
        )
        assert isinstance(report, StabilityReport)
        assert report.observed_is_nash
        assert report.deviation_gain == pytest.approx(0.0)
        assert report.stability_score == pytest.approx(1.0)

    def test_off_equilibrium_drops_stability(self):
        scorer = SwarmEquilibriumScorer(payoff_scale=4.0)
        report = scorer.score(
            game=_prisoners_dilemma(),
            observed=StrategyProfile(("C", "C")),
        )
        assert not report.observed_is_nash
        assert report.stability_score < 1.0

    def test_scorer_with_stackelberg_leader(self):
        scorer = SwarmEquilibriumScorer()
        report = scorer.score(
            game=_prisoners_dilemma(),
            observed=StrategyProfile(("D", "D")),
            leader="row",
        )
        assert report.stackelberg_profile is not None
        assert report.stackelberg_profile.profile == StrategyProfile(("D", "D"))

    def test_scorer_exposes_all_pure_equilibria(self):
        scorer = SwarmEquilibriumScorer()
        report = scorer.score(
            game=_stag_hunt(),
            observed=StrategyProfile(("Stag", "Stag")),
        )
        profiles = {eq.profile for eq in report.nash_profiles if eq.kind == "pure"}
        assert StrategyProfile(("Stag", "Stag")) in profiles
        assert StrategyProfile(("Hare", "Hare")) in profiles

    def test_unknown_observed_profile_rejected(self):
        scorer = SwarmEquilibriumScorer()
        with pytest.raises(KeyError):
            scorer.score(
                game=_prisoners_dilemma(),
                observed=StrategyProfile(("X", "Y")),
            )

    def test_bad_payoff_scale(self):
        with pytest.raises(ValueError, match="payoff_scale"):
            SwarmEquilibriumScorer(payoff_scale=0.0)

    def test_mean_nash_payoff(self):
        scorer = SwarmEquilibriumScorer()
        payoff = scorer.mean_nash_payoff(_stag_hunt(), "row")
        # Two pure NE: (4, 4) and (2, 2). Mean row payoff = 3.
        assert payoff == pytest.approx(3.0)

    def test_mean_nash_payoff_no_pure(self):
        scorer = SwarmEquilibriumScorer()
        payoff = scorer.mean_nash_payoff(_matching_pennies(), "row")
        assert math.isnan(payoff)

    def test_mean_nash_payoff_unknown_player(self):
        scorer = SwarmEquilibriumScorer()
        with pytest.raises(ValueError, match="unknown player"):
            scorer.mean_nash_payoff(_prisoners_dilemma(), "ghost")
