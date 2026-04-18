# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Nash + Stackelberg solvers

"""Solvers for :class:`NormalFormGame` equilibria.

:class:`NashSolver` enumerates pure-strategy Nash equilibria
directly (every profile is a candidate; a profile is a Nash
equilibrium iff no player has a strictly better response). For
two-player games with no pure Nash, the solver falls back to the
analytical 2x2 mixed-strategy formula where possible.

:class:`StackelbergSolver` runs backward induction on two-player
sequential games — the leader commits first, the follower
observes the commitment and plays a best response, and the
leader picks the commitment that maximises their own payoff
under the induced follower response.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .game import NormalFormGame, StrategyProfile


@dataclass(frozen=True)
class NashEquilibrium:
    """One equilibrium.

    ``kind`` discriminates pure ("pure") vs. mixed ("mixed").
    ``profile`` is the concrete :class:`StrategyProfile` for pure
    equilibria; ``mixed_strategies`` carries the per-player
    strategy → probability map for mixed equilibria.
    ``expected_payoffs`` is the per-player expected payoff
    under the equilibrium.
    """

    kind: str
    expected_payoffs: tuple[float, ...]
    profile: StrategyProfile | None = None
    mixed_strategies: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"pure", "mixed"}:
            raise ValueError(f"kind must be 'pure' or 'mixed'; got {self.kind!r}")
        if self.kind == "pure" and self.profile is None:
            raise ValueError("pure equilibrium requires a profile")
        if self.kind == "mixed" and not self.mixed_strategies:
            raise ValueError("mixed equilibrium requires mixed_strategies")


class NashSolver:
    """Find pure and mixed Nash equilibria of a
    :class:`NormalFormGame`.

    Parameters
    ----------
    eps :
        Absolute tolerance for best-response comparisons. Default
        ``1e-9``.
    """

    def __init__(self, *, eps: float = 1e-9) -> None:
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self._eps = eps

    def pure_equilibria(self, game: NormalFormGame) -> tuple[NashEquilibrium, ...]:
        """Enumerate every pure-strategy Nash equilibrium."""
        equilibria: list[NashEquilibrium] = []
        for profile in game.profiles():
            if self._is_pure_ne(game, profile):
                equilibria.append(
                    NashEquilibrium(
                        kind="pure",
                        profile=profile,
                        expected_payoffs=tuple(game.payoffs[profile]),
                    )
                )
        return tuple(equilibria)

    def mixed_equilibrium_2x2(self, game: NormalFormGame) -> NashEquilibrium | None:
        """Return the interior mixed-strategy Nash equilibrium of a
        2x2 two-player game, or ``None`` when the closed-form
        solution lies outside the simplex (i.e. a pure equilibrium
        dominates).
        """
        if len(game.players) != 2:
            return None
        p0, p1 = game.players
        if len(game.strategies[p0]) != 2 or len(game.strategies[p1]) != 2:
            return None
        s0_a, s0_b = game.strategies[p0]
        s1_a, s1_b = game.strategies[p1]
        # Player 0 picks probability p of s0_a so that player 1 is
        # indifferent between s1_a and s1_b:
        #     p * u1(s0_a, s1_a) + (1-p) * u1(s0_b, s1_a)
        #   = p * u1(s0_a, s1_b) + (1-p) * u1(s0_b, s1_b)
        u1_aa = game.payoff(StrategyProfile((s0_a, s1_a)), p1)
        u1_ba = game.payoff(StrategyProfile((s0_b, s1_a)), p1)
        u1_ab = game.payoff(StrategyProfile((s0_a, s1_b)), p1)
        u1_bb = game.payoff(StrategyProfile((s0_b, s1_b)), p1)
        denom0 = (u1_aa - u1_ba) - (u1_ab - u1_bb)
        if abs(denom0) < self._eps:
            return None
        p = (u1_bb - u1_ba) / denom0
        if not 0.0 < p < 1.0:
            return None
        # Player 1 picks probability q of s1_a so that player 0 is
        # indifferent between s0_a and s0_b:
        u0_aa = game.payoff(StrategyProfile((s0_a, s1_a)), p0)
        u0_ba = game.payoff(StrategyProfile((s0_b, s1_a)), p0)
        u0_ab = game.payoff(StrategyProfile((s0_a, s1_b)), p0)
        u0_bb = game.payoff(StrategyProfile((s0_b, s1_b)), p0)
        denom1 = (u0_aa - u0_ab) - (u0_ba - u0_bb)
        if abs(denom1) < self._eps:
            return None
        q = (u0_bb - u0_ab) / denom1
        if not 0.0 < q < 1.0:
            return None
        expected_p0 = (
            p * q * u0_aa
            + p * (1 - q) * u0_ab
            + (1 - p) * q * u0_ba
            + (1 - p) * (1 - q) * u0_bb
        )
        expected_p1 = (
            p * q * u1_aa
            + p * (1 - q) * u1_ab
            + (1 - p) * q * u1_ba
            + (1 - p) * (1 - q) * u1_bb
        )
        return NashEquilibrium(
            kind="mixed",
            expected_payoffs=(expected_p0, expected_p1),
            mixed_strategies={
                p0: {s0_a: p, s0_b: 1.0 - p},
                p1: {s1_a: q, s1_b: 1.0 - q},
            },
        )

    def equilibria(self, game: NormalFormGame) -> tuple[NashEquilibrium, ...]:
        """Every equilibrium: pure set first, then a mixed 2x2
        result if applicable."""
        pures = self.pure_equilibria(game)
        mixed = self.mixed_equilibrium_2x2(game)
        return pures + ((mixed,) if mixed is not None else ())

    def deviation_gain(self, game: NormalFormGame, profile: StrategyProfile) -> float:
        """Largest unilateral payoff improvement any single player
        can achieve by deviating from ``profile``. Zero when
        ``profile`` is a Nash equilibrium.
        """
        current = [game.payoff(profile, p) for p in game.players]
        worst_gain = 0.0
        for idx, player in enumerate(game.players):
            for alt in game.strategies[player]:
                if alt == profile.choices[idx]:
                    continue
                alt_choices = list(profile.choices)
                alt_choices[idx] = alt
                alt_profile = StrategyProfile(choices=tuple(alt_choices))
                gain = game.payoff(alt_profile, player) - current[idx]
                worst_gain = max(worst_gain, gain)
        return worst_gain

    def _is_pure_ne(self, game: NormalFormGame, profile: StrategyProfile) -> bool:
        for idx, player in enumerate(game.players):
            best = game.best_responses(profile, player)
            if profile.choices[idx] not in best:
                return False
        return True


class StackelbergSolver:
    """Backward induction for two-player sequential games."""

    def __init__(self, *, eps: float = 1e-9) -> None:
        if eps < 0:
            raise ValueError("eps must be non-negative")
        self._eps = eps

    def solve(self, game: NormalFormGame, *, leader: str) -> NashEquilibrium:
        """Return the subgame-perfect equilibrium with ``leader``
        moving first. The follower best-responds to the leader's
        commitment; the leader picks the commitment that
        maximises their own payoff given the induced follower
        response. Ties between follower best responses are broken
        in favour of the leader (the leader picks the follower
        response that is best for the leader, a standard
        Stackelberg assumption)."""
        if len(game.players) != 2:
            raise ValueError("Stackelberg solver handles two-player games")
        if leader not in game.players:
            raise ValueError(f"leader {leader!r} is not in the player list")
        follower = next(p for p in game.players if p != leader)
        leader_idx = game.players.index(leader)
        follower_idx = game.players.index(follower)
        best_leader_payoff = -float("inf")
        chosen_profile: StrategyProfile | None = None
        chosen_payoffs: tuple[float, float] | None = None
        for leader_strategy in game.strategies[leader]:
            for profile in game.profiles():
                if profile.choices[leader_idx] != leader_strategy:
                    continue
                follower_best = game.best_responses(profile, follower)
                if profile.choices[follower_idx] not in follower_best:
                    continue
                leader_payoff = game.payoff(profile, leader)
                follower_payoff = game.payoff(profile, follower)
                if leader_payoff > best_leader_payoff + self._eps:
                    best_leader_payoff = leader_payoff
                    chosen_profile = profile
                    chosen_payoffs = (leader_payoff, follower_payoff)
        if chosen_profile is None or chosen_payoffs is None:
            raise ValueError("no Stackelberg outcome found")
        leader_payoff, follower_payoff = chosen_payoffs
        if game.players[0] == leader:
            payoffs_tuple: tuple[float, ...] = (leader_payoff, follower_payoff)
        else:
            payoffs_tuple = (follower_payoff, leader_payoff)
        return NashEquilibrium(
            kind="pure",
            profile=chosen_profile,
            expected_payoffs=payoffs_tuple,
        )
