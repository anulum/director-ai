# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SwarmEquilibriumScorer

"""Score an observed :class:`StrategyProfile` against a game's
Nash / Stackelberg equilibria.

The scorer takes a game, an observed profile, and an optional
Stackelberg leader. It reports:

* ``nash_profiles`` — every pure Nash equilibrium of the game.
* ``observed_is_nash`` — whether the observed profile is one of
  them.
* ``deviation_gain`` — the largest unilateral payoff improvement
  any single player can still obtain from the observed profile.
  Zero on a Nash equilibrium.
* ``stackelberg_profile`` — the subgame-perfect equilibrium when
  a leader is named, else ``None``.
* ``stability_score`` — a scalar in ``[0, 1]`` that folds
  deviation gain into a monotone score so downstream routers
  can halt drifting swarms with a single threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .game import NormalFormGame, StrategyProfile
from .solvers import NashEquilibrium, NashSolver, StackelbergSolver


@dataclass(frozen=True)
class StabilityReport:
    """Outcome of one :meth:`SwarmEquilibriumScorer.score` call."""

    observed: StrategyProfile
    deviation_gain: float
    stability_score: float
    observed_is_nash: bool
    nash_profiles: tuple[NashEquilibrium, ...] = field(default_factory=tuple)
    stackelberg_profile: NashEquilibrium | None = None


class SwarmEquilibriumScorer:
    """Compose :class:`NashSolver` and :class:`StackelbergSolver`
    into a single scorer.

    Parameters
    ----------
    nash :
        Solver for Nash equilibria. Defaults to a fresh
        :class:`NashSolver`.
    stackelberg :
        Solver for sequential games. Defaults to a fresh
        :class:`StackelbergSolver`.
    payoff_scale :
        Expected payoff range used to normalise
        :attr:`StabilityReport.stability_score`. The stability
        score is ``max(0, 1 - deviation_gain / payoff_scale)``
        so callers compare across games by tuning this knob.
        Default 1.0.
    """

    def __init__(
        self,
        *,
        nash: NashSolver | None = None,
        stackelberg: StackelbergSolver | None = None,
        payoff_scale: float = 1.0,
    ) -> None:
        if payoff_scale <= 0:
            raise ValueError("payoff_scale must be positive")
        self._nash = nash or NashSolver()
        self._stackelberg = stackelberg or StackelbergSolver()
        self._payoff_scale = payoff_scale

    def score(
        self,
        *,
        game: NormalFormGame,
        observed: StrategyProfile,
        leader: str | None = None,
    ) -> StabilityReport:
        game.payoffs[observed]  # raises KeyError on an unknown profile
        nash_profiles = self._nash.equilibria(game)
        deviation = self._nash.deviation_gain(game, observed)
        observed_is_nash = any(
            eq.profile == observed for eq in nash_profiles if eq.kind == "pure"
        )
        stability = max(0.0, 1.0 - deviation / self._payoff_scale)
        stackelberg = (
            self._stackelberg.solve(game, leader=leader)
            if leader is not None
            else None
        )
        return StabilityReport(
            observed=observed,
            deviation_gain=deviation,
            stability_score=min(1.0, stability),
            observed_is_nash=observed_is_nash,
            nash_profiles=nash_profiles,
            stackelberg_profile=stackelberg,
        )

    def mean_nash_payoff(
        self, game: NormalFormGame, player: str
    ) -> float:
        """Average Nash-equilibrium payoff for ``player`` across
        every pure equilibrium. Returns ``nan`` when no pure
        equilibrium exists so callers branch on
        :func:`math.isnan` rather than swallowing a silent zero.
        """
        if player not in game.players:
            raise ValueError(f"unknown player {player!r}")
        pures = self._nash.pure_equilibria(game)
        if not pures:
            return float("nan")
        idx = game.players.index(player)
        return sum(eq.expected_payoffs[idx] for eq in pures) / len(pures)
