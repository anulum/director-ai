# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NormalFormGame

"""Typed normal-form game over a small number of players.

``payoffs`` is a mapping from :class:`StrategyProfile` (one
strategy per player) to a tuple of floats giving each player's
payoff. The game rejects malformed constructions immediately:

* The payoff mapping must cover every profile in the Cartesian
  product of per-player strategy sets (no missing combinations).
* Every payoff tuple has length equal to ``len(players)``.
* Every payoff is finite.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field


class PayoffError(ValueError):
    """Raised when the payoff specification is malformed."""


@dataclass(frozen=True)
class StrategyProfile:
    """One strategy per player.

    ``choices`` is a tuple the same length as the game's player
    list, keyed positionally. Equality + hashing are automatic so
    profiles slot into ``dict`` / ``set`` without boilerplate.
    """

    choices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("StrategyProfile.choices must be non-empty")
        for c in self.choices:
            if not c:
                raise ValueError("every choice must be non-empty")


@dataclass(frozen=True)
class NormalFormGame:
    """Normal-form game.

    ``players`` is the ordered player list. ``strategies`` maps a
    player name to the finite list of strategies available to
    them. ``payoffs`` maps a :class:`StrategyProfile` to the
    per-player payoff tuple.
    """

    players: tuple[str, ...]
    strategies: Mapping[str, tuple[str, ...]]
    payoffs: Mapping[StrategyProfile, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.players) < 2:
            raise ValueError("a game needs at least two players")
        if len(set(self.players)) != len(self.players):
            raise ValueError("duplicate player names")
        for player in self.players:
            if player not in self.strategies:
                raise PayoffError(f"no strategies supplied for player {player!r}")
            if not self.strategies[player]:
                raise PayoffError(f"player {player!r} has an empty strategy set")
        expected_size = 1
        for player in self.players:
            expected_size *= len(self.strategies[player])
        if len(self.payoffs) != expected_size:
            raise PayoffError(
                f"payoffs cover {len(self.payoffs)} profiles but "
                f"{expected_size} are required"
            )
        for profile, payoff_tuple in self.payoffs.items():
            if len(profile.choices) != len(self.players):
                raise PayoffError(
                    f"profile {profile.choices} does not match player count"
                )
            for choice, player in zip(profile.choices, self.players, strict=True):
                if choice not in self.strategies[player]:
                    raise PayoffError(
                        f"choice {choice!r} is not in player {player!r}'s strategy set"
                    )
            if len(payoff_tuple) != len(self.players):
                raise PayoffError(
                    f"payoff tuple length {len(payoff_tuple)} does not match "
                    f"player count {len(self.players)}"
                )
            for value in payoff_tuple:
                if not math.isfinite(float(value)):
                    raise PayoffError(f"payoff {value!r} is not finite")

    def profiles(self) -> tuple[StrategyProfile, ...]:
        """Every profile the payoff mapping covers, in insertion-stable order."""
        pieces = [self.strategies[p] for p in self.players]
        return tuple(
            StrategyProfile(choices=tuple(combo))
            for combo in itertools.product(*pieces)
        )

    def payoff(self, profile: StrategyProfile, player: str) -> float:
        """Return ``player``'s payoff under ``profile``."""
        self._check_known(profile)
        idx = self.players.index(player)
        return float(self.payoffs[profile][idx])

    def best_responses(
        self, opponent_profile: StrategyProfile, player: str
    ) -> tuple[str, ...]:
        """Return the strategies that maximise ``player``'s payoff
        holding all other players at ``opponent_profile``. Returns
        the full set — a player may have ties at the best response.
        """
        if player not in self.players:
            raise ValueError(f"unknown player {player!r}")
        self._check_known(opponent_profile)
        player_idx = self.players.index(player)
        best_payoff = -math.inf
        winners: list[str] = []
        for strategy in self.strategies[player]:
            choices = list(opponent_profile.choices)
            choices[player_idx] = strategy
            profile = StrategyProfile(choices=tuple(choices))
            payoff = float(self.payoffs[profile][player_idx])
            if payoff > best_payoff + 1e-12:
                best_payoff = payoff
                winners = [strategy]
            elif abs(payoff - best_payoff) <= 1e-12:
                winners.append(strategy)
        return tuple(winners)

    def _check_known(self, profile: StrategyProfile) -> None:
        if profile not in self.payoffs:
            raise KeyError(f"unknown profile {profile.choices}")
