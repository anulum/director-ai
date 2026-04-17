# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — swarm equilibrium scorer

"""Nash / Stackelberg equilibrium analysis for small inter-agent
games.

When a multi-agent system commits to a joint action, the guard
asks two questions. First, is the observed joint profile a
(pure or mixed) Nash equilibrium of the implied game — i.e. can
any player unilaterally improve their payoff? Second, if one
agent is the leader of a Stackelberg game, is the observed
outcome the backward-induction solution, or have the followers
deviated?

* :class:`NormalFormGame` — players + per-player strategy sets
  + caller-supplied payoff matrices. Rejects malformed games at
  construction (dimension mismatch, non-finite payoffs, empty
  strategy set).
* :class:`NashSolver` — two-stage: iterated strict
  best-response to enumerate pure-strategy Nash equilibria
  (PSNE), then analytical 2x2 mixed-strategy Nash for
  two-player games when no PSNE exists. Reports the set of
  equilibria and the deviation gain for an observed profile.
* :class:`StackelbergSolver` — backward induction for two-player
  sequential games with a named leader. Returns the
  subgame-perfect equilibrium (SPE).
* :class:`SwarmEquilibriumScorer` — takes an observed
  :class:`StrategyProfile` and returns a :class:`StabilityReport`
  with the Nash / Stackelberg verdicts and a single scalar
  stability score in ``[0, 1]`` so downstream routers can halt
  swarms drifting away from their own equilibria.
"""

from .game import NormalFormGame, PayoffError, StrategyProfile
from .scorer import StabilityReport, SwarmEquilibriumScorer
from .solvers import NashEquilibrium, NashSolver, StackelbergSolver

__all__ = [
    "NashEquilibrium",
    "NashSolver",
    "NormalFormGame",
    "PayoffError",
    "StabilityReport",
    "StackelbergSolver",
    "StrategyProfile",
    "SwarmEquilibriumScorer",
]
