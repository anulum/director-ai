# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NashBargainingSolver

"""N-agent Nash bargaining solution on a discretised grid.

Given a total resource budget and a per-agent utility function,
the Nash bargaining solution picks the Pareto-efficient
allocation that maximises the product

    ∏_i (u_i(x_i) − d_i)

subject to ``Σ x_i ≤ budget`` and ``x_i ≥ 0`` for every agent,
where ``d_i`` is agent ``i``'s disagreement-point utility.

The solver searches a discretised grid. At ``step = 0.05`` and
budget ``B`` the grid has ``B / step + 1`` points per axis;
for N agents the enumeration count is
``C(B/step + N - 1, N - 1)`` — tractable for the small swarms
the guardrail cares about (2–5 agents).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .pool import AgentEconomicState


@dataclass(frozen=True)
class DisagreementPoint:
    """Per-agent disagreement utility — the payoff the agent
    receives when bargaining fails. Typically zero; a caller
    can tune it to the agent's best outside option."""

    agent_id: str
    utility: float

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("agent_id must be non-empty")
        if self.utility < 0:
            raise ValueError("utility must be non-negative")


@dataclass(frozen=True)
class BargainingSolution:
    """Result of one :meth:`NashBargainingSolver.solve` call."""

    allocation: Mapping[str, float]
    nash_product: float
    utilities: Mapping[str, float]
    total_allocated: float

    @property
    def fairness_gap(self) -> float:
        """Max-min utility gap normalised by the highest
        utility. Zero = perfectly equal utilities;
        approaching 1 = one agent dominates."""
        values = list(self.utilities.values())
        if not values:
            return 0.0
        top = max(values)
        bottom = min(values)
        if top <= 0:
            return 0.0
        return (top - bottom) / top


class NashBargainingSolver:
    """Grid-search Nash bargaining solver.

    Parameters
    ----------
    step :
        Grid resolution in units of the resource. Default 0.05.
    epsilon :
        Minimum positive payoff above the disagreement point —
        a standard Nash bargaining regulariser that prevents the
        product from collapsing to zero when one agent gets
        exactly their disagreement utility. Default 1e-6.
    """

    def __init__(self, *, step: float = 0.05, epsilon: float = 1e-6) -> None:
        if step <= 0:
            raise ValueError("step must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._step = step
        self._epsilon = epsilon

    def solve(
        self,
        *,
        agents: tuple[AgentEconomicState, ...],
        budget: float,
        disagreement: tuple[DisagreementPoint, ...] = (),
    ) -> BargainingSolution:
        if len(agents) < 2:
            raise ValueError("bargaining requires at least two agents")
        if budget <= 0:
            raise ValueError("budget must be positive")
        ids = [a.agent_id for a in agents]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate agent ids")
        d_map = {d.agent_id: d.utility for d in disagreement}
        unknown = set(d_map) - set(ids)
        if unknown:
            raise ValueError(f"disagreement mentions unknown agents: {sorted(unknown)}")
        grid_size = int(budget / self._step)
        if grid_size <= 0:
            raise ValueError("step larger than budget — no grid points")
        best = _BestTracker()
        for grid_alloc in _enumerate_partitions(grid_size=grid_size, count=len(agents)):
            product, allocation, utilities, total = self._score(
                grid_alloc=grid_alloc, agents=agents, d_map=d_map
            )
            best.update(
                allocation=allocation,
                utilities=utilities,
                product=product,
                total=total,
            )
        if best.allocation is None:
            raise ValueError("no feasible allocation found")
        return BargainingSolution(
            allocation=best.allocation,
            nash_product=best.product,
            utilities=best.utilities or {},
            total_allocated=best.total,
        )

    def _score(
        self,
        *,
        grid_alloc: tuple[int, ...],
        agents: tuple[AgentEconomicState, ...],
        d_map: dict[str, float],
    ) -> tuple[float, dict[str, float], dict[str, float], float]:
        allocation: dict[str, float] = {}
        utilities: dict[str, float] = {}
        product = 1.0
        total = 0.0
        for grid_units, agent in zip(grid_alloc, agents, strict=True):
            units = grid_units * self._step
            utility = agent.valuation * units
            surplus = max(self._epsilon, utility - d_map.get(agent.agent_id, 0.0))
            product *= surplus
            allocation[agent.agent_id] = units
            utilities[agent.agent_id] = utility
            total += units
        return product, allocation, utilities, total


class _BestTracker:
    """Tracks the highest-product allocation seen so far."""

    def __init__(self) -> None:
        self.product: float = -1.0
        self.allocation: dict[str, float] | None = None
        self.utilities: dict[str, float] | None = None
        self.total: float = 0.0

    def update(
        self,
        *,
        allocation: dict[str, float],
        utilities: dict[str, float],
        product: float,
        total: float,
    ) -> None:
        if product > self.product:
            self.product = product
            self.allocation = allocation
            self.utilities = utilities
            self.total = total


def _enumerate_partitions(*, grid_size: int, count: int):
    """Yield every ``count``-tuple of non-negative integers that
    sum to exactly ``grid_size``. Iterative, memory-safe for
    small counts."""
    if count == 1:
        yield (grid_size,)
        return
    for taken in range(grid_size + 1):
        for suffix in _enumerate_partitions(
            grid_size=grid_size - taken, count=count - 1
        ):
            yield (taken,) + suffix
