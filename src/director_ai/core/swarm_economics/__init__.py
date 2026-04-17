# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — swarm economic risk scorer

"""Inter-agent resource economics and tragedy-of-the-commons
detection.

When multiple agents share a bounded resource (LLM tokens, GPU
minutes, database IO), individually rational consumption can
collectively exhaust the pool. This package turns the classical
Hardin / Nash bargaining machinery into a live risk signal the
guardrail can act on.

* :class:`ResourcePool` — the shared resource with capacity,
  regeneration rate, and an append-only consumption ledger.
* :class:`AgentEconomicState` — per-agent snapshot: credit
  balance, running consumption, declared valuation of the
  resource.
* :class:`NashBargainingSolver` — solves the standard N-agent
  Nash bargaining problem on a discretised allocation grid and
  returns the Pareto-efficient allocation that maximises the
  product of agent payoffs above their disagreement point.
* :class:`TragedyDetector` — detects runaway consumption by
  comparing aggregate draw against the pool's sustainable rate,
  with a caller-tunable over-consumption grace factor.
* :class:`EconomicRiskScorer` — composes the three into a
  single :class:`EconomicVerdict` with ``risk`` in ``[0, 1]``
  plus named sub-signals (exhaustion_headroom, fairness_gap,
  tragedy_pressure).
"""

from .bargaining import (
    BargainingSolution,
    DisagreementPoint,
    NashBargainingSolver,
)
from .detector import TragedyDetector, TragedySignal
from .pool import (
    AgentEconomicState,
    ConsumptionRecord,
    PoolError,
    ResourcePool,
)
from .scorer import EconomicRiskScorer, EconomicVerdict

__all__ = [
    "AgentEconomicState",
    "BargainingSolution",
    "ConsumptionRecord",
    "DisagreementPoint",
    "EconomicRiskScorer",
    "EconomicVerdict",
    "NashBargainingSolver",
    "PoolError",
    "ResourcePool",
    "TragedyDetector",
    "TragedySignal",
]
