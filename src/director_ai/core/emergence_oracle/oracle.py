# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — EmergenceOracle

"""Composite risk score over a :class:`SwarmEvent` trace.

The oracle runs three analyses in parallel and collapses the
results into a single :class:`EmergenceVerdict`:

1. **Attractor mass** — does a small node set hold a
   disproportionate fraction of the lazy random walk's
   stationary probability? High mass = the swarm is funnelling
   into a few hub agents.
2. **Cycle dominance** — is there at least one directed cycle
   in the interaction graph? Cycles that keep regenerating the
   same interactions are a signature of runaway dynamics.
3. **Community imbalance** — do a few communities dominate the
   node count? A single community that holds most nodes is a
   normal collaboration pattern; two or three nearly-equal
   communities that never interact are a fragmentation signal.

The composite ``risk`` score in ``[0, 1]`` is the operator-
configurable weighted sum of the three signals so callers who
only care about one dimension can zero the others.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from .graph import InteractionGraph, SwarmEvent
from .spectrum import (
    CommunityAssignment,
    CommunityDetector,
    RandomWalkSpectrum,
    StationaryDistribution,
)


@dataclass(frozen=True)
class EmergenceVerdict:
    """One ``EmergenceOracle.analyse`` outcome.

    Every sub-signal is exposed so callers can make fine-grained
    routing decisions instead of trusting the composite score.
    """

    risk: float
    attractor_mass: float
    cycle_detected: bool
    community_imbalance: float
    stationary: StationaryDistribution
    communities: CommunityAssignment
    top_hubs: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    @property
    def safe(self) -> bool:
        return self.risk < 0.5


class EmergenceOracle:
    """Compose InteractionGraph + RandomWalkSpectrum + CommunityDetector.

    Parameters
    ----------
    spectrum :
        Injected :class:`RandomWalkSpectrum`. Defaults to a fresh
        one with the default power-iteration knobs.
    communities :
        Injected :class:`CommunityDetector`. Default as above.
    attractor_top_k :
        How many nodes count toward ``attractor_mass``. Default
        3 — small hubs dominate the metric.
    weight_attractor :
        Composite weight on the attractor channel. Default 0.5.
    weight_cycle :
        Composite weight on the cycle channel. Default 0.2.
    weight_imbalance :
        Composite weight on the community-imbalance channel.
        Default 0.3. Weights must be non-negative and sum to 1.
    """

    def __init__(
        self,
        *,
        spectrum: RandomWalkSpectrum | None = None,
        communities: CommunityDetector | None = None,
        attractor_top_k: int = 3,
        weight_attractor: float = 0.5,
        weight_cycle: float = 0.2,
        weight_imbalance: float = 0.3,
    ) -> None:
        if attractor_top_k <= 0:
            raise ValueError("attractor_top_k must be positive")
        weights = (weight_attractor, weight_cycle, weight_imbalance)
        for w in weights:
            if w < 0:
                raise ValueError("weights must be non-negative")
        total = sum(weights)
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"weights must sum to 1.0; got {total}")
        self._spectrum = spectrum or RandomWalkSpectrum()
        self._communities = communities or CommunityDetector()
        self._top_k = attractor_top_k
        self._weight_attractor = weight_attractor
        self._weight_cycle = weight_cycle
        self._weight_imbalance = weight_imbalance

    def analyse(self, events: Iterable[SwarmEvent]) -> EmergenceVerdict:
        graph = InteractionGraph.from_events(events)
        stationary = self._spectrum.stationary(graph)
        communities = self._communities.detect(graph)
        cycle_detected = graph.has_cycle()
        if graph.node_count == 0:
            return EmergenceVerdict(
                risk=0.0,
                attractor_mass=0.0,
                cycle_detected=False,
                community_imbalance=0.0,
                stationary=stationary,
                communities=communities,
                top_hubs=(),
            )
        top_hubs = stationary.top_nodes(k=min(self._top_k, graph.node_count))
        attractor_mass = sum(p for _, p in top_hubs)
        # Normalised deviation above a uniform expectation — so a
        # uniform distribution returns 0 and full concentration
        # returns ~1.
        expected_mass = len(top_hubs) / graph.node_count
        attractor_excess = max(0.0, attractor_mass - expected_mass) / max(
            1.0 - expected_mass, 1e-9
        )
        imbalance = _community_imbalance(communities, graph)
        risk = (
            self._weight_attractor * attractor_excess
            + self._weight_cycle * (1.0 if cycle_detected else 0.0)
            + self._weight_imbalance * imbalance
        )
        risk = max(0.0, min(1.0, risk))
        return EmergenceVerdict(
            risk=risk,
            attractor_mass=attractor_excess,
            cycle_detected=cycle_detected,
            community_imbalance=imbalance,
            stationary=stationary,
            communities=communities,
            top_hubs=top_hubs,
        )


def _community_imbalance(
    communities: CommunityAssignment, graph: InteractionGraph
) -> float:
    """Gini-style imbalance of community sizes.

    Returns 0 when every community has the same size, ~1 when a
    single community dominates. Uses normalised entropy deficit
    so the metric is comparable across graphs.
    """
    n = graph.node_count
    buckets = communities.communities()
    if n == 0 or len(buckets) <= 1:
        return 0.0
    sizes = [len(members) for members in buckets.values()]
    probs = [size / n for size in sizes]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(buckets))
    if max_entropy <= 0:
        return 0.0
    return max(0.0, 1.0 - entropy / max_entropy)
