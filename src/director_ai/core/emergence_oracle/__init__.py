# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — emergent behaviour oracle

"""Detect dangerous collective patterns on a live swarm trace.

Partial agent-interaction traces go through four analysis
stages:

* :class:`InteractionGraph` — directed weighted graph built from
  a :class:`SwarmEvent` sequence. An edge ``(a, b, w)`` records
  ``w`` interactions from ``a`` to ``b`` inside the observed
  window. Exposes density, degree distribution, local clustering
  coefficient, and a fast DFS-based cycle check.
* :class:`RandomWalkSpectrum` — stationary distribution of a
  lazy random walk on the interaction graph via power iteration.
  Reports per-node stationary probability and the spectral gap
  between the first two iterates — a large probability mass
  concentrated on a small node set with high self-transition
  rate is the signature of an attractor.
* :class:`CommunityDetector` — deterministic label propagation
  to partition the graph into communities. Each node adopts the
  most frequent label among its neighbours; ties break by
  lexicographic order of the label.
* :class:`EmergenceOracle` — composes the three into a single
  :class:`EmergenceVerdict` with a ``risk`` score in ``[0, 1]``
  and named sub-signals so downstream routers can halt a swarm
  on the strongest signal rather than an opaque aggregate.
"""

from .graph import InteractionGraph, SwarmEvent
from .oracle import EmergenceOracle, EmergenceVerdict
from .spectrum import CommunityDetector, RandomWalkSpectrum

__all__ = [
    "CommunityDetector",
    "EmergenceOracle",
    "EmergenceVerdict",
    "InteractionGraph",
    "RandomWalkSpectrum",
    "SwarmEvent",
]
