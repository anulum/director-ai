# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RandomWalkSpectrum + CommunityDetector

"""Random-walk analysis and deterministic label-propagation
community detection.

The lazy random walk has transition matrix ``P = (I + D^{-1} A)
/ 2`` — half the time the walker stays put, half the time it
follows an edge with probability proportional to edge weight.
Laziness guarantees convergence on bipartite graphs and prevents
the alternating-sign oscillation that standard random walks
suffer from.

Label propagation picks the most frequent label among each
node's neighbours, breaking ties by the smallest label
lexicographically so the algorithm is reproducible across
runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .graph import InteractionGraph


@dataclass(frozen=True)
class StationaryDistribution:
    """One-row stationary distribution.

    ``probabilities`` maps node name → stationary probability.
    ``iterations`` is the number of power-iteration steps until
    the L1 residual fell below ``tolerance``. ``converged`` is
    True when the iteration finished before the cap.
    """

    probabilities: Mapping[str, float]
    iterations: int
    converged: bool
    spectral_gap: float

    def top_nodes(self, *, k: int) -> tuple[tuple[str, float], ...]:
        """Return the ``k`` nodes with the largest stationary
        probability, largest first."""
        if k <= 0:
            raise ValueError("k must be positive")
        items = sorted(self.probabilities.items(), key=lambda kv: -kv[1])
        return tuple(items[:k])


class RandomWalkSpectrum:
    """Stationary distribution of a lazy random walk on an
    :class:`InteractionGraph` via power iteration.

    Parameters
    ----------
    max_iterations :
        Upper bound on power-iteration steps. Default 256.
    tolerance :
        L1 residual at which iteration stops. Default 1e-6.
    laziness :
        Probability the walker stays at the current node each
        step. Must be in ``(0, 1)``. Default 0.5 so ``P`` is
        symmetric about the identity and the spectrum never
        oscillates.
    """

    def __init__(
        self,
        *,
        max_iterations: int = 256,
        tolerance: float = 1e-6,
        laziness: float = 0.5,
    ) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if not 0.0 < laziness < 1.0:
            raise ValueError("laziness must be in (0, 1)")
        self._max_iterations = max_iterations
        self._tolerance = tolerance
        self._laziness = laziness

    def stationary(self, graph: InteractionGraph) -> StationaryDistribution:
        nodes = graph.nodes()
        n = len(nodes)
        if n == 0:
            return StationaryDistribution(
                probabilities={},
                iterations=0,
                converged=True,
                spectral_gap=0.0,
            )
        # Row-stochastic transition matrix: P[i][j] = prob i → j.
        # Dangling nodes (zero out-weight) teleport uniformly so the
        # chain is irreducible.
        transition: list[list[float]] = []
        out_weights = [max(graph.out_weight(n_name), 0) for n_name in nodes]
        uniform = 1.0 / n
        for i, src in enumerate(nodes):
            row = [0.0] * n
            out_w = out_weights[i]
            for j, dst in enumerate(nodes):
                stay = self._laziness if i == j else 0.0
                if out_w == 0:
                    follow = (1.0 - self._laziness) * uniform
                else:
                    follow = (1.0 - self._laziness) * (
                        graph.edge_weight(src, dst) / out_w
                    )
                row[j] = stay + follow
            transition.append(row)
        distribution = [uniform] * n
        iterations = 0
        converged = False
        previous_delta = 0.0
        spectral_gap = 0.0
        for step in range(self._max_iterations):
            iterations = step + 1
            new_distribution = [0.0] * n
            for j in range(n):
                acc = 0.0
                for i in range(n):
                    acc += distribution[i] * transition[i][j]
                new_distribution[j] = acc
            delta = sum(
                abs(a - b) for a, b in zip(distribution, new_distribution, strict=False)
            )
            if previous_delta > 0:
                spectral_gap = max(0.0, 1.0 - delta / previous_delta)
            previous_delta = delta
            distribution = new_distribution
            if delta < self._tolerance:
                converged = True
                break
        total = sum(distribution)
        if total > 0:
            distribution = [p / total for p in distribution]
        probabilities = {node: distribution[i] for i, node in enumerate(nodes)}
        return StationaryDistribution(
            probabilities=probabilities,
            iterations=iterations,
            converged=converged,
            spectral_gap=spectral_gap,
        )


@dataclass(frozen=True)
class CommunityAssignment:
    """Result of one :meth:`CommunityDetector.detect` call."""

    labels: Mapping[str, str]
    iterations: int
    converged: bool

    def communities(self) -> dict[str, tuple[str, ...]]:
        """Group nodes by label — each value is a sorted tuple."""
        buckets: dict[str, list[str]] = {}
        for node, label in self.labels.items():
            buckets.setdefault(label, []).append(node)
        return {lbl: tuple(sorted(members)) for lbl, members in buckets.items()}

    @property
    def community_count(self) -> int:
        return len(set(self.labels.values()))


class CommunityDetector:
    """Deterministic asynchronous label propagation.

    Every node starts with its own label. On each pass the
    algorithm visits nodes in lexicographic order; each node
    tallies the weighted label frequency among its undirected
    neighbours (using the *latest* labels, not the
    beginning-of-round snapshot) and adopts the winner. Ties
    break first in favour of the node's current label (stable)
    and then by the smallest label lexicographically
    (deterministic). The algorithm terminates when a full pass
    leaves every label unchanged or the cap is hit.

    Asynchronous updates avoid the flip-flop that
    synchronous label propagation suffers from on bipartite-like
    interactions (e.g. two mutually-pointing agents swapping
    labels forever).

    Parameters
    ----------
    max_iterations :
        Upper bound on passes. Default 64.
    """

    def __init__(self, *, max_iterations: int = 64) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self._max_iterations = max_iterations

    def detect(self, graph: InteractionGraph) -> CommunityAssignment:
        nodes = graph.nodes()
        labels = {node: node for node in nodes}
        undirected_neighbours = _undirected_neighbours(graph)
        iterations = 0
        converged = False
        for step in range(self._max_iterations):
            iterations = step + 1
            changed = False
            for node in nodes:
                neighbours = undirected_neighbours[node]
                if not neighbours:
                    continue
                tally: dict[str, int] = {}
                for neighbour, weight in neighbours.items():
                    label = labels[neighbour]
                    tally[label] = tally.get(label, 0) + weight
                max_count = max(tally.values())
                winners = [
                    label for label, count in tally.items() if count == max_count
                ]
                current = labels[node]
                new_label = current if current in winners else min(winners)
                if new_label != current:
                    labels[node] = new_label
                    changed = True
            if not changed:
                converged = True
                break
        return CommunityAssignment(
            labels=labels, iterations=iterations, converged=converged
        )


def _undirected_neighbours(graph: InteractionGraph) -> dict[str, dict[str, int]]:
    """Build an undirected adjacency mapping: node →
    {neighbour: weight}. In- and out-edges both contribute
    weight — the community detector treats the graph as
    undirected for label propagation since direction doesn't
    identify communities."""
    result: dict[str, dict[str, int]] = {node: {} for node in graph.nodes()}
    for src, dst, weight in graph.edges():
        result[src][dst] = result[src].get(dst, 0) + weight
        result[dst][src] = result[dst].get(src, 0) + weight
    return result
