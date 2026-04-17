# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — InteractionGraph

"""Directed weighted graph of agent interactions built from a
:class:`SwarmEvent` stream.

An edge ``(src, dst, weight)`` records how many interactions
from ``src`` targeted ``dst`` inside the observed window.
Construction is O(E) and every analysis helper (density,
clustering, cycle detection) runs in O(V + E).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SwarmEvent:
    """One event in the observed trace.

    ``source`` and ``target`` are the agent identifiers involved.
    ``timestamp`` is a monotonic non-negative number — wall-clock
    seconds, a tick counter, whatever the caller uses. ``action``
    is an opaque label preserved for audit; the graph does not
    consume it.
    """

    source: str
    target: str
    timestamp: float
    action: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source must be non-empty")
        if not self.target:
            raise ValueError("target must be non-empty")
        if self.source == self.target:
            raise ValueError(f"self-interaction not allowed on {self.source!r}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative; got {self.timestamp!r}")


@dataclass
class InteractionGraph:
    """Directed weighted multigraph.

    Internal edges use a dict keyed by ``(source, target)``; the
    value is the integer weight. Node set is the union of every
    source + target seen at construction time.
    """

    _nodes: set[str] = field(default_factory=set)
    _edges: dict[tuple[str, str], int] = field(default_factory=dict)

    @classmethod
    def from_events(cls, events: Iterable[SwarmEvent]) -> InteractionGraph:
        graph = cls()
        for event in events:
            graph._add_edge(event.source, event.target)
        return graph

    def _add_edge(self, source: str, target: str) -> None:
        self._nodes.add(source)
        self._nodes.add(target)
        key = (source, target)
        self._edges[key] = self._edges.get(key, 0) + 1

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Distinct (source, target) edges. Weights are available
        via :meth:`edge_weight`."""
        return len(self._edges)

    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self._nodes))

    def edges(self) -> tuple[tuple[str, str, int], ...]:
        return tuple(
            (src, dst, weight) for (src, dst), weight in self._edges.items()
        )

    def edge_weight(self, source: str, target: str) -> int:
        return self._edges.get((source, target), 0)

    def out_neighbours(self, node: str) -> tuple[str, ...]:
        return tuple(dst for (src, dst) in self._edges if src == node)

    def out_weight(self, node: str) -> int:
        return sum(w for (src, _), w in self._edges.items() if src == node)

    def in_weight(self, node: str) -> int:
        return sum(w for (_, dst), w in self._edges.items() if dst == node)

    def density(self) -> float:
        """Edge density of the underlying simple graph — ratio of
        observed edges to the maximum possible for a directed
        graph without self-loops."""
        n = self.node_count
        if n < 2:
            return 0.0
        return self.edge_count / (n * (n - 1))

    def local_clustering(self, node: str) -> float:
        """Undirected local clustering coefficient.

        For node ``v`` with ``k`` distinct neighbours (ignoring
        edge direction), counts the fraction of triangles ``v-u-w``
        with any edge between ``u`` and ``w``.
        """
        if node not in self._nodes:
            raise KeyError(f"unknown node {node!r}")
        neighbours = {
            other
            for (src, dst) in self._edges
            for other in (src, dst)
            if other != node and node in (src, dst)
        }
        k = len(neighbours)
        if k < 2:
            return 0.0
        triangles = 0
        neighbour_list = sorted(neighbours)
        for i, u in enumerate(neighbour_list):
            for w in neighbour_list[i + 1 :]:
                if (
                    (u, w) in self._edges
                    or (w, u) in self._edges
                ):
                    triangles += 1
        return (2 * triangles) / (k * (k - 1))

    def mean_clustering(self) -> float:
        if self.node_count == 0:
            return 0.0
        return sum(self.local_clustering(n) for n in self._nodes) / self.node_count

    def has_cycle(self) -> bool:
        """DFS-based cycle detection."""
        color: dict[str, int] = {n: 0 for n in self._nodes}
        # 0 = unvisited, 1 = in-stack, 2 = finished.
        for start in self._nodes:
            if color[start] != 0:
                continue
            stack: list[tuple[str, list[str]]] = [
                (start, list(self.out_neighbours(start)))
            ]
            color[start] = 1
            while stack:
                node, remaining = stack[-1]
                if not remaining:
                    color[node] = 2
                    stack.pop()
                    continue
                nxt = remaining.pop()
                if color[nxt] == 1:
                    return True
                if color[nxt] == 0:
                    color[nxt] = 1
                    stack.append((nxt, list(self.out_neighbours(nxt))))
        return False
