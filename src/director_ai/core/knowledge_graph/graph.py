# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — KnowledgeGraph

"""Typed directed graph of skills + sanctioned transitions.

Nodes are :class:`SkillNode` instances — capabilities, metadata,
tenant scope. Edges are :class:`SkillEdge` — typed transition
(``TraversalAction``) with weight + policy. The graph exposes a
policy-aware Dijkstra (:meth:`shortest_sanctioned_path`) so
callers can plan a traversal the principal is guaranteed to be
allowed to take.

Cycle detection is opt-in — some skill graphs (retry, loopback)
need cycles. Callers who forbid them call
:meth:`require_acyclic` after construction.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass, field

from .policy import Principal, TraversalAction, TraversalPolicy


class KnowledgeGraphCycleError(ValueError):
    """Raised by :meth:`KnowledgeGraph.require_acyclic` when the
    graph contains one or more cycles."""


@dataclass(frozen=True)
class SkillNode:
    """One skill the agent can invoke.

    ``id`` — stable handle used everywhere else in the graph.
    ``capabilities`` — unordered set of capability tokens the
    skill provides (e.g. ``"retrieve"``, ``"synthesise"``,
    ``"write_file"``). Consumers filter skills by capability.
    ``tenant_id`` — empty means "shared"; non-empty scopes the
    skill to one tenant.
    """

    id: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    tenant_id: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SkillNode.id must be non-empty")


@dataclass(frozen=True)
class SkillEdge:
    """A sanctioned transition between two skills.

    ``source`` / ``target`` — the connected :class:`SkillNode` IDs.
    ``action`` — which :data:`TraversalAction` this edge represents.
    ``weight`` — non-negative cost for shortest-path searches.
    Default 1.0 — callers set higher weights on expensive or
    risky transitions (model hops, human escalations).
    ``policy`` — the :class:`TraversalPolicy` enforced on this
    edge. Default: :meth:`TraversalPolicy.allow_all`.
    ``tenant_id`` — edge-level tenant scope used for
    ``require_same_tenant`` policies.
    """

    source: str
    target: str
    action: TraversalAction
    weight: float = 1.0
    policy: TraversalPolicy = field(default_factory=TraversalPolicy.allow_all)
    tenant_id: str = ""

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("SkillEdge source/target must be non-empty")
        if self.source == self.target:
            raise ValueError(f"SkillEdge self-loop on {self.source!r}")
        if self.weight < 0:
            raise ValueError(f"SkillEdge.weight must be non-negative; got {self.weight!r}")


@dataclass
class KnowledgeGraph:
    """In-memory knowledge graph.

    Not thread-safe on writes — callers who build the graph once
    and share it read-only are fine; concurrent mutation requires
    external synchronisation.
    """

    _nodes: dict[str, SkillNode] = field(default_factory=dict)
    _edges_by_source: dict[str, list[SkillEdge]] = field(default_factory=dict)
    _all_edges: list[SkillEdge] = field(default_factory=list)

    def add_node(self, node: SkillNode) -> None:
        if node.id in self._nodes:
            raise ValueError(f"duplicate skill {node.id!r}")
        self._nodes[node.id] = node
        self._edges_by_source.setdefault(node.id, [])

    def add_edge(self, edge: SkillEdge) -> None:
        if edge.source not in self._nodes:
            raise ValueError(f"unknown source skill {edge.source!r}")
        if edge.target not in self._nodes:
            raise ValueError(f"unknown target skill {edge.target!r}")
        # Forbid multi-edges with the same (target, action) to keep
        # shortest-path semantics clean. Callers who need several
        # parallel edges compose actions.
        for existing in self._edges_by_source[edge.source]:
            if existing.target == edge.target and existing.action == edge.action:
                raise ValueError(
                    f"duplicate edge {edge.source!r} -> {edge.target!r} "
                    f"with action {edge.action!r}"
                )
        self._edges_by_source[edge.source].append(edge)
        self._all_edges.append(edge)

    def node(self, node_id: str) -> SkillNode:
        if node_id not in self._nodes:
            raise KeyError(f"unknown skill {node_id!r}")
        return self._nodes[node_id]

    def nodes(self) -> tuple[SkillNode, ...]:
        return tuple(self._nodes.values())

    def edges(self) -> tuple[SkillEdge, ...]:
        return tuple(self._all_edges)

    def outgoing(self, source: str) -> tuple[SkillEdge, ...]:
        return tuple(self._edges_by_source.get(source, ()))

    def skills_with_capability(self, capability: str) -> tuple[SkillNode, ...]:
        return tuple(n for n in self._nodes.values() if capability in n.capabilities)

    def shortest_sanctioned_path(
        self,
        *,
        source: str,
        target: str,
        principal: Principal,
        action: TraversalAction = "invoke",
    ) -> tuple[SkillEdge, ...]:
        """Return the lowest-weight edge sequence from ``source``
        to ``target`` that the ``principal`` may traverse under
        ``action``. Raises :class:`KeyError` when either node is
        missing; :class:`ValueError` when no sanctioned path exists.
        """
        if source not in self._nodes:
            raise KeyError(f"unknown source skill {source!r}")
        if target not in self._nodes:
            raise KeyError(f"unknown target skill {target!r}")
        # (distance, insertion_order, node, path_edges)
        counter = 0
        start_state: tuple[float, int, str, tuple[SkillEdge, ...]] = (
            0.0,
            counter,
            source,
            (),
        )
        heap: list[tuple[float, int, str, tuple[SkillEdge, ...]]] = [start_state]
        best: dict[str, float] = {source: 0.0}
        while heap:
            dist, _, current, path = heapq.heappop(heap)
            if current == target:
                return path
            if dist > best.get(current, float("inf")):
                continue
            for edge in self._edges_by_source.get(current, ()):
                allowed, _ = edge.policy.check(
                    principal, action=action, edge_tenant_id=edge.tenant_id
                )
                if not allowed:
                    continue
                new_dist = dist + edge.weight
                if new_dist < best.get(edge.target, float("inf")):
                    best[edge.target] = new_dist
                    counter += 1
                    heapq.heappush(
                        heap,
                        (new_dist, counter, edge.target, path + (edge,)),
                    )
        raise ValueError(
            f"no sanctioned path from {source!r} to {target!r} "
            f"for principal role={principal.role!r}"
        )

    def require_acyclic(self) -> None:
        """Raise :class:`KnowledgeGraphCycleError` if the directed
        graph has a cycle. Uses Kahn's algorithm — O(V + E)."""
        indeg: dict[str, int] = {n: 0 for n in self._nodes}
        for edge in self._all_edges:
            indeg[edge.target] = indeg.get(edge.target, 0) + 1
        ready = [n for n, d in indeg.items() if d == 0]
        visited = 0
        while ready:
            current = ready.pop(0)
            visited += 1
            for edge in self._edges_by_source.get(current, ()):
                indeg[edge.target] -= 1
                if indeg[edge.target] == 0:
                    ready.append(edge.target)
        if visited != len(self._nodes):
            raise KnowledgeGraphCycleError(
                f"cycle detected; only {visited}/{len(self._nodes)} nodes reachable "
                f"in topological order"
            )

    def merge(self, others: Iterable[KnowledgeGraph]) -> None:
        """Merge the ``others`` into ``self`` in place. Duplicate
        nodes with the same ID raise :class:`ValueError`; duplicate
        edges are silently deduplicated."""
        for other in others:
            for node in other.nodes():
                self.add_node(node)
            for edge in other.edges():
                try:
                    self.add_edge(edge)
                except ValueError as exc:
                    if "duplicate edge" in str(exc):
                        continue
                    raise
