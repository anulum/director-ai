# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CausalGraph

"""Lightweight structural-causal-model DAG.

Each variable has a structural equation: a callable that takes a
mapping from parent-variable name to value and returns the
variable's value. The graph rejects cycles at registration time so
evaluation can run a pre-computed topological order.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

# A structural equation takes parent values (by variable name) and
# returns this variable's value. Values are typed as object so the
# framework supports booleans, floats, strings — typing discipline is
# the caller's responsibility.
StructuralEq = Callable[[Mapping[str, object]], object]


class GraphCycleError(ValueError):
    """Raised when adding an edge would create a cycle. Carries the
    offending edge for operator-facing error messages."""


@dataclass(frozen=True)
class _Node:
    name: str
    parents: tuple[str, ...]
    equation: StructuralEq


@dataclass
class CausalGraph:
    """DAG of variables with structural equations.

    Build with ``.add(name, equation, parents=...)``. Freeze by
    calling ``.topological_order()`` once — subsequent mutations
    invalidate the cached order.
    """

    _nodes: dict[str, _Node] = field(default_factory=dict)
    _topo_cache: tuple[str, ...] | None = field(default=None, repr=False)

    def add(
        self,
        name: str,
        equation: StructuralEq,
        *,
        parents: tuple[str, ...] = (),
    ) -> None:
        """Register ``name`` with its parents and structural equation.

        Raises :class:`ValueError` on duplicate names, unknown
        parents, or self-loops; :class:`GraphCycleError` when the
        edge would create a cycle.
        """
        if not name:
            raise ValueError("variable name must be non-empty")
        if name in self._nodes:
            raise ValueError(f"duplicate variable {name!r}")
        if name in parents:
            raise ValueError(f"self-loop on {name!r}")
        for p in parents:
            if p not in self._nodes:
                raise ValueError(f"unknown parent {p!r} for {name!r}")
        self._nodes[name] = _Node(name=name, parents=tuple(parents), equation=equation)
        self._topo_cache = None
        # The new node only adds outgoing edges from existing parents
        # into itself, so no new cycle can appear — the pre-add check
        # for ``name in parents`` already blocks the only way the new
        # node could participate in a cycle.

    def variables(self) -> tuple[str, ...]:
        """Snapshot of every registered variable name (insertion order)."""
        return tuple(self._nodes.keys())

    def parents(self, name: str) -> tuple[str, ...]:
        return self._nodes[name].parents

    def equation(self, name: str) -> StructuralEq:
        return self._nodes[name].equation

    def topological_order(self) -> tuple[str, ...]:
        """Kahn's algorithm; cached until the next :meth:`add` call.

        Raises :class:`GraphCycleError` when the graph contains a
        cycle despite the edge-time checks (defensive — reachable
        only if a caller mutates the private attributes).
        """
        if self._topo_cache is not None:
            return self._topo_cache
        indeg = {n: len(self._nodes[n].parents) for n in self._nodes}
        ready = [n for n, d in indeg.items() if d == 0]
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            # Find children of n by scanning nodes whose parents include n.
            for m, node in self._nodes.items():
                if n in node.parents:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        ready.append(m)
        if len(order) != len(self._nodes):
            missing = tuple(n for n in self._nodes if n not in order)
            raise GraphCycleError(f"cycle detected involving {missing}")
        self._topo_cache = tuple(order)
        return self._topo_cache

    def evaluate(self, inputs: Mapping[str, object]) -> dict[str, object]:
        """Run every structural equation in topological order.

        ``inputs`` provides exogenous variables (nodes without
        parents). Each equation sees a read-only view of already
        computed values.
        """
        values: dict[str, object] = dict(inputs)
        for name in self.topological_order():
            node = self._nodes[name]
            if not node.parents:
                if name not in values:
                    # Exogenous node with no input — evaluate with empty mapping.
                    values[name] = node.equation({})
                continue
            parent_values = {p: values[p] for p in node.parents}
            values[name] = node.equation(parent_values)
        return values
