# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TraversalValidator

"""Walk a caller-proposed :class:`TraversalStep` sequence and
return a verdict.

Unlike :meth:`KnowledgeGraph.shortest_sanctioned_path` which
*plans* a sanctioned path, this validator *checks* a path that
has already been proposed — typically by an actor who wants to
commit to a concrete plan. The verdict either approves every
step or reports the first denial with its reason.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .graph import KnowledgeGraph, SkillEdge
from .policy import Principal, TraversalAction, TraversalPolicy


@dataclass(frozen=True)
class TraversalStep:
    """One caller-proposed transition.

    ``source`` and ``target`` name the nodes; ``action`` is the
    :data:`TraversalAction` requested. The validator looks up
    the matching :class:`SkillEdge` in the graph and runs its
    policy against the principal.
    """

    source: str
    target: str
    action: TraversalAction


@dataclass(frozen=True)
class TraversalVerdict:
    """Result of a :meth:`TraversalValidator.validate` call.

    ``allowed`` is ``True`` only when every step is sanctioned.
    ``denied_step_index`` is the 0-based index of the first
    denial (or ``None`` when allowed). ``reason`` holds the
    policy message from the denial.
    """

    allowed: bool
    reason: str
    denied_step_index: int | None = None
    edges: tuple[SkillEdge, ...] = field(default_factory=tuple)


class TraversalValidator:
    """Policy-aware walker over a caller-proposed path.

    Parameters
    ----------
    graph :
        The :class:`KnowledgeGraph` that owns the nodes and edges.
    default_policy :
        Optional graph-level policy merged with every edge's
        policy (via :meth:`TraversalPolicy.merge`). Use this to
        pin a hard floor such as "every traversal requires
        ``audit:trace``".
    """

    def __init__(
        self,
        *,
        graph: KnowledgeGraph,
        default_policy: TraversalPolicy | None = None,
    ) -> None:
        self._graph = graph
        self._default_policy = default_policy

    def validate(
        self,
        *,
        steps: Iterable[TraversalStep],
        principal: Principal,
    ) -> TraversalVerdict:
        step_list = list(steps)
        if not step_list:
            return TraversalVerdict(
                allowed=False,
                reason="empty step list — nothing to validate",
            )
        resolved: list[SkillEdge] = []
        for i, step in enumerate(step_list):
            if i > 0 and step.source != step_list[i - 1].target:
                return TraversalVerdict(
                    allowed=False,
                    reason=(
                        f"path is not contiguous: step {i - 1} ends at "
                        f"{step_list[i - 1].target!r}, step {i} starts at "
                        f"{step.source!r}"
                    ),
                    denied_step_index=i,
                    edges=tuple(resolved),
                )
            edge = self._find_edge(step)
            if edge is None:
                return TraversalVerdict(
                    allowed=False,
                    reason=(
                        f"no edge {step.source!r} -> {step.target!r} "
                        f"with action {step.action!r}"
                    ),
                    denied_step_index=i,
                    edges=tuple(resolved),
                )
            policy = edge.policy
            if self._default_policy is not None:
                policy = policy.merge(self._default_policy)
            allowed, reason = policy.check(
                principal, action=step.action, edge_tenant_id=edge.tenant_id
            )
            if not allowed:
                return TraversalVerdict(
                    allowed=False,
                    reason=reason,
                    denied_step_index=i,
                    edges=tuple(resolved),
                )
            resolved.append(edge)
        return TraversalVerdict(
            allowed=True, reason="all steps sanctioned", edges=tuple(resolved)
        )

    def _find_edge(self, step: TraversalStep) -> SkillEdge | None:
        for edge in self._graph.outgoing(step.source):
            if edge.target == step.target and edge.action == step.action:
                return edge
        return None
