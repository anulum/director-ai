# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — institutional knowledge graph

"""Directed graph of skills + sanctioned transitions between them.

An agent holds a :class:`Principal` — role, permissions, and
optional tenant scoping. Every edge in the :class:`KnowledgeGraph`
carries a :class:`TraversalPolicy` that states which principals
may take the transition. :class:`TraversalValidator` walks a
proposed path and returns a :class:`TraversalVerdict` with the
first denial (if any) plus the reason, so the calling agent can
surface an actionable error.

Shortest-path search (:meth:`KnowledgeGraph.shortest_sanctioned_path`)
is policy-aware: Dijkstra over edge weights with policy pre-filter,
so the returned path is always traversable by the principal.
"""

from .graph import KnowledgeGraph, KnowledgeGraphCycleError, SkillEdge, SkillNode
from .policy import Principal, TraversalAction, TraversalPolicy
from .validator import TraversalStep, TraversalValidator, TraversalVerdict

__all__ = [
    "KnowledgeGraph",
    "KnowledgeGraphCycleError",
    "Principal",
    "SkillEdge",
    "SkillNode",
    "TraversalAction",
    "TraversalPolicy",
    "TraversalStep",
    "TraversalValidator",
    "TraversalVerdict",
]
