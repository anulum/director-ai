# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — knowledge graph tests

"""Multi-angle coverage: SkillNode / SkillEdge validation,
Principal + TraversalPolicy gates, TraversalPolicy.merge
semantics, KnowledgeGraph adds + duplicates + shortest-path,
Dijkstra policy-aware skipping, acyclic check, merge of
subgraphs, and TraversalValidator verdicts including contiguity
and graph-level default_policy."""

from __future__ import annotations

from typing import Any, cast

import pytest

from director_ai.core.knowledge_graph import (
    KnowledgeGraph,
    KnowledgeGraphCycleError,
    Principal,
    SkillEdge,
    SkillNode,
    TraversalPolicy,
    TraversalStep,
    TraversalValidator,
)

# --- Principal ------------------------------------------------------


class TestPrincipal:
    def test_valid(self):
        p = Principal(role="admin", permissions=frozenset({"write"}))
        assert p.has("write")

    def test_empty_role_rejected(self):
        with pytest.raises(ValueError, match="role"):
            Principal(role="")


# --- TraversalPolicy ------------------------------------------------


class TestPolicy:
    def test_allow_all_accepts_any_principal(self):
        p = Principal(role="user")
        allowed, _ = TraversalPolicy.allow_all().check(p, action="invoke")
        assert allowed

    def test_role_restriction(self):
        policy = TraversalPolicy(allowed_roles=frozenset({"admin"}))
        user = Principal(role="user")
        admin = Principal(role="admin")
        assert not policy.check(user, action="invoke")[0]
        assert policy.check(admin, action="invoke")[0]

    def test_permission_requirement(self):
        policy = TraversalPolicy(required_permissions=frozenset({"write"}))
        p_no = Principal(role="a")
        p_yes = Principal(role="a", permissions=frozenset({"write"}))
        assert not policy.check(p_no, action="invoke")[0]
        assert policy.check(p_yes, action="invoke")[0]

    def test_tenant_required(self):
        policy = TraversalPolicy(require_same_tenant=True)
        p = Principal(role="a", tenant_id="t1")
        ok, _ = policy.check(p, action="invoke", edge_tenant_id="t1")
        assert ok
        ok, _ = policy.check(p, action="invoke", edge_tenant_id="t2")
        assert not ok

    def test_action_restriction(self):
        policy = TraversalPolicy(allowed_actions=frozenset({"delegate", "escalate"}))
        p = Principal(role="a")
        assert policy.check(p, action="delegate")[0]
        assert not policy.check(p, action="invoke")[0]

    def test_invalid_action_in_check(self):
        policy = TraversalPolicy.allow_all()
        p = Principal(role="a")
        bad = cast(Any, "freeform")
        ok, _ = policy.check(p, action=bad)
        assert not ok

    def test_invalid_action_in_policy(self):
        bad = cast(Any, frozenset({"bogus"}))
        with pytest.raises(ValueError, match="allowed_actions"):
            TraversalPolicy(allowed_actions=bad)

    def test_merge_intersects_roles(self):
        a = TraversalPolicy(allowed_roles=frozenset({"admin", "audit"}))
        b = TraversalPolicy(allowed_roles=frozenset({"admin", "tool"}))
        merged = a.merge(b)
        assert merged.allowed_roles == frozenset({"admin"})

    def test_merge_unions_permissions(self):
        a = TraversalPolicy(required_permissions=frozenset({"read"}))
        b = TraversalPolicy(required_permissions=frozenset({"write"}))
        merged = a.merge(b)
        assert merged.required_permissions == frozenset({"read", "write"})

    def test_merge_empty_roles_means_any(self):
        a = TraversalPolicy.allow_all()
        b = TraversalPolicy(allowed_roles=frozenset({"admin"}))
        merged = a.merge(b)
        assert merged.allowed_roles == frozenset({"admin"})

    def test_merge_tenant_is_or(self):
        a = TraversalPolicy(require_same_tenant=False)
        b = TraversalPolicy(require_same_tenant=True)
        assert a.merge(b).require_same_tenant


# --- SkillNode / SkillEdge ------------------------------------------


class TestNodesEdges:
    def test_valid_node(self):
        n = SkillNode(id="s1", capabilities=frozenset({"retrieve"}))
        assert "retrieve" in n.capabilities

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="id"):
            SkillNode(id="")

    def test_valid_edge(self):
        e = SkillEdge(source="a", target="b", action="invoke")
        assert e.weight == 1.0

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loop"):
            SkillEdge(source="a", target="a", action="invoke")

    def test_negative_weight(self):
        with pytest.raises(ValueError, match="weight"):
            SkillEdge(source="a", target="b", action="invoke", weight=-1.0)


# --- KnowledgeGraph -------------------------------------------------


class TestGraph:
    def _graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        for name in ("a", "b", "c", "d"):
            g.add_node(SkillNode(id=name))
        g.add_edge(SkillEdge(source="a", target="b", action="invoke", weight=1))
        g.add_edge(SkillEdge(source="b", target="c", action="invoke", weight=1))
        g.add_edge(SkillEdge(source="a", target="c", action="invoke", weight=3))
        g.add_edge(SkillEdge(source="c", target="d", action="invoke", weight=1))
        return g

    def test_add_and_retrieve(self):
        g = self._graph()
        assert g.node("a").id == "a"
        assert len(g.edges()) == 4

    def test_duplicate_node_rejected(self):
        g = self._graph()
        with pytest.raises(ValueError, match="duplicate skill"):
            g.add_node(SkillNode(id="a"))

    def test_unknown_endpoint_rejected(self):
        g = self._graph()
        with pytest.raises(ValueError, match="unknown source"):
            g.add_edge(SkillEdge(source="z", target="a", action="invoke"))

    def test_duplicate_edge_rejected(self):
        g = self._graph()
        with pytest.raises(ValueError, match="duplicate edge"):
            g.add_edge(SkillEdge(source="a", target="b", action="invoke"))

    def test_unknown_node_lookup(self):
        g = self._graph()
        with pytest.raises(KeyError):
            g.node("z")

    def test_capability_index(self):
        g = self._graph()
        g.add_node(SkillNode(id="retriever", capabilities=frozenset({"rag"})))
        results = g.skills_with_capability("rag")
        assert len(results) == 1 and results[0].id == "retriever"

    def test_shortest_path_picks_two_hop(self):
        g = self._graph()
        principal = Principal(role="u")
        path = g.shortest_sanctioned_path(source="a", target="c", principal=principal)
        # a -> b -> c (cost 2) beats a -> c (cost 3).
        assert [e.target for e in path] == ["b", "c"]

    def test_shortest_path_skips_denied(self):
        g = KnowledgeGraph()
        for name in ("a", "b", "c"):
            g.add_node(SkillNode(id=name))
        g.add_edge(
            SkillEdge(
                source="a",
                target="b",
                action="invoke",
                policy=TraversalPolicy(allowed_roles=frozenset({"admin"})),
            )
        )
        g.add_edge(SkillEdge(source="a", target="c", action="invoke", weight=5))
        g.add_edge(SkillEdge(source="c", target="b", action="invoke"))
        principal = Principal(role="user")
        path = g.shortest_sanctioned_path(source="a", target="b", principal=principal)
        # The cheap a->b edge is admin-only; user must go a->c->b.
        assert [e.target for e in path] == ["c", "b"]

    def test_no_path_raises(self):
        g = KnowledgeGraph()
        g.add_node(SkillNode(id="a"))
        g.add_node(SkillNode(id="b"))
        with pytest.raises(ValueError, match="no sanctioned path"):
            g.shortest_sanctioned_path(
                source="a", target="b", principal=Principal(role="u")
            )

    def test_unknown_endpoint_in_path(self):
        g = self._graph()
        with pytest.raises(KeyError):
            g.shortest_sanctioned_path(
                source="z", target="a", principal=Principal(role="u")
            )

    def test_require_acyclic_on_dag(self):
        g = self._graph()
        g.require_acyclic()  # must not raise

    def test_require_acyclic_on_cycle(self):
        g = KnowledgeGraph()
        for name in ("a", "b"):
            g.add_node(SkillNode(id=name))
        g.add_edge(SkillEdge(source="a", target="b", action="invoke"))
        g.add_edge(SkillEdge(source="b", target="a", action="invoke"))
        with pytest.raises(KnowledgeGraphCycleError):
            g.require_acyclic()

    def test_merge_subgraphs(self):
        a = KnowledgeGraph()
        a.add_node(SkillNode(id="s1"))
        b = KnowledgeGraph()
        b.add_node(SkillNode(id="s2"))
        a.merge([b])
        assert {n.id for n in a.nodes()} == {"s1", "s2"}


# --- TraversalValidator --------------------------------------------


class TestValidator:
    def _graph(self) -> KnowledgeGraph:
        g = KnowledgeGraph()
        for name in ("plan", "retrieve", "answer"):
            g.add_node(SkillNode(id=name))
        g.add_edge(SkillEdge(source="plan", target="retrieve", action="delegate"))
        g.add_edge(
            SkillEdge(
                source="retrieve",
                target="answer",
                action="compose",
                policy=TraversalPolicy(required_permissions=frozenset({"respond"})),
            )
        )
        return g

    def test_approves_sanctioned_path(self):
        g = self._graph()
        v = TraversalValidator(graph=g)
        principal = Principal(role="agent", permissions=frozenset({"respond"}))
        verdict = v.validate(
            steps=[
                TraversalStep("plan", "retrieve", "delegate"),
                TraversalStep("retrieve", "answer", "compose"),
            ],
            principal=principal,
        )
        assert verdict.allowed
        assert len(verdict.edges) == 2

    def test_denies_missing_permission(self):
        g = self._graph()
        v = TraversalValidator(graph=g)
        principal = Principal(role="agent")
        verdict = v.validate(
            steps=[
                TraversalStep("plan", "retrieve", "delegate"),
                TraversalStep("retrieve", "answer", "compose"),
            ],
            principal=principal,
        )
        assert not verdict.allowed
        assert verdict.denied_step_index == 1
        assert "respond" in verdict.reason

    def test_denies_unknown_edge(self):
        g = self._graph()
        v = TraversalValidator(graph=g)
        principal = Principal(role="agent", permissions=frozenset({"respond"}))
        verdict = v.validate(
            steps=[TraversalStep("plan", "answer", "invoke")],
            principal=principal,
        )
        assert not verdict.allowed
        assert "no edge" in verdict.reason

    def test_non_contiguous_path(self):
        g = self._graph()
        v = TraversalValidator(graph=g)
        principal = Principal(role="agent", permissions=frozenset({"respond"}))
        verdict = v.validate(
            steps=[
                TraversalStep("plan", "retrieve", "delegate"),
                TraversalStep("plan", "answer", "compose"),  # discontinuous
            ],
            principal=principal,
        )
        assert not verdict.allowed
        assert "contiguous" in verdict.reason

    def test_default_policy_applied(self):
        g = self._graph()
        v = TraversalValidator(
            graph=g,
            default_policy=TraversalPolicy(
                required_permissions=frozenset({"audit:trace"})
            ),
        )
        principal = Principal(role="agent", permissions=frozenset({"respond"}))
        verdict = v.validate(
            steps=[TraversalStep("plan", "retrieve", "delegate")],
            principal=principal,
        )
        assert not verdict.allowed
        assert "audit:trace" in verdict.reason

    def test_empty_steps(self):
        g = self._graph()
        v = TraversalValidator(graph=g)
        verdict = v.validate(steps=[], principal=Principal(role="u"))
        assert not verdict.allowed
        assert "empty" in verdict.reason
