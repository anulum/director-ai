# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.integrations.langgraph_swarm``.

Covers guardian edge creation, state inspection, handoff approval,
handoff rejection, quarantine propagation, SwarmGraphBuilder, and
edge cases. No LangGraph dependency — tests use mock state dicts.
"""

from __future__ import annotations

from director_ai.agentic.agent_profile import AgentProfile
from director_ai.agentic.swarm_guardian import SwarmGuardian
from director_ai.integrations.langgraph_swarm import (
    GuardianNodeResult,
    SwarmGraphBuilder,
    guardian_edge,
)


def _make_guardian() -> SwarmGuardian:
    g = SwarmGuardian(hallucination_threshold=0.5)
    g.register_agent(AgentProfile.for_role("researcher", agent_id="r1"))
    g.register_agent(AgentProfile.for_role("summariser", agent_id="s1"))
    return g


# ── Guardian edge ──────────────────────────────────────────────────────


class TestGuardianEdge:
    def test_approved_handoff(self):
        g = _make_guardian()
        edge = guardian_edge(g, "r1", "s1")
        state = {
            "context": "Paris is the capital of France",
            "messages": [{"content": "Paris is the capital of France"}],
        }
        result = edge(state)
        assert result == "s1"  # approved, forwarded to next agent

    def test_rejected_handoff(self):
        g = _make_guardian()
        g.register_agent(
            AgentProfile(
                agent_id="strict",
                role="reviewer",
                coherence_threshold=0.99,
            )
        )
        edge = guardian_edge(g, "strict", "s1")
        state = {
            "context": "completely different context xyz",
            "messages": [{"content": "unrelated gibberish abc"}],
        }
        result = edge(state)
        assert result == "__end__"

    def test_quarantines_on_reject(self):
        g = _make_guardian()
        g.register_agent(
            AgentProfile(
                agent_id="bad",
                role="researcher",
                coherence_threshold=0.99,
            )
        )
        edge = guardian_edge(g, "bad", "s1")
        state = {
            "context": "real context here",
            "messages": [{"content": "totally unrelated stuff xyz"}],
        }
        edge(state)
        assert g.is_quarantined("bad")

    def test_empty_messages(self):
        g = _make_guardian()
        edge = guardian_edge(g, "r1", "s1")
        state = {"context": "ctx", "messages": []}
        result = edge(state)
        # Empty message → neutral score → should pass
        assert isinstance(result, str)

    def test_string_messages(self):
        g = _make_guardian()
        edge = guardian_edge(g, "r1", "s1")
        state = {
            "context": "topic content here",
            "messages": ["topic content message"],
        }
        result = edge(state)
        assert result == "s1"

    def test_custom_keys(self):
        g = _make_guardian()
        edge = guardian_edge(
            g,
            "r1",
            "s1",
            context_key="ground_truth",
            message_key="output",
        )
        state = {
            "ground_truth": "fact content here",
            "output": [{"content": "fact content response"}],
        }
        result = edge(state)
        assert result == "s1"

    def test_edge_function_name(self):
        g = _make_guardian()
        edge = guardian_edge(g, "r1", "s1")
        assert "r1" in edge.__name__
        assert "s1" in edge.__name__


# ── SwarmGraphBuilder ──────────────────────────────────────────────────


class TestSwarmGraphBuilder:
    def test_add_edge(self):
        g = _make_guardian()
        builder = SwarmGraphBuilder(g)
        builder.add_guarded_edge("r1", "s1")
        assert builder.edge_count == 1

    def test_multiple_edges(self):
        g = _make_guardian()
        builder = SwarmGraphBuilder(g)
        builder.add_guarded_edge("r1", "s1")
        builder.add_guarded_edge("s1", "r1")
        assert builder.edge_count == 2

    def test_get_edges(self):
        g = _make_guardian()
        builder = SwarmGraphBuilder(g)
        builder.add_guarded_edge("r1", "s1")
        edges = builder.get_edges()
        assert len(edges) == 1
        assert edges[0]["from"] == "r1"
        assert edges[0]["to"] == "s1"
        assert callable(edges[0]["edge_fn"])

    def test_edge_fn_callable(self):
        g = _make_guardian()
        builder = SwarmGraphBuilder(g)
        builder.add_guarded_edge("r1", "s1")
        edge_fn = builder.get_edges()[0]["edge_fn"]
        state = {
            "context": "test content",
            "messages": [{"content": "test content message"}],
        }
        result = edge_fn(state)
        assert isinstance(result, str)

    def test_empty_builder(self):
        g = _make_guardian()
        builder = SwarmGraphBuilder(g)
        assert builder.edge_count == 0
        assert builder.get_edges() == []


# ── GuardianNodeResult ─────────────────────────────────────────────────


class TestGuardianNodeResult:
    def test_dataclass(self):
        r = GuardianNodeResult(
            approved=True,
            from_agent="r1",
            to_agent="s1",
            score=0.2,
        )
        assert r.approved
        assert r.score == 0.2

    def test_defaults(self):
        r = GuardianNodeResult(True, "a", "b", 0.1)
        assert r.quarantined is False
        assert r.reason == ""
