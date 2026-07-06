# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - LangGraph integration real-surface tests
"""Production-surface coverage for the LangGraph adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict, cast

import pytest
from langgraph.graph import END, START, StateGraph

from director_ai.integrations.langgraph import (
    director_ai_conditional_edge,
    director_ai_node,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _AgentState(TypedDict, total=False):
    """State shape used by the real LangGraph smoke graph."""

    query: str
    response: str
    messages: list[dict[str, str]]
    route: str
    director_ai_score: float
    director_ai_approved: bool
    director_ai_h_logical: float
    director_ai_h_factual: float


def test_langgraph_unit_guard_has_real_surface_companion() -> None:
    """The LangGraph unit guard should declare this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_langgraph_integration.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_langgraph_integration_real_surface.py" in category


def test_public_langgraph_adapter_routes_compiled_graph() -> None:
    """A compiled LangGraph should route through the public guard node."""

    def generate(_state: _AgentState) -> _AgentState:
        return {
            "query": "What is the refund window?",
            "response": "Refunds are available within 30 days.",
        }

    def output(state: _AgentState) -> _AgentState:
        return {**state, "route": "output"}

    def retry(state: _AgentState) -> _AgentState:
        return {**state, "route": "retry"}

    graph = StateGraph(_AgentState)
    graph.add_node("generate", cast(Callable[..., object], generate))
    graph.add_node(
        "guardrail",
        director_ai_node(
            facts={"refund": "Refunds are available within 30 days."},
            threshold=0.1,
            use_nli=False,
            on_fail="flag",
        ),
    )
    graph.add_node("output", cast(Callable[..., object], output))
    graph.add_node("retry", cast(Callable[..., object], retry))
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        director_ai_conditional_edge(approved_node="output", rejected_node="retry"),
        {"output": "output", "retry": "retry"},
    )
    graph.add_edge("output", END)
    graph.add_edge("retry", END)

    result = cast(dict[str, Any], graph.compile().invoke({}))

    assert result["route"] == "output"
    assert result["director_ai_approved"] is True
    assert result["director_ai_score"] >= 0.1


def test_public_langgraph_node_reads_dict_message_content() -> None:
    """The public node should read LangGraph message dictionaries."""
    node = director_ai_node(
        facts={"refund": "Refunds are available within 30 days."},
        threshold=0.1,
        use_nli=False,
        on_fail="flag",
    )

    result = node(
        {
            "query": "What is the refund window?",
            "messages": [{"content": "Refunds are available within 30 days."}],
        },
    )

    assert result["director_ai_approved"] is True
    assert result["director_ai_score"] >= 0.1


def test_public_langgraph_rewrite_mode_leaves_response_when_context_missing() -> None:
    """Rewrite mode should not fabricate replacement text without KB context."""
    node = director_ai_node(threshold=1.0, use_nli=False, on_fail="rewrite")
    state = {
        "query": "What is the retention policy?",
        "response": "The answer is unsupported.",
    }

    result = node(state)

    assert result["response"] == "The answer is unsupported."
    assert result["director_ai_approved"] is False
    assert "director_ai_rewritten" not in result


def test_public_langgraph_adapter_rejects_invalid_failure_mode() -> None:
    """Unsupported failure modes should fail before graph construction."""
    with pytest.raises(
        ValueError, match='on_fail must be one of "flag", "raise", "rewrite"'
    ):
        director_ai_node(on_fail="continue")


@pytest.mark.parametrize(
    ("query_key", "response_key", "match"),
    [
        ("", "response", "query_key"),
        ("query", "   ", "response_key"),
        ("message", "message", "must be distinct"),
    ],
)
def test_public_langgraph_adapter_rejects_invalid_state_keys(
    query_key: str,
    response_key: str,
    match: str,
) -> None:
    """State key configuration should fail before graph construction."""
    with pytest.raises(ValueError, match=match):
        director_ai_node(query_key=query_key, response_key=response_key)
