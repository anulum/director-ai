"""
Director-AI LangGraph integration.

Requires: pip install director-ai[langgraph]

Usage::

    from director_ai.integrations.langgraph import director_ai_node

    graph = StateGraph(AgentState)
    graph.add_node("generate", llm_node)
    graph.add_node("guardrail", director_ai_node(facts={"key": "value"}))
    graph.add_edge("generate", "guardrail")
"""

from __future__ import annotations

from typing import Any

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.exceptions import HallucinationError


def director_ai_node(
    facts: dict[str, str] | None = None,
    store: GroundTruthStore | None = None,
    threshold: float = 0.6,
    use_nli: bool | None = None,
    on_fail: str = "raise",
    query_key: str = "query",
    response_key: str = "response",
):
    """Create a LangGraph node function that scores and gates LLM output.

    Parameters
    ----------
    facts : key-value facts for the knowledge base.
    store : pre-built GroundTruthStore.
    threshold : minimum coherence to pass.
    use_nli : NLI mode (None=auto-detect).
    on_fail : "raise" (HallucinationError), "flag" (add field), or "rewrite".
    query_key : state key containing the user query.
    response_key : state key containing the LLM response.
    """
    gts = store or GroundTruthStore()
    if facts:
        for k, v in facts.items():
            gts.add(k, v)
    scorer = CoherenceScorer(
        threshold=threshold, ground_truth_store=gts, use_nli=use_nli
    )

    def _node(state: dict[str, Any]) -> dict[str, Any]:
        query = state.get(query_key, "")
        response = state.get(response_key, "")
        approved, cs = scorer.review(str(query), str(response))

        state["director_ai_score"] = cs.score
        state["director_ai_approved"] = approved
        state["director_ai_h_logical"] = cs.h_logical
        state["director_ai_h_factual"] = cs.h_factual

        if not approved:
            if on_fail == "raise":
                raise HallucinationError(str(query), str(response), cs)
            elif on_fail == "rewrite":
                ctx = gts.retrieve_context(str(query))
                if ctx:
                    state[response_key] = f"Based on verified sources: {ctx}"
                    state["director_ai_rewritten"] = True
        return state

    return _node


def director_ai_conditional_edge(
    approved_node: str = "output",
    rejected_node: str = "retry",
):
    """Create a conditional edge function for LangGraph routing.

    Routes to ``approved_node`` if coherence check passed, else ``rejected_node``.
    """

    def _route(state: dict[str, Any]) -> str:
        if state.get("director_ai_approved", False):
            return approved_node
        return rejected_node

    return _route
