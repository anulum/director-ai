# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — agent framework guardrail examples

"""Dependency-light guardrail examples for agent framework integrations.

The examples in this module are the executable source for the LangGraph,
CrewAI, and LlamaIndex documentation. They deliberately avoid importing the
optional framework packages so the contract can be tested in the base
environment while still showing the same Director-AI adapter objects that are
used in real applications.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

from director_ai.integrations.crewai import DirectorAITool
from director_ai.integrations.langgraph import (
    director_ai_conditional_edge,
    director_ai_node,
)
from director_ai.integrations.llamaindex import DirectorAIPostprocessor

FACTS: dict[str, str] = {
    "refund_policy": "Customers can request a refund within 30 days of purchase.",
    "support_sla": "The Pro self-host plan includes a 99.5 percent SLA.",
    "company_founding": "Director-AI was first released as an open-core guardrail.",
}


def run_langgraph_guardrail_smoke() -> dict[str, Any]:
    """Exercise the LangGraph guard node and conditional route contract.

    Returns
    -------
    dict[str, Any]
        State fields produced by the Director-AI node plus the route selected by
        the conditional edge helper.
    """

    guard_node = director_ai_node(
        facts=FACTS,
        threshold=0.2,
        use_nli=False,
        on_fail="flag",
    )
    route = director_ai_conditional_edge(
        approved_node="ship_response",
        rejected_node="regenerate",
    )
    state = {
        "query": "What is the refund window?",
        "response": "Customers can request a refund within 30 days of purchase.",
    }
    guarded_state = cast(dict[str, Any], guard_node(state))
    guarded_state["route"] = route(guarded_state)
    return guarded_state


def run_crewai_guardrail_smoke() -> dict[str, Any]:
    """Exercise the CrewAI tool contract through the direct API and tool string."""

    tool = DirectorAITool(facts=FACTS, threshold=0.2, use_nli=False)
    direct = tool.check(
        "What SLA does Pro self-host include?",
        "The Pro self-host plan includes a 99.5 percent SLA.",
    )
    tool_output = tool.run(
        "What is the refund window? | Customers can request a refund within 30 days.",
    )
    return {
        "direct": direct,
        "tool_output": tool_output,
        "tool_name": tool.name,
    }


def run_llamaindex_guardrail_smoke() -> dict[str, Any]:
    """Exercise the LlamaIndex node-postprocessor and final-response contract."""

    postprocessor = DirectorAIPostprocessor(
        facts=FACTS,
        threshold=0.0,
        use_nli=False,
    )
    node = SimpleNamespace(
        text="Customers can request a refund within 30 days of purchase.",
        metadata={},
    )
    kept_nodes = postprocessor.postprocess_nodes(
        [node],
        query_bundle=SimpleNamespace(query_str="What is the refund window?"),
    )
    approved, score = postprocessor.validate_response(
        "What is the refund window?",
        "Customers can request a refund within 30 days of purchase.",
    )
    return {
        "kept_nodes": len(kept_nodes),
        "node_metadata": kept_nodes[0].metadata if kept_nodes else {},
        "response_approved": approved,
        "response_score": score.score,
    }


def run_all_smokes() -> dict[str, Any]:
    """Run every framework adapter smoke and return a JSON-serializable packet."""

    return {
        "langgraph": run_langgraph_guardrail_smoke(),
        "crewai": run_crewai_guardrail_smoke(),
        "llamaindex": run_llamaindex_guardrail_smoke(),
    }


def main() -> None:
    """Print a compact JSON packet for manual smoke runs."""

    print(json.dumps(run_all_smokes(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
