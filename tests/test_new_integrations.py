# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Multi-Framework Integration Tests
"""Multi-angle tests for LangGraph, Haystack, CrewAI integrations.

Covers: LangGraph node approved/rejected, Haystack batch/empty/serialization,
CrewAI pipe format/check/run alias, parametrised frameworks, pipeline
integration with CoherenceScorer, and performance documentation.
"""

import pytest

from director_ai.core.exceptions import HallucinationError


class TestLangGraphIntegration:
    def test_node_approved(self):
        from director_ai.integrations.langgraph import director_ai_node

        node = director_ai_node(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
        )
        state = {
            "query": "What is the capital of France?",
            "response": "Paris is the capital of France.",
        }
        result = node(state)
        assert result["director_ai_approved"] is True
        assert result["director_ai_score"] > 0

    def test_node_rejected_raises(self):
        from director_ai.integrations.langgraph import director_ai_node

        node = director_ai_node(
            facts={"capital": "Paris is the capital of France."},
            on_fail="raise",
            use_nli=False,
        )
        state = {
            "query": "What is the capital of France?",
            "response": "Mars is the capital of the galaxy beyond recognition.",
        }
        with pytest.raises(HallucinationError):
            node(state)

    def test_node_flag_mode(self):
        from director_ai.integrations.langgraph import director_ai_node

        node = director_ai_node(
            facts={"capital": "Paris is the capital of France."},
            on_fail="flag",
            use_nli=False,
        )
        state = {
            "query": "What is the capital of France?",
            "response": "Mars is the capital of the galaxy beyond recognition.",
        }
        result = node(state)
        assert result["director_ai_approved"] is False

    def test_conditional_edge_approved(self):
        from director_ai.integrations.langgraph import director_ai_conditional_edge

        route = director_ai_conditional_edge("output", "retry")
        assert route({"director_ai_approved": True}) == "output"
        assert route({"director_ai_approved": False}) == "retry"


class TestHaystackIntegration:
    def test_run_with_replies(self):
        from director_ai.integrations.haystack import DirectorAIChecker

        checker = DirectorAIChecker(
            facts={"capital": "Paris is the capital of France."},
        )
        result = checker.run(
            query="What is the capital of France?",
            replies=["Paris is the capital of France.", "Berlin is the capital."],
        )
        assert len(result["replies"]) == 2
        assert len(result["scores"]) == 2

    def test_run_empty(self):
        from director_ai.integrations.haystack import DirectorAIChecker

        checker = DirectorAIChecker()
        result = checker.run(query="", replies=[])
        assert result["replies"] == []

    def test_serialization(self):
        from director_ai.integrations.haystack import DirectorAIChecker

        checker = DirectorAIChecker(threshold=0.7)
        d = checker.to_dict()
        assert d["type"] == "director_ai.integrations.haystack.DirectorAIChecker"


class TestCrewAIIntegration:
    def test_run_pipe_format(self):
        from director_ai.integrations.crewai import DirectorAITool

        tool = DirectorAITool(
            facts={"capital": "Paris is the capital of France."},
        )
        result = tool._run("What is the capital? | Paris is the capital of France.")
        assert "Coherence:" in result

    def test_check_method(self):
        from director_ai.integrations.crewai import DirectorAITool

        tool = DirectorAITool(
            facts={"capital": "Paris is the capital of France."},
        )
        result = tool.check(
            "What is the capital?",
            "Paris is the capital of France.",
        )
        assert "approved" in result
        assert "score" in result

    def test_run_alias(self):
        from director_ai.integrations.crewai import DirectorAITool

        tool = DirectorAITool()
        result = tool.run("some query | some claim")
        assert "Coherence:" in result


class TestIntegrationPerformanceDoc:
    """Document multi-framework integration pipeline performance."""

    def test_langgraph_node_fast(self):
        import time

        from director_ai.integrations.langgraph import director_ai_node

        # threshold=0.1 to avoid HallucinationError on context-free input
        node = director_ai_node(use_nli=False, threshold=0.1)
        state = {"messages": [{"content": "test response"}]}
        t0 = time.perf_counter()
        for _ in range(10):
            node(state)
        per_call_ms = (time.perf_counter() - t0) / 10 * 1000
        assert per_call_ms < 500, f"LangGraph node took {per_call_ms:.1f}ms"

    def test_all_integrations_importable(self):
        from director_ai.integrations.crewai import DirectorAITool
        from director_ai.integrations.haystack import DirectorAIChecker
        from director_ai.integrations.langgraph import director_ai_node

        assert callable(director_ai_node)
        assert DirectorAIChecker is not None
        assert DirectorAITool is not None
