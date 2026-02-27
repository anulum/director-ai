# Tests for LangGraph, Haystack, CrewAI integrations

import pytest

from director_ai.core.exceptions import HallucinationError


class TestLangGraphIntegration:
    def test_node_approved(self):
        from director_ai.integrations.langgraph import director_ai_node

        node = director_ai_node(facts={"capital": "Paris is the capital of France."})
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
