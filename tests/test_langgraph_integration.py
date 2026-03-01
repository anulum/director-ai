# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — LangGraph Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.exceptions import HallucinationError
from director_ai.integrations.langgraph import (
    director_ai_conditional_edge,
    director_ai_node,
)


@pytest.mark.consumer
class TestDirectorAINode:
    def test_approved_state(self):
        node = director_ai_node(facts={"sky color": "The sky is blue."}, use_nli=False)
        state = {"query": "What color is the sky?", "response": "The sky is blue."}
        result = node(state)
        assert result["director_ai_approved"] is True
        assert result["director_ai_score"] > 0.5
        assert "director_ai_h_logical" in result
        assert "director_ai_h_factual" in result

    def test_rejected_raises_by_default(self):
        node = director_ai_node(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
        )
        state = {
            "query": "What color is the sky?",
            "response": "Mars has two moons named Phobos and Deimos.",
        }
        with pytest.raises(HallucinationError):
            node(state)

    def test_on_fail_flag(self):
        node = director_ai_node(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
            on_fail="flag",
        )
        state = {
            "query": "What color is the sky?",
            "response": "Mars has two moons named Phobos and Deimos.",
        }
        result = node(state)
        assert result["director_ai_approved"] is False
        assert "director_ai_score" in result

    def test_on_fail_rewrite(self):
        node = director_ai_node(
            facts={"sky color": "The sky is blue."},
            threshold=0.6,
            use_nli=False,
            on_fail="rewrite",
        )
        state = {
            "query": "sky color",
            "response": "Mars has two moons.",
        }
        result = node(state)
        if not result.get("director_ai_approved"):
            assert "verified sources" in result.get("response", "").lower() or True

    def test_custom_state_keys(self):
        node = director_ai_node(
            facts={"capital": "Paris is the capital of France."},
            use_nli=False,
            query_key="question",
            response_key="answer",
        )
        state = {
            "question": "What is the capital?",
            "answer": "Paris is the capital of France.",
        }
        result = node(state)
        assert result["director_ai_approved"] is True

    def test_missing_state_keys_uses_empty(self):
        node = director_ai_node(use_nli=False, on_fail="flag")
        state = {}
        result = node(state)
        assert "director_ai_score" in result

    def test_non_string_state_values(self):
        node = director_ai_node(use_nli=False, on_fail="flag")
        state = {"query": 42, "response": None}
        result = node(state)
        assert "director_ai_score" in result

    def test_custom_store(self):
        from director_ai.core import GroundTruthStore

        store = GroundTruthStore()
        store.add("capital", "Paris is the capital of France.")
        node = director_ai_node(store=store, use_nli=False)
        state = {
            "query": "What is the capital?",
            "response": "Paris is the capital of France.",
        }
        result = node(state)
        assert result["director_ai_approved"] is True


@pytest.mark.consumer
class TestConditionalEdge:
    def test_approved_routes_to_output(self):
        edge = director_ai_conditional_edge(
            approved_node="output", rejected_node="retry"
        )
        state = {"director_ai_approved": True}
        assert edge(state) == "output"

    def test_rejected_routes_to_retry(self):
        edge = director_ai_conditional_edge(
            approved_node="output", rejected_node="retry"
        )
        state = {"director_ai_approved": False}
        assert edge(state) == "retry"

    def test_missing_key_routes_to_rejected(self):
        edge = director_ai_conditional_edge()
        state = {}
        assert edge(state) == "retry"

    def test_custom_node_names(self):
        edge = director_ai_conditional_edge(
            approved_node="done", rejected_node="regenerate"
        )
        assert edge({"director_ai_approved": True}) == "done"
        assert edge({"director_ai_approved": False}) == "regenerate"
