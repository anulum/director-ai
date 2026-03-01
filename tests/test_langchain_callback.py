# ─────────────────────────────────────────────────────────────────────
# Tests — LangChain callback handler integration
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

from types import SimpleNamespace

import pytest

from director_ai.integrations.langchain_callback import CoherenceCallbackHandler


def _make_llm_result(text):
    """Minimal mock of langchain LLMResult."""
    gen = SimpleNamespace(text=text)
    return SimpleNamespace(generations=[[gen]])


class TestConstruction:
    def test_defaults(self):
        handler = CoherenceCallbackHandler()
        assert handler.raise_on_failure is False
        assert handler.last_score is None
        assert handler.scores == []

    def test_custom_threshold(self):
        handler = CoherenceCallbackHandler(threshold=0.8)
        assert handler.scorer.threshold == 0.8

    def test_raise_on_failure_flag(self):
        handler = CoherenceCallbackHandler(raise_on_failure=True)
        assert handler.raise_on_failure is True


class TestOnLlmStart:
    def test_captures_first_prompt(self):
        handler = CoherenceCallbackHandler()
        handler.on_llm_start(serialized={}, prompts=["What is the sky?"])
        assert handler._current_prompt == "What is the sky?"

    def test_empty_prompts(self):
        handler = CoherenceCallbackHandler()
        handler._current_prompt = "old"
        handler.on_llm_start(serialized={}, prompts=[])
        assert handler._current_prompt == "old"


class TestOnLlmEnd:
    def test_scores_response(self):
        handler = CoherenceCallbackHandler()
        handler._current_prompt = "What color is the sky?"
        handler.on_llm_end(_make_llm_result("The sky is blue."))
        assert handler.last_score is not None
        assert len(handler.scores) == 1
        assert 0.0 <= handler.last_score.score <= 1.0

    def test_empty_response_no_score(self):
        handler = CoherenceCallbackHandler()
        handler._current_prompt = "test"
        handler.on_llm_end(_make_llm_result(""))
        assert handler.last_score is None

    def test_malformed_response_no_crash(self):
        handler = CoherenceCallbackHandler()
        handler._current_prompt = "test"
        handler.on_llm_end(SimpleNamespace(generations=None))
        assert handler.last_score is None

    def test_raise_on_failure_triggers(self):
        from director_ai.core.exceptions import CoherenceError

        handler = CoherenceCallbackHandler(raise_on_failure=True, threshold=999.0)
        handler._current_prompt = "test"
        with pytest.raises(CoherenceError, match="below coherence threshold"):
            handler.on_llm_end(_make_llm_result("any response text"))

    def test_multiple_calls_accumulate(self):
        handler = CoherenceCallbackHandler()
        handler._current_prompt = "test"
        handler.on_llm_end(_make_llm_result("First answer"))
        handler.on_llm_end(_make_llm_result("Second answer"))
        assert len(handler.scores) == 2


class TestOnChainStart:
    def test_captures_input_key(self):
        handler = CoherenceCallbackHandler()
        handler.on_chain_start(serialized={}, inputs={"input": "chain query"})
        assert handler._current_prompt == "chain query"

    def test_captures_query_key(self):
        handler = CoherenceCallbackHandler()
        handler.on_chain_start(serialized={}, inputs={"query": "my query"})
        assert handler._current_prompt == "my query"

    def test_no_known_key_keeps_old(self):
        handler = CoherenceCallbackHandler()
        handler._current_prompt = "old"
        handler.on_chain_start(serialized={}, inputs={"unknown": "value"})
        assert handler._current_prompt == "old"
