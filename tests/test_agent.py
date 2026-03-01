# ─────────────────────────────────────────────────────────────────────
# Tests — CoherenceAgent orchestration
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from director_ai.core.agent import CoherenceAgent
from director_ai.core.types import ReviewResult

# ── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    def test_default_uses_mock_generator(self):
        agent = CoherenceAgent()
        assert agent.generator.__class__.__name__ == "MockGenerator"

    def test_llm_url_creates_llm_generator(self):
        agent = CoherenceAgent(llm_api_url="http://localhost:8080/completion")
        assert agent.generator.__class__.__name__ == "LLMGenerator"

    def test_provider_and_url_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            CoherenceAgent(llm_api_url="http://x", provider="openai")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            CoherenceAgent(provider="deepseek")

    def test_provider_missing_env_raises(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="not set"),
        ):
            CoherenceAgent(provider="openai")

    def test_scorer_created_on_init(self):
        agent = CoherenceAgent()
        assert hasattr(agent.scorer, "review")

    def test_scorer_has_threshold(self):
        agent = CoherenceAgent()
        assert 0.0 < agent.scorer.threshold <= 1.0

    def test_kernel_active_on_init(self):
        agent = CoherenceAgent()
        assert agent.kernel.is_active is True


# ── process() happy path ─────────────────────────────────────────────


class TestProcessHappyPath:
    def test_returns_review_result(self):
        agent = CoherenceAgent()
        result = agent.process("What color is the sky?")
        assert isinstance(result, ReviewResult)

    def test_approved_output_contains_agi_prefix(self):
        agent = CoherenceAgent()
        result = agent.process("What color is the sky?")
        if not result.halted:
            assert "[AGI Output]" in result.output

    def test_candidates_evaluated_equals_three(self):
        agent = CoherenceAgent()
        result = agent.process("test prompt")
        assert result.candidates_evaluated == 3

    def test_coherence_score_populated(self):
        agent = CoherenceAgent()
        result = agent.process("test prompt")
        assert result.coherence is not None
        assert 0.0 <= result.coherence.score <= 1.0


# ── process() rejection path ─────────────────────────────────────────


class TestProcessRejection:
    def test_empty_prompt_raises(self):
        agent = CoherenceAgent()
        with pytest.raises(ValueError, match="non-empty"):
            agent.process("")

    def test_non_string_prompt_raises(self):
        agent = CoherenceAgent()
        with pytest.raises(ValueError, match="non-empty"):
            agent.process(42)  # type: ignore[arg-type]

    def test_whitespace_only_raises(self):
        agent = CoherenceAgent()
        with pytest.raises(ValueError, match="non-empty"):
            agent.process("   ")


# ── Fallback modes ───────────────────────────────────────────────────


class TestFallbacks:
    def test_retrieval_fallback(self):
        agent = CoherenceAgent(fallback="retrieval")
        agent.scorer.threshold = 999.0  # force all candidates to fail
        result = agent.process("sky color")
        # GroundTruthStore may or may not return context, but should not crash
        assert isinstance(result, ReviewResult)

    def test_disclaimer_fallback(self):
        agent = CoherenceAgent(fallback="disclaimer")
        agent.scorer.threshold = 999.0
        result = agent.process("test")
        assert isinstance(result, ReviewResult)
        if result.fallback_used:
            assert "could not be fully verified" in result.output

    def test_no_fallback_halts(self):
        agent = CoherenceAgent()
        agent.scorer.threshold = 999.0
        result = agent.process("test")
        assert result.halted is True
        assert "SYSTEM HALT" in result.output

    def test_disclaimer_prefix_on_warning(self):
        agent = CoherenceAgent(disclaimer_prefix="[LOW] ")
        agent.scorer.soft_limit = 1.0  # everything triggers warning
        result = agent.process("test")
        if not result.halted and result.coherence and result.coherence.warning:
            assert result.output.startswith("[LOW] ")


# ── process_query backward compat ────────────────────────────────────


class TestBackwardCompat:
    def test_process_query_returns_string(self):
        agent = CoherenceAgent()
        output = agent.process_query("test")
        assert isinstance(output, str)
