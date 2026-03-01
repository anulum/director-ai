# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Consumer API Coverage Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Tests the consumer-facing API: CoherenceAgent, CoherenceScorer, SafetyKernel,
MockGenerator, GroundTruthStore, and dataclass types.
"""

from unittest.mock import MagicMock, patch

import pytest

import director_ai
from director_ai.core import (
    CoherenceScore,
    CoherenceScorer,
    ReviewResult,
)


@pytest.mark.consumer
class TestVersion:
    def test_version_string(self):
        assert director_ai.__version__ == "1.7.0"

    def test_all_exports_present(self):
        for name in [
            "CoherenceAgent",
            "CoherenceScorer",
            "SafetyKernel",
            "MockGenerator",
            "LLMGenerator",
            "GroundTruthStore",
            "CoherenceScore",
            "ReviewResult",
        ]:
            assert hasattr(director_ai, name), f"Missing export: {name}"


class TestCoherenceScore:
    def test_dataclass_fields(self):
        cs = CoherenceScore(score=0.85, approved=True, h_logical=0.1, h_factual=0.2)
        assert cs.score == 0.85
        assert cs.approved is True
        assert cs.h_logical == 0.1
        assert cs.h_factual == 0.2
        assert cs.evidence is None
        assert cs.warning is False


class TestReviewResult:
    def test_dataclass_fields(self):
        cs = CoherenceScore(score=0.9, approved=True, h_logical=0.05, h_factual=0.1)
        rr = ReviewResult(
            output="Hello", coherence=cs, halted=False, candidates_evaluated=3
        )
        assert rr.output == "Hello"
        assert rr.coherence.score == 0.9
        assert rr.halted is False
        assert rr.candidates_evaluated == 3

    def test_halted_result(self):
        rr = ReviewResult(
            output="HALT", coherence=None, halted=True, candidates_evaluated=0
        )
        assert rr.halted is True
        assert rr.coherence is None
        assert rr.fallback_used is False


class TestMockGenerator:
    def test_generate_candidates(self, generator):
        candidates = generator.generate_candidates("What color is the sky?")
        assert isinstance(candidates, list)
        assert len(candidates) == 3
        for cand in candidates:
            assert "text" in cand

    def test_generate_respects_n(self, generator):
        assert len(generator.generate_candidates("test", n=1)) == 1
        assert len(generator.generate_candidates("test", n=5)) == 5


class TestLLMGenerator:
    @patch("director_ai.core.actor.requests.post")
    def test_successful_response(self, mock_post):
        from director_ai.core import LLMGenerator

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"content": "The sky is blue."}
        mock_post.return_value = mock_resp

        gen = LLMGenerator(api_url="http://localhost:8080/completion")
        candidates = gen.generate_candidates("What color is the sky?", n=2)
        assert len(candidates) == 2
        assert candidates[0]["text"] == "The sky is blue."

    @patch("director_ai.core.actor.requests.post")
    def test_error_response(self, mock_post):
        from director_ai.core import LLMGenerator

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_post.return_value = mock_resp

        gen = LLMGenerator(api_url="http://localhost:8080/completion")
        candidates = gen.generate_candidates("test", n=1)
        assert len(candidates) == 1
        assert "Error" in candidates[0]["text"]

    @patch(
        "director_ai.core.actor.requests.post",
        side_effect=ConnectionError("refused"),
    )
    def test_connection_failure(self, mock_post):
        from director_ai.core import LLMGenerator

        gen = LLMGenerator(api_url="http://localhost:8080/completion")
        candidates = gen.generate_candidates("test", n=1)
        assert len(candidates) == 1
        assert "Error" in candidates[0]["text"]


class TestGroundTruthStore:
    def test_retrieve_known_topic(self, store):
        context = store.retrieve_context("How many layers are in the SCPN?")
        assert context is not None
        assert "16" in context

    def test_retrieve_sky_color(self, store):
        context = store.retrieve_context("What color is the sky?")
        assert context is not None


class TestCoherenceScorerNoStore:
    def test_factual_divergence_without_store(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        h = scorer.calculate_factual_divergence("test", "anything")
        assert h == 0.5  # neutral when no store

    def test_review_without_store(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        approved, score = scorer.review("test", "consistent with reality")
        assert isinstance(approved, bool)


class TestCoherenceScorer:
    def test_factual_divergence_truth(self, scorer):
        h = scorer.calculate_factual_divergence(
            "What color is the sky?", "The sky color is blue."
        )
        assert h < 0.5

    def test_factual_divergence_lie(self, scorer):
        h = scorer.calculate_factual_divergence(
            "What color is the sky?", "The sky color is green."
        )
        assert h > 0.5

    def test_logical_divergence_consistent(self, scorer):
        h = scorer.calculate_logical_divergence(
            "test", "This is consistent with reality"
        )
        assert h == pytest.approx(0.1)

    def test_logical_divergence_contradictory(self, scorer):
        h = scorer.calculate_logical_divergence("test", "The opposite is true")
        assert h == pytest.approx(0.9)

    def test_review_approved(self, scorer):
        approved, score = scorer.review(
            "sky", "The sky color is blue. This is consistent with reality"
        )
        assert isinstance(score, CoherenceScore)
        assert score.score >= 0.0

    def test_compute_divergence(self, scorer):
        d = scorer.compute_divergence("test", "consistent with reality")
        assert 0.0 <= d <= 1.0

    def test_backward_compat_aliases(self, scorer):
        # Aliases should produce identical results
        prompt, text = "test", "consistent with reality"
        assert scorer.calculate_factual_entropy(
            prompt, text
        ) == scorer.calculate_factual_divergence(prompt, text)
        assert scorer.calculate_logical_entropy(
            prompt, text
        ) == scorer.calculate_logical_divergence(prompt, text)
        assert scorer.simulate_future_state(prompt, text) == scorer.compute_divergence(
            prompt, text
        )


class TestScorerStrictMode:
    def test_strict_mode_logical_returns_neutral(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False, strict_mode=True)
        h = scorer.calculate_logical_divergence("test", "opposite is true")
        assert h == 0.5

    def test_strict_mode_factual_returns_neutral(self, store):
        scorer = CoherenceScorer(
            threshold=0.5,
            ground_truth_store=store,
            use_nli=False,
            strict_mode=True,
        )
        h = scorer.calculate_factual_divergence(
            "What color is the sky?", "totally wrong"
        )
        assert h == 0.5

    def test_heuristic_mode_logical_differs(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False, strict_mode=False)
        h = scorer.calculate_logical_divergence("test", "The opposite is true")
        assert h == pytest.approx(0.9)

    def test_custom_weights(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False, w_logic=0.3, w_fact=0.7)
        assert scorer.W_LOGIC == 0.3
        assert scorer.W_FACT == 0.7


class TestSafetyKernel:
    def test_stream_output_safe(self, kernel):
        output = kernel.stream_output(["Hello ", "world"], lambda t: 0.8)
        assert output == "Hello world"

    def test_stream_output_halt(self, kernel):
        output = kernel.stream_output(["Bad ", "output"], lambda t: 0.3)
        assert "KERNEL INTERRUPT" in output

    def test_emergency_stop_deactivates(self, kernel):
        assert kernel.is_active is True
        kernel.emergency_stop()
        assert kernel.is_active is False

    def test_custom_hard_limit(self):
        from director_ai.core import SafetyKernel

        kernel = SafetyKernel(hard_limit=0.7)
        # Score 0.6 is above default 0.5 but below custom 0.7 → should halt
        output = kernel.stream_output(["test"], lambda t: 0.6)
        assert "KERNEL INTERRUPT" in output


class TestCoherenceAgent:
    def test_process_returns_review_result(self, agent):
        result = agent.process("What is 2+2?")
        assert isinstance(result, ReviewResult)
        assert isinstance(result.output, str)
        assert result.candidates_evaluated > 0

    def test_process_query_backward_compat(self, agent):
        output = agent.process_query("What color is the sky?")
        assert isinstance(output, str)
        assert "AGI Output" in output or "SYSTEM HALT" in output

    def test_truthful_query_produces_result(self, agent):
        result = agent.process("What color is the sky?")
        assert "AGI Output" in result.output or "SYSTEM HALT" in result.output

    def test_process_rejects_empty_prompt(self, agent):
        with pytest.raises(ValueError, match="non-empty string"):
            agent.process("")

    def test_process_rejects_non_string(self, agent):
        with pytest.raises(ValueError, match="non-empty string"):
            agent.process(123)
