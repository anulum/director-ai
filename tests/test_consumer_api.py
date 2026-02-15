# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Consumer API Coverage Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Tests the consumer-facing API: CoherenceAgent, CoherenceScorer, SafetyKernel,
MockGenerator, GroundTruthStore, and dataclass types.
"""

import pytest

from director_ai.core import (
    CoherenceAgent,
    CoherenceScorer,
    SafetyKernel,
    MockGenerator,
    LLMGenerator,
    GroundTruthStore,
    CoherenceScore,
    ReviewResult,
)
import director_ai


class TestVersion:
    def test_version_string(self):
        assert director_ai.__version__ == "0.3.1"

    def test_all_exports_present(self):
        for name in [
            "CoherenceAgent", "CoherenceScorer", "SafetyKernel",
            "MockGenerator", "LLMGenerator", "GroundTruthStore",
            "CoherenceScore", "ReviewResult",
        ]:
            assert hasattr(director_ai, name), f"Missing export: {name}"


class TestCoherenceScore:
    def test_dataclass_fields(self):
        cs = CoherenceScore(score=0.85, approved=True, h_logical=0.1, h_factual=0.2)
        assert cs.score == 0.85
        assert cs.approved is True
        assert cs.h_logical == 0.1
        assert cs.h_factual == 0.2


class TestReviewResult:
    def test_dataclass_fields(self):
        cs = CoherenceScore(score=0.9, approved=True, h_logical=0.05, h_factual=0.1)
        rr = ReviewResult(output="Hello", coherence=cs, halted=False, candidates_evaluated=3)
        assert rr.output == "Hello"
        assert rr.coherence.score == 0.9
        assert rr.halted is False
        assert rr.candidates_evaluated == 3

    def test_halted_result(self):
        rr = ReviewResult(output="HALT", coherence=None, halted=True, candidates_evaluated=0)
        assert rr.halted is True
        assert rr.coherence is None


class TestMockGenerator:
    def test_generate_candidates(self, generator):
        candidates = generator.generate_candidates("What color is the sky?")
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        for cand in candidates:
            assert "text" in cand


class TestGroundTruthStore:
    def test_retrieve_known_topic(self, store):
        context = store.retrieve_context("How many layers are in the SCPN?")
        assert context is not None
        assert "16" in context

    def test_retrieve_sky_color(self, store):
        context = store.retrieve_context("What color is the sky?")
        assert context is not None


class TestCoherenceScorer:
    def test_factual_divergence_truth(self, scorer):
        h = scorer.calculate_factual_divergence("What color is the sky?", "The sky color is blue.")
        assert h < 0.5

    def test_factual_divergence_lie(self, scorer):
        h = scorer.calculate_factual_divergence("What color is the sky?", "The sky color is green.")
        assert h > 0.5

    def test_logical_divergence_consistent(self, scorer):
        h = scorer.calculate_logical_divergence("test", "This is consistent with reality")
        assert h == pytest.approx(0.1)

    def test_logical_divergence_contradictory(self, scorer):
        h = scorer.calculate_logical_divergence("test", "The opposite is true")
        assert h == pytest.approx(0.9)

    def test_review_approved(self, scorer):
        approved, score = scorer.review("sky", "The sky color is blue. This is consistent with reality")
        assert isinstance(score, CoherenceScore)
        assert score.score >= 0.0

    def test_compute_divergence(self, scorer):
        d = scorer.compute_divergence("test", "consistent with reality")
        assert 0.0 <= d <= 1.0

    def test_backward_compat_aliases(self, scorer):
        # Aliases should produce identical results
        prompt, text = "test", "consistent with reality"
        assert scorer.calculate_factual_entropy(prompt, text) == scorer.calculate_factual_divergence(prompt, text)
        assert scorer.calculate_logical_entropy(prompt, text) == scorer.calculate_logical_divergence(prompt, text)
        assert scorer.simulate_future_state(prompt, text) == scorer.compute_divergence(prompt, text)


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

    def test_truthful_query_approved(self, agent):
        result = agent.process("What color is the sky?")
        assert "AGI Output" in result.output
