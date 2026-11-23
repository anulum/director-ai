# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for multi-signal explainability.

Covers: new field defaults, field settability, existing field compatibility,
task type detection, verdict confidence, signal agreement, retrieval
confidence, judge escalation, parametrised task detection, and pipeline
performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core import CoherenceScorer
from director_ai.core.types import CoherenceScore


class TestCoherenceScoreFields:
    """Verify new explainability fields exist and have correct defaults."""

    def test_new_fields_default_none(self) -> None:
        """The explainability fields default to None."""
        score = CoherenceScore(
            score=0.8,
            approved=True,
            h_logical=0.1,
            h_factual=0.2,
        )
        assert score.detected_task_type is None
        assert score.escalated_to_judge is None
        assert score.nli_probs is None
        assert score.retrieval_confidence is None

    def test_new_fields_settable(self) -> None:
        """The explainability fields accept values at construction."""
        score = CoherenceScore(
            score=0.8,
            approved=True,
            h_logical=0.1,
            h_factual=0.2,
            detected_task_type="qa",
            escalated_to_judge=False,
            nli_probs={"entailment": 0.1, "neutral": 0.2, "contradiction": 0.7},
            retrieval_confidence=0.85,
        )
        assert score.detected_task_type == "qa"
        assert score.escalated_to_judge is False
        assert score.nli_probs is not None
        assert score.nli_probs["contradiction"] == 0.7
        assert score.retrieval_confidence == 0.85

    def test_existing_fields_still_work(self) -> None:
        """The earlier confidence fields keep their values."""
        score = CoherenceScore(
            score=0.9,
            approved=True,
            h_logical=0.05,
            h_factual=0.1,
            verdict_confidence=0.95,
            signal_agreement=0.88,
            nli_model_confidence=0.92,
            contradiction_index=0.02,
        )
        assert score.verdict_confidence == 0.95
        assert score.signal_agreement == 0.88
        assert score.nli_model_confidence == 0.92
        assert score.contradiction_index == 0.02


class TestReviewExplainability:
    """Verify scorer.review() populates explainability fields."""

    def test_review_populates_task_type(self) -> None:
        """A review records the detected task type."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        approved, score = scorer.review(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        assert score.detected_task_type is not None
        assert score.detected_task_type in (
            "dialogue",
            "summarization",
            "qa",
            "rag",
            "fact_check",
            "default",
        )

    def test_review_populates_verdict_confidence(self) -> None:
        """A review records a verdict confidence between 0 and 1."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review(
            "What is 2+2?",
            "2+2 equals 4.",
        )
        assert score.verdict_confidence is not None
        assert 0.0 <= score.verdict_confidence <= 1.0

    def test_review_populates_signal_agreement(self) -> None:
        """A review records a signal agreement between 0 and 1."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review(
            "Describe photosynthesis.",
            "Photosynthesis converts sunlight into chemical energy.",
        )
        assert score.signal_agreement is not None
        assert 0.0 <= score.signal_agreement <= 1.0

    def test_retrieval_confidence_none_without_kb(self) -> None:
        """Without a knowledge base the retrieval confidence stays None."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review("test", "response")
        # No KB = no retrieval = None
        assert score.retrieval_confidence is None

    def test_escalated_to_judge_none_without_judge(self) -> None:
        """Without a judge the escalation flag stays None, not False."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review("test", "response")
        # No judge configured = None (not False — we don't know)
        assert score.escalated_to_judge is None

    @pytest.mark.parametrize(
        "prompt,expected_task",
        [
            ("What is 2+2?", "qa"),
            ("Summarise the following document:", "summarization"),
        ],
    )
    def test_task_type_detection_parametrised(
        self, prompt: str, expected_task: str
    ) -> None:
        """Every prompt kind gets a detected task type."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review(prompt, "Response text.")
        assert score.detected_task_type is not None

    @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_explainability_at_various_thresholds(self, threshold: float) -> None:
        """Every threshold still exposes the explainability fields."""
        scorer = CoherenceScorer(threshold=threshold, use_nli=False)
        _, score = scorer.review("test", "response")
        assert hasattr(score, "detected_task_type")
        assert hasattr(score, "verdict_confidence")
        assert hasattr(score, "signal_agreement")


class TestExplainabilityPerformanceDoc:
    """Document explainability pipeline performance."""

    def test_score_has_all_explainability_fields(self) -> None:
        """A review's score exposes every explainability field."""
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review("What is AI?", "AI is intelligence.")
        fields = [
            "score",
            "approved",
            "h_logical",
            "h_factual",
            "detected_task_type",
            "verdict_confidence",
            "signal_agreement",
        ]
        for field in fields:
            assert hasattr(score, field), f"Missing: {field}"

    def test_explainability_overhead_minimal(self) -> None:
        """A heuristic review with explainability takes under 5 ms."""
        import time

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        for _ in range(10):
            scorer.review("warmup", "warmup")

        t0 = time.perf_counter()
        for _ in range(100):
            scorer.review("What is AI?", "AI is intelligence.")
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_call_ms < 5.0, (
            f"Explainability review took {per_call_ms:.3f}ms/call (expected <5ms)"
        )
