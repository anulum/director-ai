# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Phase 5 Gem 1: Multi-Signal Explainability."""

from __future__ import annotations

from director_ai.core import CoherenceScorer
from director_ai.core.types import CoherenceScore


class TestCoherenceScoreFields:
    """Verify new explainability fields exist and have correct defaults."""

    def test_new_fields_default_none(self):
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

    def test_new_fields_settable(self):
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
        assert score.nli_probs["contradiction"] == 0.7
        assert score.retrieval_confidence == 0.85

    def test_existing_fields_still_work(self):
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

    def test_review_populates_task_type(self):
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

    def test_review_populates_verdict_confidence(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review(
            "What is 2+2?",
            "2+2 equals 4.",
        )
        assert score.verdict_confidence is not None
        assert 0.0 <= score.verdict_confidence <= 1.0

    def test_review_populates_signal_agreement(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review(
            "Describe photosynthesis.",
            "Photosynthesis converts sunlight into chemical energy.",
        )
        assert score.signal_agreement is not None
        assert 0.0 <= score.signal_agreement <= 1.0

    def test_retrieval_confidence_none_without_kb(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review("test", "response")
        # No KB = no retrieval = None
        assert score.retrieval_confidence is None

    def test_escalated_to_judge_none_without_judge(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _, score = scorer.review("test", "response")
        # No judge configured = None (not False — we don't know)
        assert score.escalated_to_judge is None
