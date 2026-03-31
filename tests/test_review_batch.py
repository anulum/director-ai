# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Coalesced review_batch Tests (STRONG)
"""Multi-angle tests for coalesced review_batch pipeline (STRONG)."""

"""Tests for CoherenceScorer.review_batch() and BatchProcessor coalesced path."""

from __future__ import annotations

import pytest

from director_ai.core.batch import BatchProcessor, BatchResult
from director_ai.core.knowledge import GroundTruthStore
from director_ai.core.scorer import CoherenceScorer
from director_ai.core.types import CoherenceScore

# â”€â”€ CoherenceScorer.review_batch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestScorerReviewBatchEmpty:
    def test_empty_returns_empty(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        assert scorer.review_batch([]) == []


class TestScorerReviewBatchSingle:
    def test_single_item_delegates_to_review(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        result = scorer.review_batch([("What is 2+2?", "4")])
        assert len(result) == 1
        approved, score = result[0]
        assert isinstance(approved, bool)
        assert isinstance(score, CoherenceScore)
        assert 0.0 <= score.score <= 1.0


class TestScorerReviewBatchMultiHeuristic:
    def test_multi_item_heuristic_path(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        items = [
            ("What is 2+2?", "4"),
            ("Capital of France?", "Paris"),
            ("Color of sky?", "Blue, consistent with reality"),
        ]
        results = scorer.review_batch(items)
        assert len(results) == 3
        for approved, score in results:
            assert isinstance(approved, bool)
            assert isinstance(score, CoherenceScore)
            assert score.h_logical is not None
            assert score.h_factual is not None

    def test_multi_item_with_ground_truth(self):
        store = GroundTruthStore()
        store.add("What is 2+2?", "2+2 equals 4")
        scorer = CoherenceScorer(threshold=0.3, use_nli=False, ground_truth_store=store)
        items = [
            ("What is 2+2?", "4"),
            ("Unknown question", "Some answer"),
        ]
        results = scorer.review_batch(items)
        assert len(results) == 2
        # First item has ground truth context
        _, score0 = results[0]
        assert score0.h_factual is not None


class TestScorerReviewBatchStrictMode:
    def test_strict_mode_without_nli(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False, strict_mode=True)
        items = [("Q1", "A1"), ("Q2", "A2")]
        results = scorer.review_batch(items)
        assert len(results) == 2
        for approved, score in results:
            assert not approved
            assert score.strict_mode_rejected is True


class TestScorerReviewBatchWithCache:
    def test_cache_hit_skips_scoring(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False, cache_size=100)
        items = [("Q1", "A1"), ("Q2", "A2")]
        # First call populates cache
        results1 = scorer.review_batch(items)
        assert len(results1) == 2
        # Second call should hit cache
        results2 = scorer.review_batch(items)
        assert len(results2) == 2
        for (a1, s1), (a2, s2) in zip(results1, results2, strict=True):
            assert a1 == a2
            assert abs(s1.score - s2.score) < 1e-9

    def test_partial_cache_hit(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False, cache_size=100)
        # Warm cache with one item
        scorer.review_batch([("Q1", "A1")])
        # Batch with one cached + one new
        results = scorer.review_batch([("Q1", "A1"), ("Q2", "A2")])
        assert len(results) == 2


class TestScorerReviewBatchLLMJudge:
    def test_escalation_path_exercised(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False, llm_judge_enabled=True)
        items = [("Q1", "A1"), ("Q2", "A2")]
        results = scorer.review_batch(items)
        assert len(results) == 2


# â”€â”€ BatchProcessor coalesced delegation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestBatchSingleParity:
    """Verify review_batch() produces identical results to sequential review()."""

    def test_task_type_and_threshold_parity(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        items = [
            ("What is 2+2?", "4"),
            ("Capital of France?", "Paris is the capital of France"),
            ("Summarize this article", "The article discusses climate change"),
        ]
        batch_results = scorer.review_batch(items)
        single_results = [scorer.review(p, a) for p, a in items]
        assert len(batch_results) == len(single_results)
        for (b_ok, b_score), (s_ok, s_score) in zip(
            batch_results, single_results, strict=True
        ):
            assert b_ok == s_ok
            assert abs(b_score.score - s_score.score) < 1e-9
            assert b_score.detected_task_type == s_score.detected_task_type

    def test_adaptive_threshold_parity(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        scorer._adaptive_threshold_enabled = True
        scorer._task_type_thresholds = {"qa": 0.25, "default": 0.3}
        items = [("Medical question", "The patient needs treatment")]
        batch_results = scorer.review_batch(items)
        single_results = [scorer.review(p, a) for p, a in items]
        for (b_ok, b_score), (s_ok, s_score) in zip(
            batch_results, single_results, strict=True
        ):
            assert b_ok == s_ok
            assert abs(b_score.score - s_score.score) < 1e-9

    def test_coalesced_nli_applies_task_type_and_meta_thresholds(self, monkeypatch):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        items = [
            ("Prompt A", "Response A"),
            ("Prompt B", "Response B"),
        ]

        class FakeNLI:
            model_available = True

            def __init__(self):
                self.calls = []

            def score_batch(self, pairs):
                self.calls.append(list(pairs))
                if len(self.calls) == 1:
                    return [0.2, 0.4]
                return [0.1, 0.5]

        class FakeStore:
            def retrieve_context(self, prompt, tenant_id=""):
                return f"context for {prompt}"

        class FakeMetaClassifier:
            def predict_threshold(self, prompt, response):
                if prompt == "Prompt A":
                    return (0.5, 0.9)
                return (None, 0.2)

        fake_nli = FakeNLI()
        captured = []

        scorer._nli = fake_nli
        scorer.ground_truth_store = FakeStore()
        scorer._adaptive_threshold_enabled = True
        scorer._task_type_thresholds = {"qa": 0.25, "summarization": 0.1}

        task_types = {"Prompt A": "qa", "Prompt B": "summarization"}
        monkeypatch.setattr(
            scorer, "_detect_task_type", lambda prompt: task_types[prompt]
        )
        monkeypatch.setattr(
            scorer, "_get_meta_classifier", lambda: FakeMetaClassifier()
        )

        def fake_finalise(
            coherence,
            h_logic,
            h_fact,
            action,
            evidence,
            threshold_override=None,
            detected_task_type=None,
        ):
            approved = coherence >= threshold_override
            captured.append(
                {
                    "action": action,
                    "h_logic": h_logic,
                    "h_fact": h_fact,
                    "threshold_override": threshold_override,
                    "detected_task_type": detected_task_type,
                }
            )
            return (
                approved,
                CoherenceScore(
                    score=coherence,
                    approved=approved,
                    h_logical=h_logic,
                    h_factual=h_fact,
                    detected_task_type=detected_task_type,
                ),
            )

        monkeypatch.setattr(scorer, "_finalise_review", fake_finalise)

        results = scorer.review_batch(items)

        assert len(fake_nli.calls) == 2
        assert fake_nli.calls[0] == items
        assert fake_nli.calls[1] == [
            ("context for Prompt A", "Response A"),
            ("context for Prompt B", "Response B"),
        ]
        assert [call["detected_task_type"] for call in captured] == [
            "qa",
            "summarization",
        ]
        assert captured[0]["threshold_override"] == pytest.approx(
            scorer.W_FACT + scorer.W_LOGIC * 0.5
        )
        assert captured[1]["threshold_override"] == pytest.approx(0.1)
        assert results[0][1].detected_task_type == "qa"
        assert results[1][1].detected_task_type == "summarization"


# — BatchProcessor coalesced delegation —


class TestBatchProcessorCoalesced:
    def test_coalesced_path_with_real_scorer(self):
        store = GroundTruthStore()
        scorer = CoherenceScorer(threshold=0.3, use_nli=False, ground_truth_store=store)
        proc = BatchProcessor(scorer, max_concurrency=2)
        items = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]
        result = proc.review_batch(items)
        assert isinstance(result, BatchResult)
        assert result.total == 3
        assert result.succeeded == 3
        assert result.failed == 0
        assert len(result.results) == 3

    def test_coalesced_path_single_item(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        proc = BatchProcessor(scorer, max_concurrency=2)
        result = proc.review_batch([("Q", "A")])
        assert result.total == 1
        assert result.succeeded == 1

    def test_coalesced_fallback_on_exception(self):
        """If scorer.review_batch raises, falls back to per-item."""

        class BrokenBatchScorer:
            def review_batch(self, items, tenant_id=""):
                raise RuntimeError("GPU OOM")

            def review(self, prompt, response, tenant_id=""):
                return (
                    True,
                    CoherenceScore(
                        score=0.9,
                        approved=True,
                        h_logical=0.05,
                        h_factual=0.05,
                    ),
                )

        proc = BatchProcessor(BrokenBatchScorer(), max_concurrency=1)
        result = proc.review_batch([("Q", "A")])
        assert result.total == 1
        assert result.succeeded == 1

    def test_coalesced_fallback_on_bad_length(self):
        """If scorer.review_batch returns wrong length, falls back."""
        _score = CoherenceScore(
            score=0.9,
            approved=True,
            h_logical=0.05,
            h_factual=0.05,
        )

        class WrongLenScorer:
            def review_batch(self, items, tenant_id=""):
                return [(True, _score)]

            def review(self, prompt, response, tenant_id=""):
                return (True, _score)

        proc = BatchProcessor(WrongLenScorer(), max_concurrency=1)
        result = proc.review_batch([("Q1", "A1"), ("Q2", "A2")])
        # Wrong length triggers TypeError â†’ fallback to per-item
        assert result.total == 2
        assert result.succeeded == 2

    def test_coalesced_none_item_counted_as_failure(self):
        """None in results list counts as failed."""
        _score = CoherenceScore(
            score=0.9,
            approved=True,
            h_logical=0.05,
            h_factual=0.05,
        )

        class NoneItemScorer:
            def review_batch(self, items, tenant_id=""):
                return [
                    (True, _score),
                    None,
                ]

        proc = BatchProcessor(NoneItemScorer(), max_concurrency=1)
        result = proc.review_batch([("Q1", "A1"), ("Q2", "A2")])
        assert result.succeeded == 1
        assert result.failed == 1
        assert len(result.errors) == 1

    def test_coalesced_metrics_recorded(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        proc = BatchProcessor(scorer, max_concurrency=2)
        result = proc.review_batch([("Q", "A")])
        assert result.duration_seconds >= 0.0


class TestBatchProcessorCoalescedAsync:
    @pytest.mark.asyncio
    async def test_review_batch_async_coalesced(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        proc = BatchProcessor(scorer, max_concurrency=2)
        items = [("Q1", "A1"), ("Q2", "A2")]
        result = await proc.review_batch_async(items)
        assert isinstance(result, BatchResult)
        assert result.total == 2
        assert result.succeeded == 2

    @pytest.mark.asyncio
    async def test_review_batch_async_fallback(self):
        """Async path falls back when coalesced fails."""

        class BrokenBatchScorer:
            def review_batch(self, items, tenant_id=""):
                raise RuntimeError("fail")

            def review(self, prompt, response, tenant_id=""):
                return (
                    True,
                    CoherenceScore(
                        score=0.9,
                        approved=True,
                        h_logical=0.05,
                        h_factual=0.05,
                    ),
                )

        proc = BatchProcessor(BrokenBatchScorer(), max_concurrency=1)
        result = await proc.review_batch_async([("Q", "A")])
        assert result.total == 1
        assert result.succeeded == 1
