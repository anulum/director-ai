# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Batch Processing Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.agent import CoherenceAgent
from director_ai.core.batch import BatchProcessor, BatchResult
from director_ai.core.knowledge import SAMPLE_FACTS, GroundTruthStore
from director_ai.core.scorer import CoherenceScorer


@pytest.fixture
def batch_agent():
    """BatchProcessor wrapping CoherenceAgent."""
    agent = CoherenceAgent()
    return BatchProcessor(agent, max_concurrency=2)


@pytest.fixture
def batch_scorer():
    """BatchProcessor wrapping CoherenceScorer."""
    store = GroundTruthStore(facts=SAMPLE_FACTS)
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
    return BatchProcessor(scorer, max_concurrency=2)


class TestBatchProcessor:
    """Tests for BatchProcessor.process_batch()."""

    def test_empty_batch(self, batch_agent):
        result = batch_agent.process_batch([])
        assert isinstance(result, BatchResult)
        assert result.total == 0
        assert result.succeeded == 0
        assert result.failed == 0

    def test_single_prompt(self, batch_agent):
        result = batch_agent.process_batch(["What is 2+2?"])
        assert result.total == 1
        assert result.succeeded == 1
        assert result.failed == 0
        assert len(result.results) == 1

    def test_multiple_prompts(self, batch_agent):
        prompts = ["Q1", "Q2", "Q3", "Q4"]
        result = batch_agent.process_batch(prompts)
        assert result.total == 4
        assert result.succeeded == 4
        assert result.failed == 0
        assert len(result.results) == 4

    def test_duration_recorded(self, batch_agent):
        result = batch_agent.process_batch(["Q1"])
        assert result.duration_seconds >= 0.0

    def test_results_are_review_results(self, batch_agent):
        result = batch_agent.process_batch(["What color is the sky?"])
        r = result.results[0]
        assert hasattr(r, "output")
        assert hasattr(r, "coherence")
        assert hasattr(r, "halted")
        assert hasattr(r, "candidates_evaluated")


class TestBatchReview:
    """Tests for BatchProcessor.review_batch()."""

    def test_review_batch(self, batch_scorer):
        items = [
            ("What is 2+2?", "4"),
            ("Capital of France?", "Paris"),
        ]
        result = batch_scorer.review_batch(items)
        assert result.total == 2
        assert result.succeeded == 2
        assert len(result.results) == 2

    def test_review_results_are_tuples(self, batch_scorer):
        items = [("Q", "A")]
        result = batch_scorer.review_batch(items)
        approved, score = result.results[0]
        assert isinstance(approved, bool)
        assert hasattr(score, "score")


class TestBatchAsync:
    """Tests for BatchProcessor.process_batch_async()."""

    @pytest.mark.asyncio
    async def test_async_batch(self):
        agent = CoherenceAgent()
        processor = BatchProcessor(agent, max_concurrency=2)
        result = await processor.process_batch_async(["Q1", "Q2"])
        assert result.total == 2
        assert result.succeeded == 2
        assert result.duration_seconds >= 0.0
