# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Batch Processing Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.agent import CoherenceAgent
from director_ai.core.batch import BatchProcessor, BatchResult
from director_ai.core.knowledge import GroundTruthStore
from director_ai.core.scorer import CoherenceScorer


@pytest.fixture
def batch_agent():
    """BatchProcessor wrapping CoherenceAgent."""
    agent = CoherenceAgent()
    return BatchProcessor(agent, max_concurrency=2)


@pytest.fixture
def batch_scorer():
    """BatchProcessor wrapping CoherenceScorer."""
    store = GroundTruthStore()
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store, use_nli=False)
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


class TestBatchValidation:
    """Tests for BatchProcessor constructor validation."""

    def test_max_concurrency_zero_raises(self):
        agent = CoherenceAgent()
        with pytest.raises(ValueError, match="max_concurrency"):
            BatchProcessor(agent, max_concurrency=0)

    def test_item_timeout_zero_raises(self):
        agent = CoherenceAgent()
        with pytest.raises(ValueError, match="item_timeout"):
            BatchProcessor(agent, item_timeout=0)

    def test_item_timeout_negative_raises(self):
        agent = CoherenceAgent()
        with pytest.raises(ValueError, match="item_timeout"):
            BatchProcessor(agent, item_timeout=-1.0)


class TestBatchErrorPaths:
    """Tests for timeout and exception handling in process_batch."""

    def test_process_batch_backend_exception(self):
        """Backend that raises should count as failed, not crash."""

        class FailingAgent:
            def process(self, prompt):
                raise RuntimeError("model exploded")

        proc = BatchProcessor(FailingAgent(), max_concurrency=1)
        result = proc.process_batch(["boom"])
        assert result.total == 1
        assert result.failed == 1
        assert result.succeeded == 0
        assert len(result.errors) == 1
        assert "model exploded" in result.errors[0][1]

    def test_review_batch_backend_exception(self):
        """review_batch exception handling."""

        class FailingScorer:
            def review(self, prompt, response):
                raise ValueError("bad input")

        proc = BatchProcessor(FailingScorer(), max_concurrency=1)
        result = proc.review_batch([("p", "r")])
        assert result.failed == 1
        assert result.succeeded == 0

    def test_mixed_success_and_failure(self):
        """Some items succeed, some fail."""

        class MixedAgent:
            def process(self, prompt):
                if prompt == "fail":
                    raise RuntimeError("nope")
                from director_ai.core.types import ReviewResult

                return ReviewResult(
                    output="ok", coherence=None, halted=False, candidates_evaluated=1
                )

        proc = BatchProcessor(MixedAgent(), max_concurrency=1)
        result = proc.process_batch(["ok", "fail", "ok"])
        assert result.total == 3
        assert result.succeeded == 2
        assert result.failed == 1


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

    @pytest.mark.asyncio
    async def test_async_batch_exception(self):
        class FailingAgent:
            def process(self, prompt):
                raise RuntimeError("async fail")

        proc = BatchProcessor(FailingAgent(), max_concurrency=1)
        result = await proc.process_batch_async(["x"])
        assert result.failed == 1
        assert result.succeeded == 0
