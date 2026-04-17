# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ReviewQueue Tests
"""Multi-angle tests for async ReviewQueue batching pipeline.

Covers: single submit, batch flush, concurrent submissions, timeout
flush, error handling, lifecycle (start/stop/drain), parametrised
batch sizes, pipeline integration, and performance documentation.
"""

from __future__ import annotations

import asyncio

import pytest

from director_ai.core.review_queue import ReviewQueue
from director_ai.core.scorer import CoherenceScorer
from director_ai.core.types import CoherenceScore


@pytest.fixture
def scorer():
    return CoherenceScorer(threshold=0.3, use_nli=False)


class TestReviewQueueBasic:
    @pytest.mark.asyncio
    async def test_single_submit(self, scorer):
        queue = ReviewQueue(scorer, max_batch=4, flush_timeout_ms=50.0)
        await queue.start()
        try:
            approved, score = await queue.submit("Q", "A")
            assert isinstance(approved, bool)
            assert isinstance(score, CoherenceScore)
            assert 0.0 <= score.score <= 1.0
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_submits(self, scorer):
        queue = ReviewQueue(scorer, max_batch=8, flush_timeout_ms=50.0)
        await queue.start()
        try:
            tasks = [queue.submit(f"Q{i}", f"A{i}") for i in range(5)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            for approved, score in results:
                assert isinstance(approved, bool)
                assert isinstance(score, CoherenceScore)
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_batch_flush_at_max_batch(self, scorer):
        queue = ReviewQueue(scorer, max_batch=3, flush_timeout_ms=5000.0)
        await queue.start()
        try:
            tasks = [queue.submit(f"Q{i}", f"A{i}") for i in range(3)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_timeout_flush(self, scorer):
        queue = ReviewQueue(scorer, max_batch=100, flush_timeout_ms=20.0)
        await queue.start()
        try:
            approved, score = await queue.submit("Q", "A")
            assert isinstance(score, CoherenceScore)
        finally:
            await queue.stop()


class TestReviewQueueTenantGrouping:
    @pytest.mark.asyncio
    async def test_multi_tenant_batch(self, scorer):
        queue = ReviewQueue(scorer, max_batch=10, flush_timeout_ms=50.0)
        await queue.start()
        try:
            tasks = [
                queue.submit("Q1", "A1", tenant_id="t1"),
                queue.submit("Q2", "A2", tenant_id="t2"),
                queue.submit("Q3", "A3", tenant_id="t1"),
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
        finally:
            await queue.stop()


class TestReviewQueueFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_batch_failure(self):
        class BrokenBatchScorer:
            def review_batch(self, items, tenant_id=""):
                raise RuntimeError("batch exploded")

            def review(self, prompt, response, session=None, tenant_id=""):
                return (
                    True,
                    CoherenceScore(
                        score=0.9,
                        approved=True,
                        h_logical=0.05,
                        h_factual=0.05,
                    ),
                )

        queue = ReviewQueue(BrokenBatchScorer(), max_batch=2, flush_timeout_ms=50.0)
        await queue.start()
        try:
            approved, score = await queue.submit("Q", "A")
            assert approved is True
            assert score.score == 0.9
        finally:
            await queue.stop()

    @pytest.mark.asyncio
    async def test_per_item_failure_propagates(self):
        class AllFailScorer:
            def review_batch(self, items, tenant_id=""):
                raise RuntimeError("batch fail")

            def review(self, prompt, response, session=None, tenant_id=""):
                raise ValueError("item fail too")

        queue = ReviewQueue(AllFailScorer(), max_batch=2, flush_timeout_ms=50.0)
        await queue.start()
        try:
            with pytest.raises(ValueError, match="item fail too"):
                await queue.submit("Q", "A")
        finally:
            await queue.stop()


class TestReviewQueueLifecycle:
    @pytest.mark.asyncio
    async def test_stop_drains_pending(self, scorer):
        queue = ReviewQueue(scorer, max_batch=100, flush_timeout_ms=20.0)
        await queue.start()
        task = asyncio.create_task(queue.submit("Q", "A"))
        await asyncio.sleep(0.05)
        await queue.stop()
        approved, score = await asyncio.wait_for(task, timeout=5.0)
        assert isinstance(score, CoherenceScore)

    @pytest.mark.asyncio
    async def test_start_stop_empty(self, scorer):
        queue = ReviewQueue(scorer)
        await queue.start()
        await queue.stop()


class TestReviewQueueParametrised:
    """Parametrised ReviewQueue tests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_batch", [1, 2, 4, 8])
    async def test_various_batch_sizes(self, scorer, max_batch):
        queue = ReviewQueue(scorer, max_batch=max_batch, flush_timeout_ms=50.0)
        await queue.start()
        try:
            approved, score = await queue.submit("test", "response")
            assert isinstance(approved, bool)
            assert isinstance(score, CoherenceScore)
        finally:
            await queue.stop()


class TestReviewQueuePerformanceDoc:
    """Document ReviewQueue pipeline performance."""

    @pytest.mark.asyncio
    async def test_submit_returns_score(self, scorer):
        queue = ReviewQueue(scorer, max_batch=4, flush_timeout_ms=50.0)
        await queue.start()
        try:
            approved, score = await queue.submit("Q", "A")
            assert hasattr(score, "score")
            assert hasattr(score, "h_logical")
            assert hasattr(score, "h_factual")
            assert 0.0 <= score.score <= 1.0
        finally:
            await queue.stop()
