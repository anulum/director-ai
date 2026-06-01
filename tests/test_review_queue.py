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
import contextlib
import threading
import time

import pytest

from director_ai.core.review_queue import ReviewQueue
from director_ai.core.runtime.review_queue import _PendingReview
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


class TestReviewQueueTimingSideChannel:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tenant_ids",
        [
            ("tenant-a", "tenant-a", "tenant-a", "tenant-a"),
            ("tenant-a", "tenant-a", "tenant-b", "tenant-b"),
            ("tenant-a", "tenant-b", "tenant-c", "tenant-d"),
        ],
    )
    async def test_tenant_mix_does_not_serialise_batch_latency(self, tenant_ids):
        class DelayedTenantScorer:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []
                self._lock = threading.Lock()

            def review_batch(self, items, tenant_id=""):
                time.sleep(0.05)
                with self._lock:
                    self.calls.append((tenant_id, len(items)))
                return [
                    (
                        True,
                        CoherenceScore(
                            score=0.9,
                            approved=True,
                            h_logical=0.05,
                            h_factual=0.05,
                        ),
                    )
                    for _ in items
                ]

        scorer = DelayedTenantScorer()
        queue = ReviewQueue(scorer, max_batch=4, flush_timeout_ms=5000.0)
        await queue.start()
        try:
            started = time.perf_counter()
            results = await asyncio.gather(
                *(
                    queue.submit(f"Q{index}", f"A{index}", tenant_id=tenant_id)
                    for index, tenant_id in enumerate(tenant_ids)
                ),
            )
            elapsed = time.perf_counter() - started
        finally:
            await queue.stop()

        assert len(results) == 4
        assert sorted(scorer.calls) == sorted(
            (tenant_id, tenant_ids.count(tenant_id)) for tenant_id in set(tenant_ids)
        )
        assert elapsed < 0.15


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

    @pytest.mark.asyncio
    async def test_stop_surfaces_worker_failure(self, scorer):
        async def failed_worker():
            raise RuntimeError("worker failed")

        queue = ReviewQueue(scorer)
        queue._task = asyncio.create_task(failed_worker())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="worker failed"):
            await queue.stop()

    @pytest.mark.asyncio
    async def test_stop_times_out_when_worker_does_not_exit(self, scorer):
        never_stop = asyncio.Event()

        async def blocked_worker():
            await never_stop.wait()

        queue = ReviewQueue(scorer)
        queue._task = asyncio.create_task(blocked_worker())

        try:
            with pytest.raises(TimeoutError, match="worker did not stop"):
                await queue.stop()
        finally:
            queue._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await queue._task


class TestReviewQueueInternalBranches:
    @pytest.mark.asyncio
    async def test_empty_flush_leaves_pending_queue_empty(self, scorer):
        queue = ReviewQueue(scorer)

        await queue._flush()

        assert queue._pending == []

    @pytest.mark.asyncio
    async def test_stop_drains_pending_items_without_worker_task(self):
        class RecordingScorer:
            def __init__(self) -> None:
                self.items = None

            def review_batch(self, items, tenant_id=""):
                self.items = (tuple(items), tenant_id)
                return [
                    (
                        True,
                        CoherenceScore(
                            score=0.8,
                            approved=True,
                            h_logical=0.1,
                            h_factual=0.1,
                        ),
                    )
                ]

        scorer = RecordingScorer()
        queue = ReviewQueue(scorer)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        queue._pending.append(_PendingReview("Q", "A", None, "tenant-a", future))

        await queue.stop()

        assert await future == (
            True,
            CoherenceScore(score=0.8, approved=True, h_logical=0.1, h_factual=0.1),
        )
        assert scorer.items == ((("Q", "A"),), "tenant-a")

    @pytest.mark.asyncio
    async def test_fallback_skips_future_that_already_has_result(self):
        class RaisingScorer:
            def review(self, prompt, response, session=None, tenant_id=""):
                raise AssertionError("review should not run for completed future")

        queue = ReviewQueue(RaisingScorer())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        expected = (
            True,
            CoherenceScore(score=0.7, approved=True, h_logical=0.1, h_factual=0.2),
        )
        future.set_result(expected)
        pending = _PendingReview("Q", "A", None, "tenant-a", future)

        await queue._fallback_pending_review(pending)

        assert future.result() == expected

    @pytest.mark.asyncio
    async def test_batch_flush_preserves_future_completed_during_executor_run(self):
        class SingleResultScorer:
            def review_batch(self, items, tenant_id=""):
                return [
                    (
                        False,
                        CoherenceScore(
                            score=0.1,
                            approved=False,
                            h_logical=0.9,
                            h_factual=0.8,
                        ),
                    )
                ]

        queue = ReviewQueue(SingleResultScorer())
        loop = asyncio.get_running_loop()
        expected = (
            True,
            CoherenceScore(score=0.9, approved=True, h_logical=0.1, h_factual=0.1),
        )
        future = loop.create_future()
        future.set_result(expected)
        pending = _PendingReview("Q", "A", None, "tenant-a", future)

        await queue._flush_tenant_group("tenant-a", [pending])

        assert future.result() == expected


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
