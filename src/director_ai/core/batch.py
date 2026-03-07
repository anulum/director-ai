# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Request Batching
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Batch processing for CoherenceScorer and CoherenceAgent.

Usage::

    from director_ai.core.batch import BatchProcessor

    processor = BatchProcessor(agent)
    results = processor.process_batch(["prompt1", "prompt2", "prompt3"])

    # Async
    results = await processor.process_batch_async(prompts, max_concurrency=8)
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from .exceptions import ValidationError
from .metrics import metrics
from .types import CoherenceScore, ReviewResult

logger = logging.getLogger("DirectorAI.Batch")


@dataclass
class BatchItem:
    """Single item in a batch request."""

    prompt: str
    response: str = ""
    index: int = 0


@dataclass
class BatchResult:
    """Result of a batch processing operation."""

    results: list[ReviewResult | tuple[bool, CoherenceScore]] = field(
        default_factory=list
    )
    errors: list[tuple[int, str]] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    duration_seconds: float = 0.0


class BatchProcessor:
    """Batch processing wrapper for CoherenceAgent or CoherenceScorer.

    Parameters
    ----------
    backend : CoherenceAgent or CoherenceScorer instance.
    max_concurrency : int — maximum parallel workers.
    """

    def __init__(
        self,
        backend: Any,
        max_concurrency: int = 4,
        item_timeout: float = 60.0,
    ) -> None:
        if max_concurrency < 1:
            msg = f"max_concurrency must be >= 1, got {max_concurrency}"
            raise ValidationError(msg)
        if item_timeout <= 0:
            raise ValidationError(f"item_timeout must be > 0, got {item_timeout}")
        self._backend = backend
        self.max_concurrency = max_concurrency
        self.item_timeout = item_timeout

    def process_batch(self, prompts: list[str], tenant_id: str = "") -> BatchResult:
        """Process a batch of prompts with concurrent execution.

        Uses ``backend.process(prompt)`` if backend is CoherenceAgent.
        """
        start = time.monotonic()
        metrics.observe("batch_size", float(len(prompts)))

        result = BatchResult(total=len(prompts))
        ordered: list[ReviewResult | None] = [None for _ in range(len(prompts))]

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {
                pool.submit(self._process_one, i, p, tenant_id): i
                for i, p in enumerate(prompts)
            }
            for future in futures:
                idx = futures[future]
                try:
                    item_result = future.result(timeout=self.item_timeout)
                    ordered[idx] = item_result
                    result.succeeded += 1
                except FutureTimeout:
                    result.errors.append((idx, "item timeout"))
                    result.failed += 1
                    logger.warning(
                        "Batch item %d timed out after %.1fs",
                        idx,
                        self.item_timeout,
                    )
                except Exception as e:
                    result.errors.append((idx, str(e)))
                    result.failed += 1
                    logger.warning("Batch item %d failed: %s", idx, e)

        result.results = [r for r in ordered if r is not None]
        result.duration_seconds = time.monotonic() - start
        return result

    def review_batch(
        self, items: list[tuple[str, str]], tenant_id: str = ""
    ) -> BatchResult:
        """Batch-review (prompt, response) pairs.

        When the backend has a ``review_batch`` method (CoherenceScorer),
        delegates to it for coalesced NLI inference (2 GPU kernel calls
        total instead of 2*N). Falls back to per-item ThreadPoolExecutor.
        """
        start = time.monotonic()
        metrics.observe("batch_size", float(len(items)))
        result = BatchResult(total=len(items))

        scorer_batch_fn = getattr(self._backend, "review_batch", None)
        if scorer_batch_fn is not None and not isinstance(
            self._backend, BatchProcessor
        ):
            try:
                batch_results = scorer_batch_fn(items, tenant_id=tenant_id)
                if not isinstance(batch_results, list) or len(batch_results) != len(
                    items
                ):
                    raise TypeError("review_batch returned invalid result")
                for idx, item_result in enumerate(batch_results):
                    if item_result is not None:
                        result.succeeded += 1
                        approved = item_result[0]
                        score = item_result[1]
                        metrics.inc("reviews_total")
                        if approved:
                            metrics.inc("reviews_approved")
                        else:
                            metrics.inc("reviews_rejected")
                        metrics.observe("coherence_score", score.score)
                    else:
                        result.errors.append((idx, "scorer returned None"))
                        result.failed += 1
                result.results = [r for r in batch_results if r is not None]
                result.duration_seconds = time.monotonic() - start
                return result
            except Exception as exc:
                logger.warning(
                    "Coalesced review_batch failed, falling back to per-item: %s", exc
                )

        ordered: list[tuple[bool, CoherenceScore] | None] = [
            None for _ in range(len(items))
        ]

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {
                pool.submit(self._review_one, i, p, r, tenant_id): i
                for i, (p, r) in enumerate(items)
            }
            for future in futures:
                idx = futures[future]
                try:
                    item_result = future.result(timeout=self.item_timeout)
                    ordered[idx] = item_result
                    result.succeeded += 1
                except FutureTimeout:
                    result.errors.append((idx, "item timeout"))
                    result.failed += 1
                except Exception as e:
                    result.errors.append((idx, str(e)))
                    result.failed += 1

        result.results = [r for r in ordered if r is not None]
        result.duration_seconds = time.monotonic() - start
        return result

    def _process_one(
        self, index: int, prompt: str, tenant_id: str = ""
    ) -> ReviewResult:
        """Process a single prompt."""
        metrics.gauge_inc("active_requests")
        try:
            with metrics.timer("review_duration_seconds"):
                try:
                    result = self._backend.process(prompt, tenant_id=tenant_id)  # type: ignore[attr-defined]  # noqa: E501
                except TypeError:
                    result = self._backend.process(prompt)  # type: ignore[attr-defined]
            metrics.inc("reviews_total")
            if result.halted:
                metrics.inc("reviews_rejected")
            else:
                metrics.inc("reviews_approved")
                if result.coherence is not None:
                    metrics.observe("coherence_score", result.coherence.score)
            return result  # type: ignore[no-any-return]
        finally:
            metrics.gauge_dec("active_requests")

    def _review_one(
        self, index: int, prompt: str, response: str, tenant_id: str = ""
    ) -> tuple[bool, CoherenceScore]:
        """Review a single (prompt, response) pair."""
        metrics.gauge_inc("active_requests")
        try:
            with metrics.timer("review_duration_seconds"):
                approved, score = self._backend.review(
                    prompt, response, tenant_id=tenant_id
                )  # type: ignore[attr-defined]
            metrics.inc("reviews_total")
            if approved:  # pragma: no cover — tested via scorer.review
                metrics.inc("reviews_approved")
            else:
                metrics.inc("reviews_rejected")
            metrics.observe("coherence_score", score.score)
            return approved, score  # type: ignore[no-any-return]
        finally:
            metrics.gauge_dec("active_requests")

    async def process_batch_async(
        self,
        prompts: list[str],
        max_concurrency: int | None = None,
        tenant_id: str = "",
    ) -> BatchResult:
        """Async version of process_batch using asyncio concurrency."""
        start = time.monotonic()
        metrics.observe("batch_size", float(len(prompts)))
        sem = asyncio.Semaphore(max_concurrency or self.max_concurrency)
        loop = asyncio.get_running_loop()

        result = BatchResult(total=len(prompts))
        ordered: list[ReviewResult | None] = [None] * len(prompts)

        async def _run(idx: int, prompt: str) -> None:
            async with sem:
                try:
                    item = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, self._process_one, idx, prompt, tenant_id
                        ),
                        timeout=self.item_timeout,
                    )
                    ordered[idx] = item
                    result.succeeded += 1
                except TimeoutError:
                    result.errors.append((idx, "item timeout"))
                    result.failed += 1
                except Exception as e:
                    result.errors.append((idx, str(e)))
                    result.failed += 1

        await asyncio.gather(*[_run(i, p) for i, p in enumerate(prompts)])
        result.results = [r for r in ordered if r is not None]
        result.duration_seconds = time.monotonic() - start
        return result

    async def review_batch_async(
        self,
        items: list[tuple[str, str]],
        max_concurrency: int | None = None,
        tenant_id: str = "",
    ) -> BatchResult:
        """Async version of review_batch.

        Offloads coalesced scorer.review_batch() to the thread pool when
        available, falling back to per-item asyncio concurrency.
        """
        start = time.monotonic()
        metrics.observe("batch_size", float(len(items)))
        loop = asyncio.get_running_loop()

        scorer_batch_fn = getattr(self._backend, "review_batch", None)
        if scorer_batch_fn is not None and not isinstance(
            self._backend, BatchProcessor
        ):
            try:
                result = await loop.run_in_executor(
                    None, lambda: self.review_batch(items, tenant_id=tenant_id)
                )
                return result
            except Exception as exc:
                logger.warning(
                    "Async coalesced review_batch failed, falling back: %s", exc
                )

        sem = asyncio.Semaphore(max_concurrency or self.max_concurrency)
        result = BatchResult(total=len(items))
        ordered: list[tuple[bool, CoherenceScore] | None] = [None] * len(items)

        async def _run(idx: int, prompt: str, response: str) -> None:
            async with sem:
                try:
                    item = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, self._review_one, idx, prompt, response, tenant_id
                        ),
                        timeout=self.item_timeout,
                    )
                    ordered[idx] = item
                    result.succeeded += 1
                except TimeoutError:
                    result.errors.append((idx, "item timeout"))
                    result.failed += 1
                except Exception as e:
                    result.errors.append((idx, str(e)))
                    result.failed += 1

        await asyncio.gather(*[_run(i, p, r) for i, (p, r) in enumerate(items)])
        result.results = [r for r in ordered if r is not None]
        result.duration_seconds = time.monotonic() - start
        return result
