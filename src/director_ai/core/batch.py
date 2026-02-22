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
from dataclasses import dataclass, field

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
        backend: object,
        max_concurrency: int = 4,
        item_timeout: float = 60.0,
    ) -> None:
        self._backend = backend
        self.max_concurrency = max_concurrency
        self.item_timeout = item_timeout

    def process_batch(self, prompts: list[str]) -> BatchResult:
        """Process a batch of prompts synchronously using thread pool.

        Uses ``backend.process(prompt)`` if backend is CoherenceAgent.
        """
        start = time.monotonic()
        metrics.observe("batch_size", float(len(prompts)))

        result = BatchResult(total=len(prompts))

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {
                pool.submit(self._process_one, i, p): i for i, p in enumerate(prompts)
            }
            for future in futures:
                idx = futures[future]
                try:
                    item_result = future.result(timeout=self.item_timeout)
                    result.results.append(item_result)
                    result.succeeded += 1
                except TimeoutError:
                    result.errors.append((idx, "item timeout"))
                    result.failed += 1
                    logger.warning("Batch item %d timed out after %.1fs", idx, self.item_timeout)
                except Exception as e:
                    result.errors.append((idx, str(e)))
                    result.failed += 1
                    logger.warning("Batch item %d failed: %s", idx, e)

        result.duration_seconds = time.monotonic() - start
        return result

    def review_batch(self, items: list[tuple[str, str]]) -> BatchResult:
        """Batch-review (prompt, response) pairs synchronously.

        Uses ``backend.review(prompt, response)`` — backend must be CoherenceScorer.
        """
        start = time.monotonic()
        metrics.observe("batch_size", float(len(items)))

        result = BatchResult(total=len(items))

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {
                pool.submit(self._review_one, i, p, r): i
                for i, (p, r) in enumerate(items)
            }
            for future in futures:
                idx = futures[future]
                try:
                    item_result = future.result(timeout=self.item_timeout)
                    result.results.append(item_result)
                    result.succeeded += 1
                except TimeoutError:
                    result.errors.append((idx, "item timeout"))
                    result.failed += 1
                except Exception as e:
                    result.errors.append((idx, str(e)))
                    result.failed += 1

        result.duration_seconds = time.monotonic() - start
        return result

    async def process_batch_async(
        self, prompts: list[str], max_concurrency: int | None = None
    ) -> BatchResult:
        """Process a batch of prompts asynchronously with concurrency control."""
        start = time.monotonic()
        metrics.observe("batch_size", float(len(prompts)))
        sem = asyncio.Semaphore(max_concurrency or self.max_concurrency)

        result = BatchResult(total=len(prompts))
        loop = asyncio.get_event_loop()

        async def _run(idx: int, prompt: str) -> None:
            async with sem:
                try:
                    item_result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, self._process_one, idx, prompt
                        ),
                        timeout=self.item_timeout,
                    )
                    result.results.append(item_result)
                    result.succeeded += 1
                except asyncio.TimeoutError:
                    result.errors.append((idx, "item timeout"))
                    result.failed += 1
                    logger.warning("Async batch item %d timed out after %.1fs", idx, self.item_timeout)
                except Exception as e:
                    result.errors.append((idx, str(e)))
                    result.failed += 1

        await asyncio.gather(*[_run(i, p) for i, p in enumerate(prompts)])
        result.duration_seconds = time.monotonic() - start
        return result

    def _process_one(self, index: int, prompt: str) -> ReviewResult:
        """Process a single prompt."""
        metrics.gauge_inc("active_requests")
        try:
            with metrics.timer("review_duration_seconds"):
                result = self._backend.process(prompt)  # type: ignore[attr-defined]
            metrics.inc("reviews_total")
            if result.halted:
                metrics.inc("reviews_rejected")
            else:
                metrics.inc("reviews_approved")
                metrics.observe("coherence_score", result.coherence.score)
            return result  # type: ignore[no-any-return]
        finally:
            metrics.gauge_dec("active_requests")

    def _review_one(
        self, index: int, prompt: str, response: str
    ) -> tuple[bool, CoherenceScore]:
        """Review a single (prompt, response) pair."""
        metrics.gauge_inc("active_requests")
        try:
            with metrics.timer("review_duration_seconds"):
                approved, score = self._backend.review(prompt, response)  # type: ignore[attr-defined]
            metrics.inc("reviews_total")
            if approved:
                metrics.inc("reviews_approved")
            else:
                metrics.inc("reviews_rejected")
            metrics.observe("coherence_score", score.score)
            return approved, score
        finally:
            metrics.gauge_dec("active_requests")
