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

    async def process_batch(
        self, prompts: list[str], tenant_id: str = ""
    ) -> BatchResult:
        """Process a batch of prompts asynchronously.

        Uses ``await backend.process(prompt)`` if backend is CoherenceAgent.
        """
        start = time.monotonic()
        metrics.observe("batch_size", float(len(prompts)))

        result = BatchResult(total=len(prompts))
        ordered: list[ReviewResult | None] = [None for _ in range(len(prompts))]
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _run(idx: int, p: str):
            async with sem:
                try:
                    item_result = await asyncio.wait_for(
                        self._process_one(idx, p, tenant_id),
                        timeout=self.item_timeout,
                    )
                    ordered[idx] = item_result
                    result.succeeded += 1
                except asyncio.TimeoutError:
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

        await asyncio.gather(*[_run(i, p) for i, p in enumerate(prompts)])

        result.results = [r for r in ordered if r is not None]
        result.duration_seconds = time.monotonic() - start
        return result

    async def review_batch(
        self, items: list[tuple[str, str]], tenant_id: str = ""
    ) -> BatchResult:
        """Batch-review (prompt, response) pairs asynchronously.

        Uses ``await backend.review(prompt, response)`` — backend must be CoherenceScorer.  # noqa: E501
        """
        start = time.monotonic()
        metrics.observe("batch_size", float(len(items)))

        result = BatchResult(total=len(items))
        ordered: list[tuple[bool, CoherenceScore] | None] = [
            None for _ in range(len(items))
        ]
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _run(idx: int, p: str, r: str):
            async with sem:
                try:
                    item_result = await asyncio.wait_for(
                        self._review_one(idx, p, r, tenant_id),
                        timeout=self.item_timeout,
                    )
                    ordered[idx] = item_result
                    result.succeeded += 1
                except asyncio.TimeoutError:
                    result.errors.append((idx, "item timeout"))
                    result.failed += 1
                except Exception as e:
                    result.errors.append((idx, str(e)))
                    result.failed += 1

        await asyncio.gather(*[_run(i, p, r) for i, (p, r) in enumerate(items)])

        result.results = [r for r in ordered if r is not None]
        result.duration_seconds = time.monotonic() - start
        return result

    async def process_batch_async(
        self,
        prompts: list[str],
        max_concurrency: int | None = None,
        tenant_id: str = "",
    ) -> BatchResult:
        """Process a batch of prompts asynchronously (legacy alias)."""
        import warnings

        warnings.warn(
            "process_batch_async is deprecated, use process_batch instead",
            DeprecationWarning,
            stacklevel=2,
        )
        old_mc = self.max_concurrency
        if max_concurrency:
            self.max_concurrency = max_concurrency
        res = await self.process_batch(prompts, tenant_id)
        self.max_concurrency = old_mc
        return res

    async def _process_one(
        self, index: int, prompt: str, tenant_id: str = ""
    ) -> ReviewResult:
        """Process a single prompt."""
        metrics.gauge_inc("active_requests")
        try:
            with metrics.timer("review_duration_seconds"):
                result = await self._backend.process(prompt, tenant_id=tenant_id)  # type: ignore[attr-defined]  # noqa: E501
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

    async def _review_one(
        self, index: int, prompt: str, response: str, tenant_id: str = ""
    ) -> tuple[bool, CoherenceScore]:
        """Review a single (prompt, response) pair."""
        metrics.gauge_inc("active_requests")
        try:
            with metrics.timer("review_duration_seconds"):
                approved, score = await self._backend.review(
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
