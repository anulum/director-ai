# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Continuous Batching Review Queue
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Server-level request queue for continuous batching.

Accumulates incoming /v1/review requests and flushes them as a single
CoherenceScorer.review_batch() call, reducing GPU kernel launches from
2*N to 2 per flush window.

Usage::

    queue = ReviewQueue(scorer, max_batch=32, flush_timeout_ms=10.0)
    await queue.start()

    # In endpoint handler:
    approved, score = await queue.submit(prompt, response, tenant_id=tid)

    # On shutdown:
    await queue.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging

from .metrics import metrics
from .types import CoherenceScore

_ReviewResult = tuple[bool, CoherenceScore]

logger = logging.getLogger("DirectorAI.ReviewQueue")


class _PendingReview:
    __slots__ = ("prompt", "response", "session", "tenant_id", "future")

    def __init__(
        self,
        prompt: str,
        response: str,
        session,
        tenant_id: str,
        future: asyncio.Future[_ReviewResult],
    ):
        self.prompt = prompt
        self.response = response
        self.session = session
        self.tenant_id = tenant_id
        self.future = future


class ReviewQueue:
    """Accumulates review requests and flushes in batches.

    Parameters
    ----------
    scorer : CoherenceScorer — backend with review() and review_batch().
    max_batch : int — flush when this many requests accumulate.
    flush_timeout_ms : float — flush after this many ms since first
        pending request arrived (even if batch isn't full).
    """

    def __init__(
        self,
        scorer,
        max_batch: int = 32,
        flush_timeout_ms: float = 10.0,
    ) -> None:
        self._scorer = scorer
        self.max_batch = max_batch
        self.flush_timeout_ms = flush_timeout_ms
        self._pending: list[_PendingReview] = []
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._flush_loop())
        logger.info(
            "ReviewQueue started (max_batch=%d, flush_timeout=%.1fms)",
            self.max_batch,
            self.flush_timeout_ms,
        )

    async def stop(self) -> None:
        self._running = False
        self._event.set()
        if self._task:
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(self._task, timeout=2.0)
        async with self._lock:
            if self._pending:
                await self._flush()

    async def submit(
        self,
        prompt: str,
        response: str,
        session=None,
        tenant_id: str = "",
    ) -> _ReviewResult:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[_ReviewResult] = loop.create_future()
        async with self._lock:
            self._pending.append(
                _PendingReview(prompt, response, session, tenant_id, future)
            )
            if len(self._pending) >= self.max_batch:
                await self._flush()
            else:
                self._event.set()
        return await future

    async def _flush_loop(self) -> None:
        timeout_s = self.flush_timeout_ms / 1000.0
        while self._running:
            self._event.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._event.wait(), timeout=timeout_s)
            async with self._lock:
                if self._pending:
                    await self._flush()

    async def _flush(self) -> None:
        batch = self._pending[:]
        self._pending.clear()
        if not batch:
            return

        metrics.observe("review_queue_batch_size", float(len(batch)))

        tenant_groups: dict[str, list[_PendingReview]] = {}
        for item in batch:
            tenant_groups.setdefault(item.tenant_id, []).append(item)

        loop = asyncio.get_running_loop()

        for tenant_id, group in tenant_groups.items():
            items = [(g.prompt, g.response) for g in group]
            try:
                results = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self._scorer.review_batch, items, tenant_id=tenant_id
                    ),
                )
                for pending, result in zip(group, results, strict=True):
                    if not pending.future.done():
                        pending.future.set_result(result)
            except Exception as exc:
                logger.warning("Batch flush failed, falling back per-item: %s", exc)
                for pending in group:
                    if pending.future.done():
                        continue
                    try:
                        result = await loop.run_in_executor(
                            None,
                            functools.partial(
                                self._scorer.review,
                                pending.prompt,
                                pending.response,
                                session=pending.session,
                                tenant_id=pending.tenant_id,
                            ),
                        )
                        pending.future.set_result(result)
                    except Exception as item_exc:
                        pending.future.set_exception(item_exc)
