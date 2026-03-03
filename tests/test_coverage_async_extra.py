"""Coverage tests for async_streaming.py — halt reason branches, token timeout."""

from __future__ import annotations

import asyncio
import time

import pytest

from director_ai.core.async_streaming import AsyncStreamingKernel


async def _async_tokens(tokens):
    for t in tokens:
        yield t


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestHaltReasonBranches:
    def test_halt_reason_hard_limit(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5, window_size=100)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a"]),
                lambda t: 0.3,
            )

        session = _run(run())
        assert "hard_limit" in session.halt_reason

    def test_halt_reason_window_avg(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1, window_size=2, window_threshold=0.7,
        )
        scores = iter([0.5, 0.5, 0.5])

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b", "c"]),
                lambda t: next(scores, 0.5),
            )

        session = _run(run())
        assert "window_avg" in session.halt_reason

    def test_halt_reason_downward_trend(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1, window_size=100, trend_window=3, trend_threshold=0.1,
        )
        scores = iter([0.9, 0.8, 0.7, 0.5])

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b", "c", "d"]),
                lambda t: next(scores, 0.5),
            )

        session = _run(run())
        assert "downward_trend" in session.halt_reason

    def test_halt_reason_unknown(self):
        kernel = AsyncStreamingKernel()
        reason = kernel._halt_reason(
            type("E", (), {"coherence": 0.9})(),
            type("S", (), {
                "coherence_history": [],
            })(),
        )
        assert reason == "unknown"
