# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Tests for AsyncStreamingKernel
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.async_streaming import AsyncStreamingKernel


@pytest.mark.consumer
@pytest.mark.asyncio
class TestAsyncStreamingKernel:
    """Tests for async streaming oversight kernel."""

    @pytest.fixture
    def kernel(self):
        return AsyncStreamingKernel(hard_limit=0.3)

    async def _collect_events(self, kernel, tokens, callback):
        events = []
        async for event in kernel.stream_tokens(tokens, callback):
            events.append(event)
            if event.halted:
                break
        return events

    async def test_stream_all_tokens(self, kernel):
        """All tokens streamed when coherence stays high."""
        tokens = ["The", " sky", " is", " blue"]
        events = await self._collect_events(kernel, tokens, lambda t: 0.9)
        assert len(events) == 4
        assert not any(e.halted for e in events)

    async def test_hard_limit_halt(self, kernel):
        """Halts when coherence drops below hard limit."""
        scores = [0.8, 0.7, 0.1]  # 3rd token below 0.3
        idx = 0

        def callback(t):
            nonlocal idx
            s = scores[idx]
            idx += 1
            return s

        tokens = ["a", "b", "c", "d"]
        events = await self._collect_events(kernel, tokens, callback)
        assert events[-1].halted
        assert len(events) == 3

    async def test_async_callback(self, kernel):
        """Works with async coherence callback."""

        async def async_score(t):
            return 0.85

        tokens = ["hello", " world"]
        events = await self._collect_events(kernel, tokens, async_score)
        assert len(events) == 2
        assert all(e.coherence == 0.85 for e in events)

    async def test_stream_to_session(self, kernel):
        """stream_to_session returns complete StreamSession."""
        tokens = ["a", "b", "c"]
        session = await kernel.stream_to_session(tokens, lambda t: 0.9)
        assert session.token_count == 3
        assert not session.halted
        assert session.avg_coherence == pytest.approx(0.9)

    async def test_session_halt_tracking(self, kernel):
        """Session records halt details."""
        scores = iter([0.8, 0.1])
        session = await kernel.stream_to_session(["a", "b"], lambda t: next(scores))
        assert session.halted
        assert "hard_limit" in session.halt_reason

    async def test_window_halt(self):
        """Sliding window halt works."""
        kernel = AsyncStreamingKernel(
            hard_limit=0.1, window_size=3, window_threshold=0.5
        )
        # All scores below window_threshold but above hard_limit
        scores = iter([0.4, 0.4, 0.4, 0.4])
        events = await self._collect_events(
            kernel, ["a", "b", "c", "d"], lambda t: next(scores)
        )
        assert events[-1].halted

    async def test_trend_halt(self):
        """Downward trend halt works."""
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=20,
            trend_window=3,
            trend_threshold=0.2,
        )
        scores = iter([0.9, 0.7, 0.5, 0.3])
        events = await self._collect_events(
            kernel, ["a", "b", "c", "d"], lambda t: next(scores)
        )
        halted = any(e.halted for e in events)
        assert halted

    async def test_inactive_kernel(self, kernel):
        """Inactive kernel halts immediately."""
        kernel.emergency_stop()
        events = await self._collect_events(kernel, ["a", "b"], lambda t: 0.9)
        assert len(events) == 1
        assert events[0].halted

    async def test_async_iterable_source(self, kernel):
        """Works with async iterable token source."""

        async def async_tokens():
            for t in ["async", " tokens", " here"]:
                yield t

        events = await self._collect_events(kernel, async_tokens(), lambda t: 0.8)
        assert len(events) == 3
