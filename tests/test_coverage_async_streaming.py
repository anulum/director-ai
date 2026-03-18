# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for async_streaming.py â€” AsyncStreamingKernel."""

from __future__ import annotations

import asyncio

import pytest

from director_ai.core.async_streaming import AsyncStreamingKernel


async def _async_tokens(tokens):
    for t in tokens:
        yield t


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAsyncStreamingValidation:
    def test_bad_hard_limit(self):
        with pytest.raises(ValueError, match="hard_limit"):
            AsyncStreamingKernel(hard_limit=1.5)

    def test_bad_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            AsyncStreamingKernel(window_size=0)

    def test_bad_window_threshold(self):
        with pytest.raises(ValueError, match="window_threshold"):
            AsyncStreamingKernel(window_threshold=2.0)

    def test_bad_trend_window(self):
        with pytest.raises(ValueError, match="trend_window"):
            AsyncStreamingKernel(trend_window=1)

    def test_bad_trend_threshold(self):
        with pytest.raises(ValueError, match="trend_threshold"):
            AsyncStreamingKernel(trend_threshold=-1.0)

    def test_bad_halt_mode(self):
        with pytest.raises(ValueError, match="halt_mode"):
            AsyncStreamingKernel(halt_mode="unknown")

    def test_bad_score_every_n(self):
        with pytest.raises(ValueError, match="score_every_n"):
            AsyncStreamingKernel(score_every_n=0)

    def test_bad_max_cadence(self):
        with pytest.raises(ValueError, match="max_cadence"):
            AsyncStreamingKernel(max_cadence=0)


class TestAsyncStreamingHalt:
    def test_hard_limit_halt(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5)

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens(["a", "b", "c"]),
                lambda t: 0.3,
            ):
                events.append(e)
            return events

        events = _run(run())
        assert events[-1].halted

    def test_window_avg_halt(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=3,
            window_threshold=0.6,
        )
        scores = iter([0.55, 0.55, 0.55, 0.55, 0.55])

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens(["a", "b", "c", "d", "e"]),
                lambda t: next(scores, 0.55),
            ):
                events.append(e)
            return events

        events = _run(run())
        assert any(e.halted for e in events)

    def test_trend_halt(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            trend_window=3,
            trend_threshold=0.1,
        )
        scores = iter([0.9, 0.8, 0.7, 0.6, 0.5])

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens(["a", "b", "c", "d", "e"]),
                lambda t: next(scores, 0.5),
            ):
                events.append(e)
            return events

        events = _run(run())
        assert any(e.halted for e in events)


class TestAsyncStreamingFeatures:
    def test_inactive_kernel(self):
        kernel = AsyncStreamingKernel()
        kernel.emergency_stop()

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens(["a", "b"]),
                lambda t: 0.9,
            ):
                events.append(e)
            return events

        events = _run(run())
        assert events[0].halted

    def test_total_timeout(self):
        kernel = AsyncStreamingKernel(total_timeout=0.001)

        async def slow_tokens():
            yield "a"
            await asyncio.sleep(0.05)
            yield "b"
            yield "c"

        async def run():
            events = []
            async for e in kernel.stream_tokens(slow_tokens(), lambda t: 0.9):
                events.append(e)
            return events

        events = _run(run())
        assert any(e.halted for e in events)

    def test_callback_error(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1)

        def bad_callback(t):
            raise ValueError("boom")

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens(["a", "b"]),
                bad_callback,
            ):
                events.append(e)
            return events

        events = _run(run())
        assert events[0].coherence == 0.0

    def test_async_callback(self):
        kernel = AsyncStreamingKernel()

        async def async_cb(t):
            return 0.9

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens(["a", "b"]),
                async_cb,
            ):
                events.append(e)
            return events

        events = _run(run())
        assert len(events) == 2
        assert events[0].coherence == 0.9

    def test_soft_limit_warning(self):
        kernel = AsyncStreamingKernel(hard_limit=0.3, soft_limit=0.8)

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens(["a"]),
                lambda t: 0.5,
            ):
                events.append(e)
            return events

        events = _run(run())
        assert events[0].warning

    def test_non_string_token(self):
        kernel = AsyncStreamingKernel()

        async def int_tokens():
            yield 42
            yield 43

        async def run():
            events = []
            async for e in kernel.stream_tokens(int_tokens(), lambda t: 0.9):
                events.append(e)
            return events

        events = _run(run())
        assert events[0].token == "42"

    def test_adaptive_cadence(self):
        kernel = AsyncStreamingKernel(
            adaptive=True,
            score_every_n=1,
            max_cadence=4,
            soft_limit=0.7,
        )

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens([f"t{i}" for i in range(10)]),
                lambda t: 0.9,
            ):
                events.append(e)
            return events

        events = _run(run())
        assert len(events) == 10

    def test_adaptive_cadence_drop(self):
        kernel = AsyncStreamingKernel(
            adaptive=True,
            score_every_n=2,
            max_cadence=4,
            soft_limit=0.7,
            hard_limit=0.1,
        )
        call_count = 0

        def score_fn(t):
            nonlocal call_count
            call_count += 1
            return 0.5

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                _async_tokens([f"t{i}" for i in range(6)]),
                score_fn,
            ):
                events.append(e)
            return events

        _run(run())
        assert call_count >= 1

    def test_sync_iterable_source(self):
        kernel = AsyncStreamingKernel()

        async def run():
            events = []
            async for e in kernel.stream_tokens(
                ["a", "b", "c"],
                lambda t: 0.9,
            ):
                events.append(e)
            return events

        events = _run(run())
        assert len(events) == 3


class TestAsyncStreamToSession:
    def test_stream_to_session(self):
        kernel = AsyncStreamingKernel()

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["x", "y", "z"]),
                lambda t: 0.9,
            )

        session = _run(run())
        assert session.token_count == 3
        assert not session.halted

    def test_stream_to_session_halt(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5)
        on_halt_called = []
        kernel.on_halt = lambda s: on_halt_called.append(True)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b"]),
                lambda t: 0.3,
            )

        session = _run(run())
        assert session.halted
        assert "hard_limit" in session.halt_reason
        assert on_halt_called

    def test_stream_to_session_warning_count(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1, soft_limit=0.8)

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a", "b"]),
                lambda t: 0.5,
            )

        session = _run(run())
        assert session.warning_count == 2

    def test_halt_reason_kernel_inactive(self):
        kernel = AsyncStreamingKernel()
        kernel.emergency_stop()

        async def run():
            return await kernel.stream_to_session(
                _async_tokens(["a"]),
                lambda t: 0.9,
            )

        session = _run(run())
        assert session.halted
