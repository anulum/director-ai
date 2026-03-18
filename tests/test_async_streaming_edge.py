# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Async Streaming Edge Case Tests

"""Tests for token_timeout, total_timeout, soft-halt, and async callbacks."""

from __future__ import annotations

import asyncio

import pytest

from director_ai.core.async_streaming import AsyncStreamingKernel


async def _collect(kernel, tokens, callback):
    events = []
    async for event in kernel.stream_tokens(tokens, callback):
        events.append(event)
        if event.halted:
            break
    return events


class TestTotalTimeout:
    @pytest.mark.asyncio
    async def test_total_timeout_fires(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            total_timeout=0.05,
        )

        async def slow_tokens():
            for tok in ["The", " sky", " is"]:
                await asyncio.sleep(0.03)
                yield tok

        events = await _collect(kernel, slow_tokens(), lambda _: 0.9)
        halted_events = [e for e in events if e.halted]
        assert len(halted_events) >= 1


class TestSoftHalt:
    @pytest.mark.asyncio
    async def test_soft_halt_waits_for_sentence_boundary(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=3,
            window_threshold=0.8,
            halt_mode="soft",
        )

        tokens = ["Low", " score", " tokens", " here", " end."]
        call_count = [0]

        def cb(_text):
            call_count[0] += 1
            return 0.3  # below window_threshold

        events = await _collect(kernel, iter(tokens), cb)
        # Soft halt should continue until sentence boundary "."
        token_texts = [e.token for e in events]
        assert any("." in t for t in token_texts), (
            "Soft halt should reach sentence boundary"
        )


class TestAsyncCallback:
    @pytest.mark.asyncio
    async def test_async_coherence_callback(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1)

        async def async_cb(text):
            await asyncio.sleep(0.001)
            return 0.9

        events = await _collect(kernel, iter(["a", "b", "c"]), async_cb)
        assert len(events) == 3
        assert all(e.coherence > 0.5 for e in events)

    @pytest.mark.asyncio
    async def test_sync_callback_also_works(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1)
        events = await _collect(kernel, iter(["a", "b"]), lambda _: 0.8)
        assert len(events) == 2


class TestHardLimitAsync:
    @pytest.mark.asyncio
    async def test_hard_limit_halts(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5)
        scores = iter([0.9, 0.3, 0.9])

        def cb(_text):
            return next(scores)

        events = await _collect(kernel, iter(["a", "b", "c"]), cb)
        assert events[-1].halted
        assert events[-1].coherence < 0.5


class TestValidation:
    def test_invalid_halt_mode(self):
        with pytest.raises(ValueError, match="halt_mode"):
            AsyncStreamingKernel(halt_mode="invalid")

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            AsyncStreamingKernel(window_size=0)

    def test_invalid_trend_window(self):
        with pytest.raises(ValueError, match="trend_window"):
            AsyncStreamingKernel(trend_window=1)
