# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AsyncStreamingKernel Tests
"""Multi-angle tests for async streaming kernel pipeline.

Covers: async token streaming, halt conditions, soft halt, timeout,
concurrent callbacks, event structure, pipeline integration with
CoherenceScorer, and performance documentation.
"""

import pytest

import director_ai.core.runtime.async_streaming as async_streaming_mod
from director_ai.core.async_streaming import AsyncStreamingKernel
from director_ai.core.streaming import StreamSession, TokenEvent


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

    async def _collect_events_without_halt_break(self, kernel, tokens, callback):
        events = []
        async for event in kernel.stream_tokens(tokens, callback):
            events.append(event)
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
            hard_limit=0.1,
            window_size=3,
            window_threshold=0.5,
        )
        # All scores below window_threshold but above hard_limit
        scores = iter([0.4, 0.4, 0.4, 0.4])
        events = await self._collect_events(
            kernel,
            ["a", "b", "c", "d"],
            lambda t: next(scores),
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
            kernel,
            ["a", "b", "c", "d"],
            lambda t: next(scores),
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

    async def test_total_timeout_halts(self):
        """Total timeout halts stream."""
        import asyncio as _aio

        kernel = AsyncStreamingKernel(hard_limit=0.1, total_timeout=0.05)

        async def slow_tokens():
            for t in ["a", "b", "c", "d"]:
                await _aio.sleep(0.02)
                yield t

        events = await self._collect_events(kernel, slow_tokens(), lambda t: 0.9)
        assert events[-1].halted
        assert not kernel.is_active

    async def test_token_timeout_halts(self):
        """Token timeout halts stream."""
        import asyncio as _aio

        kernel = AsyncStreamingKernel(hard_limit=0.1, token_timeout=0.01)

        async def slow_callback(t):
            await _aio.sleep(0.05)
            return 0.9

        events = await self._collect_events(kernel, ["a", "b"], slow_callback)
        assert events[-1].halted
        assert not kernel.is_active

    async def test_non_string_token_is_coerced_to_string(self, kernel):
        events = await self._collect_events(
            kernel,
            [1, 2.5, None, {"k": "v"}],
            lambda t: 0.9,
        )
        assert all(isinstance(event.token, str) for event in events)

    async def test_callback_fallback_uses_last_score(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1)
        call_count = 0

        def callback(_text):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise TypeError("intentional")
            return 0.9

        events = await self._collect_events_without_halt_break(
            kernel, [0, 1, 2], callback
        )
        assert events[-1].coherence == 0.9
        assert not any(e.halted for e in events)
        assert call_count == 3

    async def test_adaptive_cadence_increases(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            soft_limit=0.6,
            adaptive=True,
            max_cadence=8,
            trend_threshold=10.0,
        )
        call_count = 0

        def callback(_text):
            nonlocal call_count
            call_count += 1
            return 0.9

        events = await self._collect_events(
            kernel, [str(i) for i in range(40)], callback
        )
        assert not any(event.halted for event in events)
        assert call_count < len(events)

    async def test_adaptive_resets_cadence_on_soft_signal(self, monkeypatch):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            soft_limit=0.6,
            adaptive=True,
            max_cadence=8,
            trend_window=20,
            window_size=20,
        )
        # Keep only one branch explicit and stable:
        # reset path should execute when adaptive cadence sees a low-smoothed window.
        original_mean = async_streaming_mod._mean

        def patched_mean(values):
            if len(values) == 2:
                return 0.0
            return original_mean(values)

        monkeypatch.setattr(async_streaming_mod, "_mean", patched_mean)

        call_count = 0

        def callback(_text):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return 0.4
            return 0.9

        events = await self._collect_events_without_halt_break(
            kernel,
            [str(i) for i in range(40)],
            callback,
        )
        assert not any(event.halted for event in events)
        assert call_count > 8

    async def test_concurrent_streams_preserve_order_and_accumulated_text_isolation(
        self,
    ):
        import asyncio

        stream_count = 12
        tokens_per_stream = 32
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=tokens_per_stream + 1,
            trend_window=tokens_per_stream + 1,
        )

        async def run_stream(stream_id: int) -> list[TokenEvent]:
            tokens = [f"s{stream_id}:{idx:02d}|" for idx in range(tokens_per_stream)]
            foreign_prefixes = [
                f"s{other}:" for other in range(stream_count) if other != stream_id
            ]
            observed_accumulated: list[str] = []

            async def token_source():
                for idx, token in enumerate(tokens):
                    if idx % 4 == 0:
                        await asyncio.sleep(0)
                    yield token

            async def score(accumulated: str) -> float:
                await asyncio.sleep(0)
                assert not any(prefix in accumulated for prefix in foreign_prefixes)
                observed_accumulated.append(accumulated)
                return 0.95

            events = await self._collect_events_without_halt_break(
                kernel,
                token_source(),
                score,
            )
            assert [event.index for event in events] == list(range(tokens_per_stream))
            assert [event.token for event in events] == tokens
            assert observed_accumulated[-1] == "".join(tokens)
            assert not any(event.halted for event in events)
            return events

        streams = await asyncio.gather(*(run_stream(i) for i in range(stream_count)))

        assert len(streams) == stream_count
        assert sum(len(stream) for stream in streams) == (
            stream_count * tokens_per_stream
        )

    async def test_no_timeout_passes(self):
        """Default (no timeout) streams all tokens."""
        kernel = AsyncStreamingKernel(hard_limit=0.1)
        events = await self._collect_events(kernel, ["a", "b", "c"], lambda t: 0.9)
        assert len(events) == 3
        assert not any(e.halted for e in events)

    async def test_timeout_params_forwarded(self):
        """Timeout params are accessible on the kernel."""
        kernel = AsyncStreamingKernel(token_timeout=1.5, total_timeout=30.0)
        assert kernel.token_timeout == 1.5
        assert kernel.total_timeout == 30.0

    async def test_halt_mode_default_is_hard(self):
        kernel = AsyncStreamingKernel()
        assert kernel.halt_mode == "hard"

    async def test_halt_mode_soft_accepted(self):
        kernel = AsyncStreamingKernel(halt_mode="soft")
        assert kernel.halt_mode == "soft"

    async def test_invalid_halt_mode_raises(self):
        with pytest.raises(ValueError, match="halt_mode"):
            AsyncStreamingKernel(halt_mode="invalid")

    async def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError, match="window_size"):
            AsyncStreamingKernel(window_size=0)

    async def test_invalid_window_threshold_raises(self):
        with pytest.raises(ValueError, match="window_threshold"):
            AsyncStreamingKernel(window_threshold=-0.1)

    async def test_invalid_trend_threshold_raises(self):
        with pytest.raises(ValueError, match="trend_threshold"):
            AsyncStreamingKernel(trend_threshold=0.0)

    async def test_invalid_score_every_n_raises(self):
        with pytest.raises(ValueError, match="score_every_n"):
            AsyncStreamingKernel(score_every_n=0)

    async def test_invalid_max_cadence_raises(self):
        with pytest.raises(ValueError, match="max_cadence"):
            AsyncStreamingKernel(max_cadence=0)

    async def test_invalid_hard_limit_raises(self):
        with pytest.raises(ValueError, match="hard_limit"):
            AsyncStreamingKernel(hard_limit=1.5)

    async def test_soft_halt_waits_for_sentence_boundary(self):
        """Soft-halt yields tokens until sentence boundary, not immediately."""
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=3,
            window_threshold=0.55,
            halt_mode="soft",
        )
        # First 3 tokens have low scores (trigger window halt),
        # then continue until sentence end.
        scores = [0.4, 0.4, 0.4, 0.4, 0.4]
        idx = 0

        def callback(t):
            nonlocal idx
            s = scores[idx] if idx < len(scores) else 0.4
            idx += 1
            return s

        tokens = ["The", " sky", " is", " blue", "."]
        events = await self._collect_events(kernel, tokens, callback)
        assert events[-1].halted
        assert events[-1].token == "."
        assert len(events) == 5

    async def test_soft_halt_waits_for_sentence_boundary_without_break(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=2,
            window_threshold=0.55,
            halt_mode="soft",
        )
        scores = [0.4, 0.4, 0.4, 0.4, 0.4]
        score_iter = iter(scores)

        events = await self._collect_events_without_halt_break(
            kernel,
            ["First", " phrase", " ends", ". ", " more"],
            lambda _text: next(score_iter),
        )
        assert events[-1].halted
        assert events[-1].token == ". "

    async def test_soft_halt_cap_after_50_tokens(self):
        """Soft-halt stops after _SOFT_HALT_CAP tokens without boundary."""
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=3,
            window_threshold=0.55,
            halt_mode="soft",
        )
        # Generate enough tokens to trigger window halt then reach cap
        n = 3 + kernel._SOFT_HALT_CAP + 5
        tokens = [f"w{i}" for i in range(n)]

        events = await self._collect_events(kernel, tokens, lambda t: 0.4)
        assert events[-1].halted
        # 3 tokens to fill window + SOFT_HALT_CAP extra
        assert len(events) <= 3 + kernel._SOFT_HALT_CAP + 1

    async def test_soft_halt_cap_branch_without_break(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=2,
            window_threshold=0.55,
            halt_mode="soft",
        )
        events = await self._collect_events_without_halt_break(
            kernel,
            [f"w{i}" for i in range(70)],
            lambda _text: 0.4,
        )
        assert events[-1].halted
        assert events[-1].index >= 2
        assert len(events) <= 2 + kernel._SOFT_HALT_CAP + 2

    async def test_soft_halt_hard_limit_still_immediate(self):
        """Hard limit violations halt immediately even in soft mode."""
        kernel = AsyncStreamingKernel(
            hard_limit=0.3,
            window_size=10,
            window_threshold=0.5,
            halt_mode="soft",
        )
        scores = iter([0.8, 0.1])
        events = await self._collect_events(
            kernel,
            ["ok", "bad", "more"],
            lambda t: next(scores),
        )
        assert events[-1].halted
        assert len(events) == 2

    async def test_uses_shared_mean_helper_for_window_halt(self, monkeypatch):
        call_count = {"count": 0}
        original_mean = async_streaming_mod._mean

        def _counting_mean(values):
            call_count["count"] += 1
            return original_mean(values)

        monkeypatch.setattr(async_streaming_mod, "_mean", _counting_mean, raising=True)
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=3,
            window_threshold=0.5,
        )
        scores = iter([0.4, 0.4, 0.4, 0.4])
        events = await self._collect_events(
            kernel,
            ["a", "b", "c", "d"],
            lambda _t: next(scores),
        )
        assert events[-1].halted
        assert call_count["count"] >= 1

    async def test_halt_reason_window_average_path(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            window_size=3,
            window_threshold=0.55,
        )
        scores = iter([0.4, 0.4, 0.4, 0.4])
        session = await kernel.stream_to_session(
            ["a", "b", "c", "d"],
            lambda _text: next(scores),
        )
        assert session.halt_reason.startswith("window_avg")

    async def test_halt_reason_downward_trend_path(self):
        kernel = AsyncStreamingKernel(
            hard_limit=0.1,
            trend_window=3,
            trend_threshold=0.2,
            window_size=8,
        )
        scores = iter([0.9, 0.7, 0.5])
        session = await kernel.stream_to_session(
            ["a", "b", "c"],
            lambda _text: next(scores),
        )
        assert session.halt_reason.startswith("downward_trend")

    async def test_halt_reason_soft_flag(self):
        kernel = AsyncStreamingKernel()
        session = StreamSession(soft_halted=True, halted=True, halt_index=0)
        event = TokenEvent(token="x", index=0, coherence=0.5, timestamp=0.0)
        assert kernel._halt_reason(event, session) == "soft_halt"

    async def test_halt_reason_kernel_inactive(self):
        kernel = AsyncStreamingKernel()
        kernel.emergency_stop()
        session = StreamSession()
        event = TokenEvent(token="x", index=0, coherence=0.5, timestamp=0.0)
        assert kernel._halt_reason(event, session) == "kernel_inactive"

    async def test_halt_reason_fallback(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1)
        session = StreamSession(coherence_history=[0.9, 0.9])
        event = TokenEvent(token="x", index=0, coherence=0.9, timestamp=0.0)
        assert kernel._halt_reason(event, session) == "halt_condition_not_identified"
