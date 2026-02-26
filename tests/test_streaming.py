# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Streaming Kernel Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.streaming import StreamingKernel


@pytest.mark.consumer
class TestStreamingKernel:
    def test_normal_stream(self):
        kernel = StreamingKernel()
        tokens = ["Hello ", "world ", "how ", "are ", "you"]
        session = kernel.stream_tokens(tokens, lambda t: 0.8)
        assert not session.halted
        assert session.output == "Hello world how are you"
        assert session.token_count == 5

    def test_hard_limit_halt(self):
        kernel = StreamingKernel(hard_limit=0.5)
        tokens = ["Good ", "Bad "]
        scores = iter([0.8, 0.3])
        session = kernel.stream_tokens(tokens, lambda t: next(scores))
        assert session.halted
        assert "hard_limit" in session.halt_reason

    def test_window_average_halt(self):
        kernel = StreamingKernel(window_size=3, window_threshold=0.6)
        # Feed tokens with declining scores that average below threshold
        scores = [0.7, 0.55, 0.5, 0.45]
        score_iter = iter(scores)
        tokens = [f"tok{i} " for i in range(4)]
        session = kernel.stream_tokens(tokens, lambda t: next(score_iter))
        assert session.halted
        assert "window_avg" in session.halt_reason

    def test_downward_trend_halt(self):
        kernel = StreamingKernel(trend_window=3, trend_threshold=0.1)
        # Scores dropping by > 0.1 over 3 tokens
        scores = [0.85, 0.80, 0.70]
        score_iter = iter(scores)
        tokens = ["a ", "b ", "c "]
        session = kernel.stream_tokens(tokens, lambda t: next(score_iter))
        assert session.halted
        assert "downward_trend" in session.halt_reason

    def test_session_metrics(self):
        kernel = StreamingKernel()
        tokens = ["Hello ", "world"]
        session = kernel.stream_tokens(tokens, lambda t: 0.9)
        assert session.avg_coherence == pytest.approx(0.9)
        assert session.min_coherence == pytest.approx(0.9)
        assert session.duration_ms >= 0

    def test_partial_output_on_halt(self):
        kernel = StreamingKernel(hard_limit=0.5)
        scores = [0.8, 0.8, 0.3]
        score_iter = iter(scores)
        tokens = ["Good ", "Good ", "Bad "]
        session = kernel.stream_tokens(tokens, lambda t: next(score_iter))
        assert session.halted
        assert session.output == "Good Good "

    def test_backward_compat_stream_output(self):
        kernel = StreamingKernel(hard_limit=0.5)
        output = kernel.stream_output(["Hello ", "world"], lambda t: 0.8)
        assert output == "Hello world"

    def test_backward_compat_halt_message(self):
        kernel = StreamingKernel(hard_limit=0.5)
        output = kernel.stream_output(["Bad "], lambda t: 0.3)
        assert "KERNEL INTERRUPT" in output

    def test_events_tracked(self):
        kernel = StreamingKernel()
        tokens = ["a", "b", "c"]
        session = kernel.stream_tokens(tokens, lambda t: 0.85)
        assert len(session.events) == 3
        for event in session.events:
            assert event.coherence == pytest.approx(0.85)
            assert not event.halted

    def test_on_halt_callback_fires(self):
        halted_sessions = []
        kernel = StreamingKernel(hard_limit=0.5, on_halt=halted_sessions.append)
        scores = iter([0.8, 0.3])
        kernel.stream_tokens(["Good ", "Bad "], lambda t: next(scores))
        assert len(halted_sessions) == 1
        assert halted_sessions[0].halted
        assert "hard_limit" in halted_sessions[0].halt_reason

    def test_on_halt_not_called_when_no_halt(self):
        halted_sessions = []
        kernel = StreamingKernel(on_halt=halted_sessions.append)
        kernel.stream_tokens(["a", "b"], lambda t: 0.9)
        assert len(halted_sessions) == 0

    def test_on_halt_window_avg(self):
        halted_sessions = []
        kernel = StreamingKernel(
            window_size=3, window_threshold=0.6, on_halt=halted_sessions.append
        )
        scores = iter([0.7, 0.55, 0.5, 0.45])
        kernel.stream_tokens([f"t{i} " for i in range(4)], lambda t: next(scores))
        assert len(halted_sessions) == 1
        assert "window_avg" in halted_sessions[0].halt_reason
