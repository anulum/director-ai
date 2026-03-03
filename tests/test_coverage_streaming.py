"""Coverage tests for streaming.py — StreamingKernel."""

from __future__ import annotations

import pytest

from director_ai.core.streaming import StreamSession, StreamingKernel, TokenEvent


class TestStreamingKernelValidation:
    def test_bad_halt_mode(self):
        with pytest.raises(ValueError, match="halt_mode"):
            StreamingKernel(halt_mode="unknown")

    def test_bad_score_every_n(self):
        with pytest.raises(ValueError, match="score_every_n"):
            StreamingKernel(score_every_n=0)

    def test_bad_max_cadence(self):
        with pytest.raises(ValueError, match="max_cadence"):
            StreamingKernel(max_cadence=0)


class TestStreamSessionProperties:
    def test_output_no_halt(self):
        s = StreamSession()
        s.tokens = ["a", "b", "c"]
        assert s.output == "abc"

    def test_output_hard_halt(self):
        s = StreamSession()
        s.tokens = ["a", "b", "c"]
        s.halted = True
        s.halt_index = 2
        assert s.output == "ab"

    def test_output_soft_halt(self):
        s = StreamSession()
        s.tokens = ["a", "b", "c"]
        s.soft_halted = True
        s.halted = True
        s.halt_index = 1
        assert s.output == "abc"

    def test_avg_coherence_empty(self):
        assert StreamSession().avg_coherence == 0.0

    def test_min_coherence_empty(self):
        assert StreamSession().min_coherence == 0.0

    def test_duration_ms(self):
        s = StreamSession()
        s.start_time = 1.0
        s.end_time = 1.5
        assert s.duration_ms == 500.0


class TestStreamingKernelCheckHalt:
    def test_hard_limit(self):
        k = StreamingKernel(hard_limit=0.5)
        assert k.check_halt(0.3)

    def test_window_avg(self):
        k = StreamingKernel(hard_limit=0.1, window_size=3, window_threshold=0.6)
        for s in [0.55, 0.55]:
            k.check_halt(s)
        assert k.check_halt(0.55)

    def test_trend(self):
        k = StreamingKernel(hard_limit=0.1, trend_window=3, trend_threshold=0.1)
        k.check_halt(0.9)
        k.check_halt(0.8)
        assert k.check_halt(0.7)

    def test_no_halt(self):
        k = StreamingKernel(hard_limit=0.1)
        assert not k.check_halt(0.9)

    def test_reset_state(self):
        k = StreamingKernel()
        k.check_halt(0.5)
        k.reset_state()
        assert len(k._window) == 0


class TestStreamingKernelStreamTokens:
    def test_happy_path(self):
        k = StreamingKernel()
        session = k.stream_tokens(["a", "b", "c"], lambda t: 0.9)
        assert session.token_count == 3
        assert not session.halted

    def test_hard_halt(self):
        k = StreamingKernel(hard_limit=0.5)
        session = k.stream_tokens(["a", "b"], lambda t: 0.3)
        assert session.halted
        assert "hard_limit" in session.halt_reason

    def test_soft_halt_mode(self):
        k = StreamingKernel(
            hard_limit=0.1,
            window_size=2,
            window_threshold=0.7,
            halt_mode="soft",
        )
        session = k.stream_tokens(
            ["word. ", "other. "] * 5,
            lambda t: 0.55,
        )
        assert session.soft_halted or session.halted

    def test_soft_halt_at_sentence_boundary(self):
        k = StreamingKernel(
            hard_limit=0.1,
            window_size=2,
            window_threshold=0.7,
            halt_mode="soft",
        )
        session = k.stream_tokens(
            ["word. "],
            lambda t: 0.55,
        )
        # Not enough tokens to trigger window, so no halt
        assert session.token_count >= 1

    def test_inactive_kernel(self):
        k = StreamingKernel()
        k.emergency_stop()
        session = k.stream_tokens(["a", "b"], lambda t: 0.9)
        assert session.halted
        assert session.halt_reason == "kernel_inactive"

    def test_on_halt_callback(self):
        called = []
        k = StreamingKernel(hard_limit=0.5, on_halt=lambda s: called.append(True))
        k.stream_tokens(["a"], lambda t: 0.3)
        assert called

    def test_warning_count(self):
        k = StreamingKernel(hard_limit=0.1, soft_limit=0.8)
        session = k.stream_tokens(["a", "b"], lambda t: 0.5)
        assert session.warning_count == 2

    def test_debug_mode(self):
        k = StreamingKernel(streaming_debug=True)
        session = k.stream_tokens(["a", "b"], lambda t: 0.9)
        assert session.debug_log
        assert session.events[0].debug_info is not None

    def test_score_every_n(self):
        k = StreamingKernel(score_every_n=3)
        call_count = 0

        def counting_cb(t):
            nonlocal call_count
            call_count += 1
            return 0.9

        session = k.stream_tokens([f"t{i}" for i in range(6)], counting_cb)
        assert session.token_count == 6
        assert call_count == 2  # tokens 0 and 3

    def test_adaptive_cadence_increase(self):
        k = StreamingKernel(adaptive=True, score_every_n=1, max_cadence=4, soft_limit=0.7)
        session = k.stream_tokens([f"t{i}" for i in range(10)], lambda t: 0.9)
        assert session.token_count == 10

    def test_adaptive_cadence_reset_on_low_score(self):
        k = StreamingKernel(
            adaptive=True, score_every_n=1, max_cadence=4,
            soft_limit=0.7, hard_limit=0.1,
        )
        session = k.stream_tokens([f"t{i}" for i in range(6)], lambda t: 0.5)
        assert session.token_count == 6

    def test_evidence_callback(self):
        k = StreamingKernel(hard_limit=0.5)
        session = k.stream_tokens(
            ["a"], lambda t: 0.3,
            evidence_callback=lambda text: "bad evidence",
        )
        assert session.halted
        assert session.halt_evidence == "bad evidence"

    def test_stream_output(self):
        k = StreamingKernel()
        out = k.stream_output(["hello ", "world"], lambda t: 0.9)
        assert out == "hello world"

    def test_stream_output_halt(self):
        k = StreamingKernel(hard_limit=0.5)
        out = k.stream_output(["hello"], lambda t: 0.3)
        assert "KERNEL INTERRUPT" in out

    def test_suggested_action_hard_limit(self):
        assert "temperature" in StreamingKernel._suggested_action("hard_limit breach")

    def test_suggested_action_window_avg(self):
        assert "drifting" in StreamingKernel._suggested_action("window_avg too low")

    def test_suggested_action_trend(self):
        assert "rephrase" in StreamingKernel._suggested_action("downward_trend")

    def test_suggested_action_unknown(self):
        assert "Review" in StreamingKernel._suggested_action("other")
