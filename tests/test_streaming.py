# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Kernel Tests
"""Multi-angle tests for StreamingKernel real-time token gating.

Covers: token streaming, hard_limit halt, window_avg halt, trend detection,
soft halt, debug mode, halt evidence, pipeline integration with
CoherenceScorer and Rust backend, and performance documentation.
"""

import asyncio
from types import SimpleNamespace

import pytest

import director_ai.core.runtime.streaming as streaming_mod
from director_ai.core.async_streaming import AsyncStreamingKernel
from director_ai.core.observability.callbacks import TokenTraceCallback, TokenTraceEvent
from director_ai.core.streaming import StreamingKernel, TokenEvent
from director_ai.core.types import (
    CoherenceScore,
    EvidenceChunk,
    HaltEvidence,
    ScoringEvidence,
)


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
            window_size=3,
            window_threshold=0.6,
            on_halt=halted_sessions.append,
        )
        scores = iter([0.7, 0.55, 0.5, 0.45])
        kernel.stream_tokens([f"t{i} " for i in range(4)], lambda t: next(scores))
        assert len(halted_sessions) == 1
        assert "window_avg" in halted_sessions[0].halt_reason

    def test_soft_zone_warning_events(self):
        kernel = StreamingKernel(hard_limit=0.4, soft_limit=0.7)
        # Scores 0.5 and 0.6 are in soft zone (>= 0.4, < 0.7)
        scores = iter([0.5, 0.6, 0.8])
        tokens = ["a ", "b ", "c "]
        session = kernel.stream_tokens(tokens, lambda t: next(scores))
        assert not session.halted
        assert session.warning_count == 2
        assert session.events[0].warning is True
        assert session.events[1].warning is True
        assert session.events[2].warning is False

    def test_soft_zone_no_halt(self):
        kernel = StreamingKernel(hard_limit=0.3, soft_limit=0.6)
        scores = iter([0.4, 0.5])
        tokens = ["a ", "b "]
        session = kernel.stream_tokens(tokens, lambda t: next(scores))
        assert not session.halted
        assert session.warning_count == 2

    def test_soft_halt_finishes_sentence(self):
        kernel = StreamingKernel(
            hard_limit=0.3,
            window_size=3,
            window_threshold=0.6,
            halt_mode="soft",
        )
        # Window avg drops below 0.6 at token 3, then token 4 ends sentence.
        scores = iter([0.7, 0.55, 0.5, 0.45, 0.45])
        tokens = ["Start ", "of ", "sentence ", "end. ", "more "]
        session = kernel.stream_tokens(tokens, lambda t: next(scores))
        assert session.halted
        assert session.soft_halted
        assert "end." in session.output

    def test_soft_halt_finalizes_immediately_at_sentence_boundary(self):
        kernel = StreamingKernel(
            hard_limit=0.3,
            window_size=2,
            window_threshold=0.6,
            halt_mode="soft",
        )
        scores = iter([0.5, 0.5])

        session = kernel.stream_tokens(["start ", "end."], lambda _text: next(scores))

        assert session.halted
        assert session.soft_halted
        assert session.halt_index == 1
        assert session.output == "start end."

    def test_soft_halt_cap_at_50_tokens(self):
        kernel = StreamingKernel(
            hard_limit=0.3,
            window_size=3,
            window_threshold=0.6,
            halt_mode="soft",
        )
        # Window avg triggers halt, then 50+ tokens without sentence end
        scores = [0.7, 0.55, 0.5] + [0.55] * 60
        score_iter = iter(scores)
        tokens = ["a ", "b ", "c "] + ["word "] * 60
        session = kernel.stream_tokens(tokens, lambda t: next(score_iter))
        assert session.halted
        assert session.soft_halted
        # Should not process all 63 tokens — cap at halt_index + 50
        assert session.token_count <= 53 + 1

    def test_soft_halt_mode_default_is_hard(self):
        kernel = StreamingKernel()
        assert kernel.halt_mode == "hard"

    def test_hard_halt_still_immediate_in_soft_mode(self):
        kernel = StreamingKernel(hard_limit=0.5, halt_mode="soft")
        scores = iter([0.8, 0.3])
        session = kernel.stream_tokens(["ok ", "bad "], lambda t: next(scores))
        assert session.halted
        assert not session.soft_halted
        assert "hard_limit" in session.halt_reason

    def test_hard_limit_preempts_pending_soft_halt(self):
        kernel = StreamingKernel(
            hard_limit=0.3,
            window_size=2,
            window_threshold=0.6,
            halt_mode="soft",
        )
        scores = iter([0.5, 0.5, 0.2, 0.5])
        session = kernel.stream_tokens(
            ["soft ", "pending ", "hard ", "ignored."],
            lambda _text: next(scores),
        )

        assert session.halted
        assert not session.soft_halted
        assert session.halt_index == 2
        assert session.halt_reason.startswith("hard_limit")
        assert session.output == "soft pending "

    def test_inactive_kernel_records_halt_before_scoring(self):
        kernel = StreamingKernel()
        kernel.emergency_stop()

        session = kernel.stream_tokens(["ignored"], lambda _text: 0.9)

        assert session.halted
        assert session.halt_index == 0
        assert session.halt_reason == "kernel_inactive"
        assert session.events == []

    def test_callback_timeout_records_bounded_halt(self):
        kernel = StreamingKernel()

        def timeout(_text: str) -> float:
            raise TimeoutError("backend stalled")

        session = kernel.stream_tokens(["delayed"], timeout)

        assert session.halted
        assert session.halt_index == 0
        assert session.halt_reason == "callback_timeout"
        assert session.events == []

    def test_halt_evidence_callback_receives_accumulated_output(self):
        kernel = StreamingKernel(hard_limit=0.5)

        session = kernel.stream_tokens(
            ["unsafe"],
            lambda _text: 0.2,
            evidence_callback=lambda text: f"evidence:{text}",
        )

        assert session.halted
        assert session.halt_evidence == "evidence:unsafe"
        assert session.events[0].evidence == "evidence:unsafe"

    def test_invalid_halt_mode_raises(self):
        with pytest.raises(ValueError, match="halt_mode"):
            StreamingKernel(halt_mode="invalid")

    def test_invalid_max_cadence_raises(self):
        with pytest.raises(ValueError, match="max_cadence"):
            StreamingKernel(max_cadence=0)


@pytest.mark.consumer
class TestScoringCadence:
    def test_score_every_n_reduces_callbacks(self):
        call_count = 0

        def counting_cb(token):
            nonlocal call_count
            call_count += 1
            return 0.8

        kernel = StreamingKernel(score_every_n=4)
        tokens = [f"t{i} " for i in range(20)]
        session = kernel.stream_tokens(tokens, counting_cb)
        assert not session.halted
        assert call_count == 5  # tokens 0, 4, 8, 12, 16

    def test_score_every_n_default_scores_all(self):
        call_count = 0

        def counting_cb(token):
            nonlocal call_count
            call_count += 1
            return 0.8

        kernel = StreamingKernel()
        tokens = [f"t{i} " for i in range(20)]
        kernel.stream_tokens(tokens, counting_cb)
        assert call_count == 20

    def test_adaptive_increases_cadence(self):
        call_count = 0

        def counting_cb(token):
            nonlocal call_count
            call_count += 1
            return 0.9  # always high → cadence ramps up

        kernel = StreamingKernel(soft_limit=0.6, adaptive=True, max_cadence=8)
        tokens = [f"t{i} " for i in range(40)]
        kernel.stream_tokens(tokens, counting_cb)
        assert call_count < 40

    def test_adaptive_resets_on_low_score(self):
        scores = [0.9] * 15 + [0.4] + [0.9] * 24
        score_iter = iter(scores)
        call_count = 0

        def counting_cb(token):
            nonlocal call_count
            call_count += 1
            return next(score_iter)

        kernel = StreamingKernel(
            hard_limit=0.2,
            soft_limit=0.6,
            adaptive=True,
            max_cadence=8,
        )
        tokens = [f"t{i} " for i in range(40)]
        kernel.stream_tokens(tokens, counting_cb)
        # After the low score, cadence resets to 1, so more callbacks
        assert call_count > 5

    def test_adaptive_low_first_score_keeps_single_token_cadence(self):
        kernel = StreamingKernel(hard_limit=0.1, soft_limit=0.6, adaptive=True)

        session = kernel.stream_tokens(["low"], lambda _text: 0.5)

        assert not session.halted
        assert session.warning_count == 1

    def test_streaming_debug_records_window_and_trend_snapshot(self):
        kernel = StreamingKernel(streaming_debug=True, hard_limit=0.1)

        session = kernel.stream_tokens(["a", "b"], lambda _text: 0.8)

        assert not session.halted
        assert len(session.debug_log) == 2
        assert session.events[0].debug_info == session.debug_log[0]
        assert set(session.debug_log[0]) == {
            "index",
            "coherence",
            "window_avg",
            "trend_drop",
            "accumulated_tokens",
        }

    def test_invalid_score_every_n_raises(self):
        with pytest.raises(ValueError, match="score_every_n"):
            StreamingKernel(score_every_n=0)


class _RecordingTraceCallback(TokenTraceCallback):
    def __init__(self) -> None:
        self.events: list[TokenTraceEvent] = []
        self.ends: list[dict] = []

    def on_token(self, event: TokenTraceEvent) -> None:
        self.events.append(event)

    def on_stream_end(self, *, tenant_id: str, request_id: str, summary: dict) -> None:
        self.ends.append(
            {"tenant_id": tenant_id, "request_id": request_id, "summary": summary}
        )


class _NoChunkScoreScorer:
    def review(self, prompt: str, response: str):
        return False, CoherenceScore(
            score=0.2,
            approved=False,
            h_logical=0.8,
            h_factual=0.7,
            evidence=ScoringEvidence(
                chunks=[EvidenceChunk(text=response, distance=0.1, source=prompt)],
                nli_premise="premise",
                nli_hypothesis=response,
                nli_score=0.2,
                chunk_scores=None,
            ),
        )


class TestStreamingTraceHelpers:
    def test_math_helpers_use_python_floor_when_acceleration_disabled(
        self, monkeypatch
    ):
        monkeypatch.setattr(streaming_mod, "_RUST_TREND", False)

        assert streaming_mod._mean([0.2, 0.4, 0.6]) == pytest.approx(0.4)
        assert streaming_mod._trend_drop([0.9]) == 0.0
        assert streaming_mod._trend_drop([0.9, 0.7, 0.5]) == pytest.approx(0.4)
        assert streaming_mod._sum_float([0.25, 0.5, 1.25]) == pytest.approx(2.0)

    def test_accelerated_sum_helper_dispatches_to_rust_kernel(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_RUST_TREND", True)
        calls = {"values": []}

        def rust_sum(values: list[float]) -> float:
            calls["values"].append(values)
            return 3.5

        monkeypatch.setattr(streaming_mod, "_rust_sum_f64", rust_sum, raising=True)

        assert streaming_mod._sum_float([1.0, 2.0]) == pytest.approx(3.5)
        assert calls["values"] == [[1.0, 2.0]]

    def test_check_halt_reports_direct_window_average_breach(self):
        kernel = StreamingKernel(
            hard_limit=0.1,
            window_size=2,
            window_threshold=0.6,
            trend_window=10,
        )

        assert kernel.check_halt(0.5) is False
        assert kernel.check_halt(0.5) is True

    def test_fact_source_ignores_non_string_and_duplicate_sources(self):
        chunks = [
            EvidenceChunk(text="a", distance=0.2, source=" kb "),
            EvidenceChunk(text="b", distance=0.3, source="kb"),
            SimpleNamespace(source=123),
            EvidenceChunk(text="c", distance=0.1, source="doc"),
        ]

        assert StreamingKernel._fact_source(chunks) == "kb,doc"

    def test_trace_metrics_unknown_reason_has_no_threshold(self):
        threshold, margin = StreamingKernel()._trace_metrics(
            "manual_stop",
            TokenEvent(token="x", index=0, coherence=0.9, timestamp=0.0),
            [],
            streaming_mod.deque(),
        )

        assert threshold is None
        assert margin == 0.0

    def test_set_halt_otel_attributes_accepts_missing_span_setter_or_evidence(self):
        kernel = StreamingKernel()
        span = object()

        kernel._set_halt_otel_attributes(span, None)
        kernel._set_halt_otel_attributes(span, HaltEvidence("halt", 0.2, []))

    def test_set_halt_otel_attributes_records_absent_counterfactual(self):
        values = {}

        class Span:
            def set_attribute(self, key: str, value: object) -> None:
                values[key] = value

        evidence = HaltEvidence("halt", 0.2, [], trace_attribution=None)

        StreamingKernel._set_halt_otel_attributes(Span(), evidence)

        assert values == {"stream.counterfactual.available": False}

    def test_halt_with_scorer_accepts_chunks_without_chunk_scores(self):
        kernel = StreamingKernel(hard_limit=0.5)

        session = kernel.stream_tokens(
            ["unsafe"],
            lambda _text: 0.2,
            scorer=_NoChunkScoreScorer(),
            prompt="kb",
        )

        assert session.halted
        assert session.halt_evidence_structured is not None
        assert session.halt_evidence_structured.nli_scores is None
        assert session.safety_events

    def test_trace_callbacks_receive_tokens_and_stream_end_summary(self):
        callback = _RecordingTraceCallback()
        kernel = StreamingKernel(hard_limit=0.1)

        session = kernel.stream_tokens(
            ["a", "b"],
            lambda _text: 0.8,
            trace_callbacks=[callback],
            tenant_id="tenant-a",
            request_id="req-1",
        )

        assert not session.halted
        assert [event.token for event in callback.events] == ["a", "b"]
        assert callback.ends == [
            {
                "tenant_id": "tenant-a",
                "request_id": "req-1",
                "summary": {
                    "halted": False,
                    "soft_halted": False,
                    "halt_reason": "",
                    "token_count": 2,
                    "warning_count": 0,
                    "avg_coherence": pytest.approx(0.8),
                },
            }
        ]


class TestStreamingRustMean:
    def test_rust_mean_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_RUST_TREND", True)
        called = {"count": 0}

        def _mean(values: list[float]) -> float:
            called["count"] += 1
            return 0.75

        monkeypatch.setattr(streaming_mod, "_rust_mean", _mean, raising=True)
        assert streaming_mod._mean([0.5, 1.0]) == pytest.approx(0.75)
        assert called["count"] == 1

    def test_rust_mean_type_error_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_RUST_TREND", True)
        monkeypatch.setattr(
            streaming_mod,
            "_rust_mean",
            lambda _values: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        assert streaming_mod._mean([0.2, 0.4, 0.6]) == pytest.approx(0.4)


@pytest.mark.consumer
class TestLiveStreaming:
    """Live token generation via agent.stream()."""

    def test_mock_generator_stream_tokens(self):
        from director_ai.core.actor import MockGenerator

        gen = MockGenerator()

        async def run():
            tokens = []
            async for tok in gen.stream_tokens("test"):
                tokens.append(tok)
            return tokens

        tokens = asyncio.run(run())
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_agent_stream_yields_tuples(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()

        async def run():
            pairs = []
            async for tok, coh in agent.stream("What is 2+2?"):
                pairs.append((tok, coh))
            return pairs

        pairs = asyncio.run(run())
        assert len(pairs) > 0
        for tok, coh in pairs:
            assert isinstance(tok, str)
            assert isinstance(coh, float)

    def test_stream_halts_on_low_coherence(self):
        from collections.abc import AsyncIterator
        from unittest.mock import MagicMock

        from director_ai.core import CoherenceScore
        from director_ai.core.agent import CoherenceAgent

        mock_scorer = MagicMock()
        mock_scorer.review.return_value = (
            False,
            CoherenceScore(score=0.1, approved=False, h_logical=0.9, h_factual=0.9),
        )
        agent = CoherenceAgent(_scorer=mock_scorer)

        # Two claims. The guard scores at the claim boundary (a half-finished
        # sentence cannot be NLI-scored), so it must halt at the end of the first
        # bad claim and never deliver any token of the second.
        class _TwoClaimGenerator:
            async def stream_tokens(self, prompt: str) -> AsyncIterator[str]:
                for word in [
                    "the",
                    "moon",
                    "is",
                    "made",
                    "of",
                    "cheese.",
                    "second",
                    "claim",
                    "here",
                ]:
                    yield word

        agent.generator = _TwoClaimGenerator()

        async def run():
            pairs = []
            async for tok, coh in agent.stream("bad query"):
                pairs.append((tok, coh))
            return pairs

        pairs = asyncio.run(run())
        tokens = [t for t, _ in pairs]
        # halted at the first claim boundary: "cheese." delivered, "second" not
        assert tokens[-1] == "cheese."
        assert "second" not in tokens
        assert pairs[-1][1] < 0.5  # the completed claim scored below the hard limit

    def test_stream_rejects_empty_prompt(self):
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()

        async def run():
            async for _ in agent.stream(""):
                pass

        with pytest.raises(ValueError, match="non-empty string"):
            asyncio.run(run())

    def test_stream_fallback_without_stream_tokens(self):
        from director_ai.core.actor import MockGenerator
        from director_ai.core.agent import CoherenceAgent

        agent = CoherenceAgent()
        # Save and remove stream_tokens to trigger fallback
        original = MockGenerator.stream_tokens
        try:
            del MockGenerator.stream_tokens

            async def run():
                pairs = []
                async for tok, coh in agent.stream("test"):
                    pairs.append((tok, coh))
                return pairs

            pairs = asyncio.run(run())
            assert len(pairs) > 0
        finally:
            MockGenerator.stream_tokens = original


@pytest.mark.consumer
class TestAsyncStreamingKernel:
    def test_async_on_halt_fires(self):
        halted_sessions = []
        kernel = AsyncStreamingKernel(hard_limit=0.5, on_halt=halted_sessions.append)
        scores = iter([0.8, 0.3])

        async def run():
            return await kernel.stream_to_session(
                ["Good ", "Bad "],
                lambda t: next(scores),
            )

        session = asyncio.run(run())
        assert session.halted
        assert len(halted_sessions) == 1
        assert "hard_limit" in halted_sessions[0].halt_reason

    def test_async_on_halt_not_called_when_ok(self):
        halted_sessions = []
        kernel = AsyncStreamingKernel(hard_limit=0.5, on_halt=halted_sessions.append)

        async def run():
            return await kernel.stream_to_session(["a ", "b "], lambda t: 0.9)

        session = asyncio.run(run())
        assert not session.halted
        assert len(halted_sessions) == 0

    def test_async_soft_zone_warning(self):
        kernel = AsyncStreamingKernel(hard_limit=0.4, soft_limit=0.7)
        scores = iter([0.5, 0.8])

        async def run():
            return await kernel.stream_to_session(["a ", "b "], lambda t: next(scores))

        session = asyncio.run(run())
        assert not session.halted
        assert session.warning_count == 1


class TestStreamSessionDerivedState:
    """StreamSession derived fields preserve halt and timing semantics."""

    def test_output_truncates_at_hard_halt_index(self):
        from director_ai.core.streaming import StreamSession

        session = StreamSession()
        session.tokens = ["a", "b", "c"]
        session.halted = True
        session.halt_index = 2

        assert session.output == "ab"

    def test_output_keeps_full_sentence_for_soft_halt(self):
        from director_ai.core.streaming import StreamSession

        session = StreamSession()
        session.tokens = ["a", "b", "c"]
        session.halted = True
        session.soft_halted = True
        session.halt_index = 1

        assert session.output == "abc"

    def test_empty_session_metrics_are_zero(self):
        from director_ai.core.streaming import StreamSession

        session = StreamSession()

        assert session.avg_coherence == 0.0
        assert session.min_coherence == 0.0

    def test_duration_ms_uses_recorded_start_and_end_times(self):
        from director_ai.core.streaming import StreamSession

        session = StreamSession()
        session.start_time = 1.0
        session.end_time = 1.5

        assert session.duration_ms == 500.0


class TestAsyncStreamingKernelContracts:
    """AsyncStreamingKernel validates configuration and preserves halt state."""

    @staticmethod
    async def _tokens(values):
        for value in values:
            yield value

    def test_async_streaming_rejects_invalid_configuration(self):
        invalid_cases = [
            ({"hard_limit": 1.5}, "hard_limit"),
            ({"window_size": 0}, "window_size"),
            ({"window_threshold": 2.0}, "window_threshold"),
            ({"trend_window": 1}, "trend_window"),
            ({"trend_threshold": -1.0}, "trend_threshold"),
            ({"halt_mode": "unknown"}, "halt_mode"),
            ({"score_every_n": 0}, "score_every_n"),
            ({"max_cadence": 0}, "max_cadence"),
        ]

        for kwargs, message in invalid_cases:
            with pytest.raises(ValueError, match=message):
                AsyncStreamingKernel(**kwargs)

    def test_async_stream_to_session_records_hard_halt_reason_and_tokens(self):
        kernel = AsyncStreamingKernel(hard_limit=0.5)

        async def run():
            return await kernel.stream_to_session(
                self._tokens(["a", "b"]),
                lambda _token: 0.3,
            )

        session = asyncio.run(run())

        assert session.halted
        assert "hard_limit" in session.halt_reason
        assert session.tokens

    def test_async_stream_to_session_preserves_warning_count_without_halting(self):
        kernel = AsyncStreamingKernel(hard_limit=0.1, soft_limit=0.8)

        async def run():
            return await kernel.stream_to_session(
                self._tokens(["a", "b"]),
                lambda _token: 0.5,
            )

        session = asyncio.run(run())

        assert not session.halted
        assert session.warning_count == 2

    def test_async_streaming_accepts_sync_iterable_sources(self):
        kernel = AsyncStreamingKernel()

        async def run():
            events = []
            async for event in kernel.stream_tokens(["a", "b", "c"], lambda _t: 0.9):
                events.append(event)
            return events

        events = asyncio.run(run())

        assert [event.token for event in events] == ["a", "b", "c"]
        assert all(not event.halted for event in events)
