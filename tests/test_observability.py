# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-token observability tests

"""Multi-angle coverage of the observability package: emitter
fan-out, callback failure isolation, Langfuse-shaped adapter,
StreamingKernel integration with trace_callbacks, and the OTEL
no-op path when the tracer is absent."""

from __future__ import annotations

from typing import Any

import pytest

from director_ai.core.observability import (
    LangfuseTokenCallback,
    TokenTraceCallback,
    TokenTraceEmitter,
    TokenTraceEvent,
    trace_token,
)
from director_ai.core.runtime.streaming import StreamingKernel

# --- TokenTraceEvent -------------------------------------------------


class TestTokenTraceEvent:
    def test_token_hash_stable_and_short(self):
        e = TokenTraceEvent(index=0, token="hello", coherence=0.9, timestamp=0.0)
        h = e.token_hash()
        assert len(h) == 16
        assert e.token_hash() == h  # deterministic

    def test_different_tokens_different_hashes(self):
        a = TokenTraceEvent(index=0, token="a", coherence=0.9, timestamp=0.0)
        b = TokenTraceEvent(index=0, token="b", coherence=0.9, timestamp=0.0)
        assert a.token_hash() != b.token_hash()


# --- TokenTraceEmitter -----------------------------------------------


class _Recorder(TokenTraceCallback):
    def __init__(self) -> None:
        self.tokens: list[TokenTraceEvent] = []
        self.ends: list[dict[str, Any]] = []

    def on_token(self, event: TokenTraceEvent) -> None:
        self.tokens.append(event)

    def on_stream_end(
        self, *, tenant_id: str, request_id: str, summary: dict[str, Any]
    ) -> None:
        self.ends.append(
            {"tenant_id": tenant_id, "request_id": request_id, "summary": summary}
        )


class _Explosive(TokenTraceCallback):
    def on_token(self, event: TokenTraceEvent) -> None:
        raise RuntimeError("sink broken")

    def on_stream_end(
        self, *, tenant_id: str, request_id: str, summary: dict[str, Any]
    ) -> None:
        raise RuntimeError("end broken")


class TestEmitter:
    def test_empty_emitter_reports_disabled(self):
        e = TokenTraceEmitter()
        assert not e.enabled
        assert len(e) == 0

    def test_fanout_to_every_callback(self):
        a, b = _Recorder(), _Recorder()
        e = TokenTraceEmitter([a, b])
        assert e.enabled
        assert len(e) == 2
        ev = TokenTraceEvent(index=0, token="t", coherence=0.5, timestamp=0.0)
        e.emit(ev)
        assert a.tokens == [ev] == b.tokens

    def test_register_and_unregister(self):
        a = _Recorder()
        e = TokenTraceEmitter()
        e.register(a)
        assert len(e) == 1
        e.unregister(a)
        assert len(e) == 0
        # unregistering twice is a no-op
        e.unregister(a)

    def test_register_rejects_non_callback(self):
        e = TokenTraceEmitter()
        with pytest.raises(TypeError, match="must subclass"):
            e.register(object())  # type: ignore[arg-type]

    def test_broken_callback_does_not_break_others(self):
        a = _Recorder()
        e = TokenTraceEmitter([_Explosive(), a])
        ev = TokenTraceEvent(index=0, token="t", coherence=0.5, timestamp=0.0)
        e.emit(ev)  # must not raise
        assert a.tokens == [ev]

    def test_end_swallows_exceptions(self):
        a = _Recorder()
        e = TokenTraceEmitter([_Explosive(), a])
        e.end(tenant_id="t", request_id="r", summary={"halted": False})
        assert a.ends == [
            {"tenant_id": "t", "request_id": "r", "summary": {"halted": False}}
        ]


# --- LangfuseTokenCallback -------------------------------------------


class _FakeLangfuseTrace:
    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []

    def span(self, name: str, metadata: dict[str, Any], start_time: float) -> None:
        self.spans.append(
            {"name": name, "metadata": metadata, "start_time": start_time}
        )

    def score(
        self, name: str, value: float, comment: str | None = None
    ) -> None:
        self.scores.append({"name": name, "value": value, "comment": comment})

    def update(self, status_message: str, metadata: dict[str, Any]) -> None:
        self.updates.append(
            {"status_message": status_message, "metadata": metadata}
        )


class _FakeLangfuse:
    def __init__(self) -> None:
        self.created_traces: list[_FakeLangfuseTrace] = []

    def trace(self, name: str, user_id: str, metadata: dict[str, Any]) -> _FakeLangfuseTrace:
        t = _FakeLangfuseTrace()
        t.name = name  # type: ignore[attr-defined]
        t.user_id = user_id  # type: ignore[attr-defined]
        t.metadata = metadata  # type: ignore[attr-defined]
        self.created_traces.append(t)
        return t


class TestLangfuseAdapter:
    def test_records_span_and_score_per_token(self):
        lf = _FakeLangfuse()
        cb = LangfuseTokenCallback(lf)
        cb.on_token(
            TokenTraceEvent(index=0, token="Paris", coherence=0.88, timestamp=1.0, tenant_id="t1", request_id="r1")
        )
        cb.on_token(
            TokenTraceEvent(index=1, token=" is", coherence=0.86, timestamp=1.5, tenant_id="t1", request_id="r1")
        )
        assert len(lf.created_traces) == 1
        trace = lf.created_traces[0]
        assert trace.user_id == "t1"
        assert len(trace.spans) == 2
        assert trace.spans[0]["metadata"]["index"] == 0
        assert trace.spans[0]["metadata"]["coherence"] == pytest.approx(0.88)
        assert "token_hash" in trace.spans[0]["metadata"]
        assert "token" not in trace.spans[0]["metadata"]
        assert [s["name"] for s in trace.scores] == ["coherence", "coherence"]

    def test_record_token_text_is_opt_in(self):
        lf = _FakeLangfuse()
        cb = LangfuseTokenCallback(lf, record_token_text=True)
        cb.on_token(
            TokenTraceEvent(index=0, token="secret", coherence=0.9, timestamp=0.0)
        )
        assert lf.created_traces[0].spans[0]["metadata"]["token"] == "secret"

    def test_parallel_streams_get_separate_traces(self):
        lf = _FakeLangfuse()
        cb = LangfuseTokenCallback(lf)
        cb.on_token(
            TokenTraceEvent(index=0, token="a", coherence=0.9, timestamp=0.0, request_id="r1")
        )
        cb.on_token(
            TokenTraceEvent(index=0, token="a", coherence=0.9, timestamp=0.0, request_id="r2")
        )
        assert len(lf.created_traces) == 2

    def test_on_stream_end_updates_trace_and_frees_handle(self):
        lf = _FakeLangfuse()
        cb = LangfuseTokenCallback(lf)
        cb.on_token(
            TokenTraceEvent(index=0, token="a", coherence=0.9, timestamp=0.0, request_id="r1")
        )
        cb.on_stream_end(
            tenant_id="t",
            request_id="r1",
            summary={"halted": True, "token_count": 1, "halt_reason": "hard_limit"},
        )
        trace = lf.created_traces[0]
        assert trace.updates[0]["status_message"] == "halted"
        # subsequent tokens on the same request spawn a new trace
        cb.on_token(
            TokenTraceEvent(index=0, token="b", coherence=0.9, timestamp=0.0, request_id="r1")
        )
        assert len(lf.created_traces) == 2

    def test_rejects_none_client(self):
        with pytest.raises(ValueError, match="client is required"):
            LangfuseTokenCallback(None)  # type: ignore[arg-type]


# --- trace_token no-op path ------------------------------------------


class TestTraceToken:
    def test_context_manager_yields_span_object(self):
        with trace_token(0, token="t") as span:
            # either a real span or _NoopSpan — both support set_attribute
            span.set_attribute("x.y", 1)


# --- StreamingKernel integration -------------------------------------


class TestStreamingIntegration:
    def test_callback_fires_for_every_token(self):
        rec = _Recorder()
        kernel = StreamingKernel(hard_limit=0.3)
        tokens = ["a ", "b ", "c"]
        session = kernel.stream_tokens(
            iter(tokens),
            coherence_callback=lambda _acc: 0.9,
            trace_callbacks=[rec],
            tenant_id="tenant-x",
            request_id="req-1",
        )
        assert not session.halted
        assert len(rec.tokens) == len(tokens)
        for i, ev in enumerate(rec.tokens):
            assert ev.index == i
            assert ev.tenant_id == "tenant-x"
            assert ev.request_id == "req-1"
            assert ev.halted is False
        assert len(rec.ends) == 1
        assert rec.ends[0]["summary"]["token_count"] == len(tokens)
        assert rec.ends[0]["summary"]["halted"] is False

    def test_callback_receives_halt_event(self):
        rec = _Recorder()
        kernel = StreamingKernel(hard_limit=0.7)
        # First token scores high, second fails.
        scores = iter([0.9, 0.1])
        session = kernel.stream_tokens(
            iter(["a", "b"]),
            coherence_callback=lambda _acc: next(scores),
            trace_callbacks=[rec],
            request_id="r-1",
        )
        assert session.halted
        assert rec.tokens[-1].halted is True
        assert "hard_limit" in rec.tokens[-1].halt_reason
        assert rec.ends[0]["summary"]["halt_reason"].startswith("hard_limit")

    def test_callback_exception_does_not_break_stream(self):
        kernel = StreamingKernel(hard_limit=0.3)
        session = kernel.stream_tokens(
            iter(["a", "b"]),
            coherence_callback=lambda _acc: 0.9,
            trace_callbacks=[_Explosive()],
        )
        assert not session.halted
        assert session.token_count == 2

    def test_no_callbacks_still_runs(self):
        """No-callback path is the zero-overhead default."""
        kernel = StreamingKernel(hard_limit=0.3)
        session = kernel.stream_tokens(
            iter(["a", "b"]),
            coherence_callback=lambda _acc: 0.9,
        )
        assert session.token_count == 2
