# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — OpenTelemetry Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch

from director_ai.core.otel import (
    _NoopSpan,
    setup_otel,
    trace_review,
    trace_streaming,
)


class TestOtelNoopFallback:
    def test_noop_span_set_attribute(self):
        span = _NoopSpan()
        span.set_attribute("foo", "bar")

    def test_trace_review_noop(self):
        import director_ai.core.otel as otel_mod

        otel_mod._tracer = None
        with trace_review() as span:
            span.set_attribute("coherence.score", 0.8)
        assert isinstance(span, _NoopSpan)

    def test_trace_streaming_noop(self):
        import director_ai.core.otel as otel_mod

        otel_mod._tracer = None
        with trace_streaming() as span:
            span.set_attribute("stream.halted", False)
        assert isinstance(span, _NoopSpan)


class TestOtelWithMock:
    def test_setup_otel_with_mock_sdk(self):
        import director_ai.core.otel as otel_mod

        mock_tracer = MagicMock()
        with (
            patch.object(otel_mod, "_OTEL_AVAILABLE", True),
            patch.object(otel_mod, "trace") as mock_trace,
        ):
            mock_trace.get_tracer.return_value = mock_tracer
            setup_otel("test-service")
            mock_trace.get_tracer.assert_called_once_with("test-service")
            assert otel_mod._tracer is mock_tracer
        otel_mod._tracer = None

    def test_trace_review_creates_span(self):
        import director_ai.core.otel as otel_mod

        mock_span = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_span)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_cm

        otel_mod._tracer = mock_tracer
        try:
            with trace_review() as span:
                span.set_attribute("coherence.score", 0.85)
            mock_tracer.start_as_current_span.assert_called_once_with(
                "director_ai.review"
            )
            mock_span.set_attribute.assert_called_once_with("coherence.score", 0.85)
        finally:
            otel_mod._tracer = None

    def test_trace_streaming_creates_span(self):
        import director_ai.core.otel as otel_mod

        mock_span = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_span)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_tracer = MagicMock()
        mock_tracer.start_as_current_span.return_value = mock_cm

        otel_mod._tracer = mock_tracer
        try:
            with trace_streaming() as span:
                span.set_attribute("stream.token_count", 42)
            mock_tracer.start_as_current_span.assert_called_once_with(
                "director_ai.stream"
            )
        finally:
            otel_mod._tracer = None

    def test_setup_otel_noop_when_unavailable(self):
        import director_ai.core.otel as otel_mod

        otel_mod._tracer = None
        with patch.object(otel_mod, "_OTEL_AVAILABLE", False):
            setup_otel()
        assert otel_mod._tracer is None
