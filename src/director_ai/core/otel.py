# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — OpenTelemetry Integration
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Optional OpenTelemetry bridge for Director-AI spans and metrics.

Graceful no-op when ``opentelemetry-api`` is not installed.

Usage::

    from director_ai.core.otel import setup_otel, trace_review

    setup_otel()

    with trace_review() as span:
        approved, score = scorer.review(prompt, response)
        span.set_attribute("coherence.score", score.score)
"""

from __future__ import annotations

from contextlib import contextmanager

try:
    from opentelemetry import trace

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_tracer = None


def setup_otel(service_name: str = "director-ai") -> None:
    """Configure OTel TracerProvider if the SDK is installed."""
    global _tracer
    if not _OTEL_AVAILABLE:
        return
    _tracer = trace.get_tracer(service_name)


@contextmanager
def trace_review():
    """Span around a CoherenceScorer.review() call."""
    if _tracer is None:
        yield _NoopSpan()
        return
    with _tracer.start_as_current_span("director_ai.review") as span:
        yield span


@contextmanager
def trace_streaming():
    """Span around a StreamingKernel session."""
    if _tracer is None:
        yield _NoopSpan()
        return
    with _tracer.start_as_current_span("director_ai.stream") as span:
        yield span


class _NoopSpan:
    """Attribute-sink when OTel is not available."""

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def set_status(self, *args: object, **kwargs: object) -> None:
        pass
