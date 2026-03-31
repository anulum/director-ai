# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — OpenTelemetry Integration

"""Optional OpenTelemetry bridge for Director-AI spans and metrics.

Graceful no-op when ``opentelemetry-api`` is not installed.

Usage::

    from director_ai.core.otel import setup_otel, trace_review

    setup_otel()

    with trace_review() as span:
        approved, score = scorer.review(prompt, response)
        span.set_attribute("coherence.score", score.score)
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

try:
    from opentelemetry import trace

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_tracer = None
_tracer_lock = threading.Lock()


def setup_otel(service_name: str = "director-ai") -> None:
    """Configure OTel TracerProvider if the SDK is installed."""
    global _tracer
    if not _OTEL_AVAILABLE:
        return
    with _tracer_lock:
        _tracer = trace.get_tracer(service_name)


def _get_tracer():
    """Return tracer, lazy-initialising from global TracerProvider if needed."""
    global _tracer
    if _tracer is None and _OTEL_AVAILABLE:
        with _tracer_lock:
            if _tracer is None:
                _tracer = trace.get_tracer("director-ai")
    return _tracer


@contextmanager
def trace_review():
    """Span around a CoherenceScorer.review() call."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.review") as span:
        yield span


@contextmanager
def trace_streaming():
    """Span around a StreamingKernel session."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.stream") as span:
        yield span


@contextmanager
def trace_vector_query():
    """Span around a VectorStore query."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.vector_query") as span:
        yield span


@contextmanager
def trace_vector_add():
    """Span around a VectorStore add."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.vector_add") as span:
        yield span


class _NoopSpan:
    """Attribute-sink when OTel is not available."""

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def set_status(self, *args: object, **kwargs: object) -> None:
        pass
