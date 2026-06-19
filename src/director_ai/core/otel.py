# SPDX-License-Identifier: Apache-2.0
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
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

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


def _get_tracer() -> Any:
    """Return tracer, lazy-initialising from global TracerProvider if needed."""
    global _tracer
    if _tracer is None and _OTEL_AVAILABLE:
        with _tracer_lock:
            if _tracer is None:
                _tracer = trace.get_tracer("director-ai")
    return _tracer


@contextmanager
def _trace_named_span(
    name: str, attributes: dict[str, object] | None = None
) -> Iterator[Any]:
    """Open an optional OTel span and attach primitive attributes."""
    tracer = _get_tracer()
    if tracer is None:
        span = _NoopSpan()
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span
        return
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        yield span


@contextmanager
def trace_review() -> Iterator[Any]:
    """Span around a CoherenceScorer.review() call."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.review") as span:
        yield span


@contextmanager
def trace_streaming() -> Iterator[Any]:
    """Span around a StreamingKernel session."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.stream") as span:
        yield span


@contextmanager
def trace_vector_query() -> Iterator[Any]:
    """Span around a VectorStore query."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.vector_query") as span:
        yield span


@contextmanager
def trace_vector_add() -> Iterator[Any]:
    """Span around a VectorStore add."""
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.vector_add") as span:
        yield span


def trace_cache(
    *, hit: bool | None = None, scope_present: bool | None = None
) -> AbstractContextManager[Any]:
    """Span around score-cache lookup decisions."""
    attrs: dict[str, object] = {}
    if hit is not None:
        attrs["cache.hit"] = hit
    if scope_present is not None:
        attrs["cache.scope_present"] = scope_present
    return _trace_named_span("director_ai.cache", attrs)


def trace_retrieval(
    *, top_k: int | None = None, tenant_scoped: bool | None = None
) -> AbstractContextManager[Any]:
    """Span around grounding-context retrieval."""
    attrs: dict[str, object] = {}
    if top_k is not None:
        attrs["retrieval.top_k"] = top_k
    if tenant_scoped is not None:
        attrs["retrieval.tenant_scoped"] = tenant_scoped
    return _trace_named_span("director_ai.retrieval", attrs)


def trace_nli_inference(*, stage: str) -> AbstractContextManager[Any]:
    """Span around NLI inference for a named scoring stage."""
    return _trace_named_span("director_ai.nli", {"nli.stage": stage})


def trace_calibration(*, stage: str) -> AbstractContextManager[Any]:
    """Span around calibration/meta-confidence transforms."""
    return _trace_named_span("director_ai.calibration", {"calibration.stage": stage})


def trace_judge(*, provider: str) -> AbstractContextManager[Any]:
    """Span around LLM-as-judge escalation."""
    return _trace_named_span("director_ai.judge", {"judge.provider": provider})


class _NoopSpan:
    """Attribute-sink when OTel is not available."""

    def set_attribute(self, key: str, value: object) -> None:
        pass

    def set_status(self, *args: object, **kwargs: object) -> None:
        pass
