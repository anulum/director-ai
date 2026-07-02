# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — OpenTelemetry real-surface tests
"""Real OpenTelemetry SDK coverage for Director-AI tracing helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.types import AttributeValue

import director_ai.core.otel as otel_mod
from director_ai.core.otel import (
    _NoopSpan,
    setup_otel,
    trace_cache,
    trace_calibration,
    trace_judge,
    trace_nli_inference,
    trace_retrieval,
    trace_review,
    trace_streaming,
    trace_vector_add,
    trace_vector_query,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


@contextmanager
def _recorded_spans() -> Iterator[InMemorySpanExporter]:
    """Install a real SDK provider and yield its in-memory span exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    previous_provider = trace.get_tracer_provider()
    trace._TRACER_PROVIDER = provider
    otel_mod._tracer = None
    try:
        yield exporter
    finally:
        otel_mod._tracer = None
        trace._TRACER_PROVIDER = previous_provider


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, ReadableSpan]:
    """Return exported spans keyed by their production span names."""
    spans = exporter.get_finished_spans()
    return {span.name: span for span in spans}


def _attributes(span: ReadableSpan) -> Mapping[str, AttributeValue]:
    """Return non-null span attributes from an exported SDK span."""
    attributes = span.attributes
    assert attributes is not None
    return attributes


def test_otel_unit_guard_has_real_surface_companion() -> None:
    """The OTel unit guard should be backed by real SDK exporter coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS["tests/test_otel.py"]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_otel_real_surface.py" in category


def test_public_otel_helpers_emit_real_sdk_spans_with_attributes() -> None:
    """Public tracing helpers should export spans through the OTel SDK."""
    with _recorded_spans() as exporter:
        setup_otel("director-ai-real-surface")
        with trace_review() as review_span:
            review_span.set_attribute("coherence.score", 0.91)
        with trace_streaming() as streaming_span:
            streaming_span.set_attribute("stream.token_count", 3)
        with trace_vector_query() as query_span:
            query_span.set_attribute("vector.backend", "memory")
        with trace_vector_add() as add_span:
            add_span.set_attribute("vector.doc_id", "fact-1")

    spans = _spans_by_name(exporter)

    assert set(spans) == {
        "director_ai.review",
        "director_ai.stream",
        "director_ai.vector_add",
        "director_ai.vector_query",
    }
    assert _attributes(spans["director_ai.review"])["coherence.score"] == 0.91
    assert _attributes(spans["director_ai.stream"])["stream.token_count"] == 3
    assert _attributes(spans["director_ai.vector_query"])["vector.backend"] == "memory"
    assert _attributes(spans["director_ai.vector_add"])["vector.doc_id"] == "fact-1"


def test_optional_stage_attributes_may_be_absent_on_real_sdk_spans() -> None:
    """Optional stage attributes should not block real SDK span export."""
    with _recorded_spans() as exporter:
        setup_otel("director-ai-optional-stage-surface")
        with trace_cache():
            pass
        with trace_retrieval():
            pass

    spans = _spans_by_name(exporter)

    assert _attributes(spans["director_ai.cache"]) == {}
    assert _attributes(spans["director_ai.retrieval"]) == {}


def test_optional_stage_attributes_may_be_absent_without_otel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional attributes should still yield noop fallback spans."""
    otel_mod._tracer = None
    monkeypatch.setattr(otel_mod, "_OTEL_AVAILABLE", False)

    with trace_cache() as cache_span:
        pass
    with trace_retrieval() as retrieval_span:
        pass

    assert isinstance(cache_span, _NoopSpan)
    assert isinstance(retrieval_span, _NoopSpan)


def test_stage_tracing_helpers_export_real_sdk_attribute_spans() -> None:
    """Stage helpers should attach production attributes to exported spans."""
    with _recorded_spans() as exporter:
        setup_otel("director-ai-stage-surface")
        with trace_cache(hit=False, scope_present=True):
            pass
        with trace_retrieval(top_k=4, tenant_scoped=True):
            pass
        with trace_nli_inference(stage="logical"):
            pass
        with trace_calibration(stage="temperature"):
            pass
        with trace_judge(provider="local"):
            pass

    spans = _spans_by_name(exporter)

    assert _attributes(spans["director_ai.cache"])["cache.hit"] is False
    assert _attributes(spans["director_ai.cache"])["cache.scope_present"] is True
    assert _attributes(spans["director_ai.retrieval"])["retrieval.top_k"] == 4
    assert (
        _attributes(spans["director_ai.retrieval"])["retrieval.tenant_scoped"] is True
    )
    assert _attributes(spans["director_ai.nli"])["nli.stage"] == "logical"
    assert _attributes(spans["director_ai.calibration"])["calibration.stage"] == (
        "temperature"
    )
    assert _attributes(spans["director_ai.judge"])["judge.provider"] == "local"
