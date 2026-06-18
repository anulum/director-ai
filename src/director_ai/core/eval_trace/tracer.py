# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Eval trace emitter

"""Emit a guard decision as an OpenTelemetry span and an eval record.

:func:`record_guard_decision` opens an OTel span (or a no-op when the SDK is
absent) carrying the :mod:`director_ai.core.eval_trace.attributes` schema, so an
OTLP-native evaluation tracer (Phoenix / Arize) ingests it directly. The same
attribute dict is returned as a plain *eval record* for tracers that take
metadata rather than OTLP spans (LangSmith, Ragas).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, cast

from ..otel import _get_tracer, _NoopSpan
from .attributes import GUARD_DECISION_SPAN, guard_decision_attributes

__all__ = ["eval_record_from_guard", "record_guard_decision"]


class _SpanLike(Protocol):
    """Span interface needed by the eval-trace bridge."""

    def set_attribute(self, key: str, value: object) -> None:
        """Attach one primitive attribute to the span."""
        ...


class _TracerLike(Protocol):
    """Tracer interface used without importing optional OTel SDK types."""

    def start_as_current_span(self, name: str) -> AbstractContextManager[_SpanLike]:
        """Open a current span context manager."""
        ...


@contextmanager
def record_guard_decision(
    attributes: dict[str, str | int | float | bool],
    *,
    span_name: str = GUARD_DECISION_SPAN,
) -> Iterator[_SpanLike]:
    """Open a span for a guard decision and attach the eval attributes.

    A no-op attribute sink is yielded when the OpenTelemetry SDK is not
    installed, so callers can always use this unconditionally.
    """
    get_tracer = cast(Callable[[], _TracerLike | None], _get_tracer)
    tracer = get_tracer()
    if tracer is None:
        noop_span = _NoopSpan()
        for key, value in attributes.items():
            noop_span.set_attribute(key, value)
        yield noop_span
        return
    with tracer.start_as_current_span(span_name) as active_span:
        for key, value in attributes.items():
            active_span.set_attribute(key, value)
        yield active_span


def eval_record_from_guard(
    result: Any,
    *,
    model: str = "",
    scorer: str = "",
    tenant_id: str = "",
    domain: str = "",
    answer_id: str = "",
) -> dict[str, str | int | float | bool]:
    """Build the eval attribute record from a ``GuardResult``.

    Reads the approval, score, and threshold from the result and the evidence
    and unsupported-claim counts from its coherence score. The returned dict is
    both the OTel span attributes and a metadata record for non-OTLP tracers.
    """
    coherence = result.coherence
    threshold = (
        result.calibrated_threshold if result.calibrated_threshold is not None else 0.0
    )
    evidence = getattr(coherence, "evidence", None)
    chunks = getattr(evidence, "chunks", None) if evidence is not None else None
    evidence_count = len(chunks) if chunks else 0
    unsupported = len(coherence.unsupported_claims)
    return guard_decision_attributes(
        decision="allow" if result.approved else "halt",
        approved=result.approved,
        score=result.score,
        threshold=threshold,
        model=model,
        scorer=scorer,
        tenant_id=tenant_id,
        domain=domain,
        evidence_count=evidence_count,
        unsupported_claims=unsupported,
        answer_id=answer_id,
    )
