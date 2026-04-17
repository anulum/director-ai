# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-token OpenTelemetry spans

"""OpenTelemetry child span per streamed token.

The span is a child of whichever ``director_ai.stream`` span is
active in the current context, so the token spans automatically
nest under the parent session span set up by
:func:`director_ai.core.otel.trace_streaming`.

Graceful no-op when ``opentelemetry-api`` is not installed or when
there is no active tracer — the context manager yields a stub that
swallows ``set_attribute`` / ``set_status`` calls so call sites do
not need their own branches.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec

from ..otel import _get_tracer, _NoopSpan

_OTEL_AVAILABLE = find_spec("opentelemetry.trace") is not None


@contextmanager
def trace_token(
    index: int,
    *,
    token: str = "",
    tenant_id: str = "",
    request_id: str = "",
) -> Iterator[object]:
    """Open a child span for one token. ``token`` is not set as an
    attribute — it may contain the model's raw output, which is
    typically inappropriate for log aggregators. Callers that need
    the token text should record a hash instead and keep the raw
    text out of the span.
    """
    tracer = _get_tracer()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span("director_ai.stream.token") as span:
        try:
            span.set_attribute("token.index", index)
            if tenant_id:
                span.set_attribute("tenant.id", tenant_id)
            if request_id:
                span.set_attribute("request.id", request_id)
            if token:
                span.set_attribute("token.length", len(token))
        except AttributeError:  # pragma: no cover — degraded span
            pass
        yield span


def is_otel_available() -> bool:
    """Expose the OTEL availability flag for tests and introspection."""
    return _OTEL_AVAILABLE
