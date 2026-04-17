# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — observability package

"""Per-token observability: OTEL child spans + pluggable callbacks.

The :mod:`.tracing` module opens an OpenTelemetry child span per
token emitted by :class:`~director_ai.core.runtime.streaming.StreamingKernel`.
The :mod:`.callbacks` module defines the
:class:`TokenTraceCallback` protocol together with adapters for
Langfuse-style sinks; callers register callbacks and receive one
structured event per token.

Both paths are no-ops when their optional dependencies are not
installed, so the production scoring pipeline never pays for
observability it did not ask for.
"""

from .callbacks import (
    LangfuseTokenCallback,
    TokenTraceCallback,
    TokenTraceEmitter,
    TokenTraceEvent,
)
from .tracing import trace_token

__all__ = [
    "LangfuseTokenCallback",
    "TokenTraceCallback",
    "TokenTraceEmitter",
    "TokenTraceEvent",
    "trace_token",
]
