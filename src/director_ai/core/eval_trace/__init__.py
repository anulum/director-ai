# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Eval trace package

"""OpenTelemetry eval trace standard.

Emit a guard decision as an OTel span carrying a stable ``director.eval.*`` +
``gen_ai.*`` attribute schema, so an OTLP-native evaluation tracer (Phoenix /
Arize) ingests it directly and metadata-based tracers (LangSmith, Ragas) consume
the same attribute dict.
"""

from __future__ import annotations

from .attributes import (
    EVAL_SCHEMA_VERSION,
    GUARD_DECISION_SPAN,
    guard_decision_attributes,
)
from .tracer import eval_record_from_guard, record_guard_decision

__all__ = [
    "EVAL_SCHEMA_VERSION",
    "GUARD_DECISION_SPAN",
    "eval_record_from_guard",
    "guard_decision_attributes",
    "record_guard_decision",
]
