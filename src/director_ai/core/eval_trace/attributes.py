# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Eval trace attribute schema

"""A stable, versioned span-attribute schema for a guard decision.

Emitting a guard decision as an OpenTelemetry span only helps if the attributes
are conventional enough for an external evaluation tracer to read. This module
defines the attribute keys — Director's own ``director.eval.*`` fields plus the
``gen_ai.*`` semantic-convention keys that Phoenix/Arize and other OTLP-native
tracers already understand — and builds the tenant-safe attribute dict for one
decision. All values are OTel-primitive (str / int / float / bool); no raw
prompt, answer, or chunk text is included.
"""

from __future__ import annotations

__all__ = [
    "EVAL_SCHEMA_VERSION",
    "GUARD_DECISION_SPAN",
    "guard_decision_attributes",
]

EVAL_SCHEMA_VERSION = "director.eval.v1"
GUARD_DECISION_SPAN = "director_ai.eval.guard_decision"

_GEN_AI_SYSTEM = "director-ai"


def guard_decision_attributes(
    *,
    decision: str,
    approved: bool,
    score: float,
    threshold: float,
    model: str = "",
    scorer: str = "",
    tenant_id: str = "",
    domain: str = "",
    evidence_count: int = 0,
    unsupported_claims: int = 0,
    answer_id: str = "",
) -> dict[str, str | int | float | bool]:
    """Build the tenant-safe span attributes for one guard decision.

    Parameters
    ----------
    decision:
        The guard outcome label (e.g. ``"allow"`` / ``"halt"``).
    approved:
        Whether the guard approved the answer.
    score, threshold:
        The coherence score and the threshold it was judged against.
    model, scorer:
        The model that produced the answer and the scorer that judged it.
    tenant_id, domain:
        Tenant and domain of the request.
    evidence_count:
        Number of evidence chunks the decision rested on.
    unsupported_claims:
        Number of claims the scorer could not support (from the Answer BOM).
    answer_id:
        The Answer BOM id, linking this span to the response manifest.
    """
    attributes: dict[str, str | int | float | bool] = {
        "director.eval.schema_version": EVAL_SCHEMA_VERSION,
        "director.eval.decision": decision,
        "director.eval.approved": approved,
        "director.eval.score": float(score),
        "director.eval.threshold": float(threshold),
        "director.eval.evidence_count": int(evidence_count),
        "director.eval.unsupported_claims": int(unsupported_claims),
        # gen_ai.* semantic conventions for OTLP-native eval tracers.
        "gen_ai.system": _GEN_AI_SYSTEM,
        "gen_ai.operation.name": "guard",
    }
    if model:
        attributes["gen_ai.request.model"] = model
        attributes["director.eval.model"] = model
    if scorer:
        attributes["director.eval.scorer"] = scorer
    if tenant_id:
        attributes["director.eval.tenant_id"] = tenant_id
    if domain:
        attributes["director.eval.domain"] = domain
    if answer_id:
        attributes["director.eval.answer_id"] = answer_id
    return attributes
