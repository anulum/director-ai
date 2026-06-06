# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence packet

"""Run the narrow grounding demo and emit a verifiable evidence packet.

The packet is the one-command artefact a buyer runs to see the whole loop work:
load a small policy knowledge base, approve a grounded answer, block a
hallucinated one, and emit the decision evidence. It is self-contained and
tamper-evident — the manifest carries a SHA-256 digest over its canonical
content, and :func:`verify_evidence_packet` recomputes it — so the packet can be
shipped to a reviewer and checked without re-running the guard.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "EVIDENCE_PACKET_VERSION",
    "build_evidence_packet",
    "verify_evidence_packet",
]

EVIDENCE_PACKET_VERSION = "director.evidence_packet.v1"

# A small, self-contained policy knowledge base for the demo.
DEMO_FACTS: dict[str, str] = {
    "refund_window": "Refunds are available within 30 days of purchase.",
    "refund_method": "Refunds are issued to the original payment method.",
    "warranty": "Hardware carries a 24-month limited warranty.",
    "support_hours": "Support operates 09:00-17:00 CET on business days.",
    "data_region": "Customer data is stored in the EU (Frankfurt) region.",
}

DEMO_QUESTION = "What is the refund window?"
DEMO_GROUNDED_ANSWER = "Refunds are available within 30 days of purchase."
DEMO_HALLUCINATED_ANSWER = "Refunds are available any time, with a cash bonus."


def _canonical_digest(payload: dict[str, Any]) -> str:
    """Return a stable SHA-256 hex digest over a payload (sorted keys)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_evidence_packet(
    guard: Any,
    *,
    facts: dict[str, str] | None = None,
    question: str = DEMO_QUESTION,
    grounded_answer: str = DEMO_GROUNDED_ANSWER,
    hallucinated_answer: str = DEMO_HALLUCINATED_ANSWER,
    model: str = "demo",
    tenant_id: str = "demo",
) -> dict[str, Any]:
    """Run the narrow demo on ``guard`` and return a verifiable packet.

    Steps: load the policy facts; score a grounded answer (expected approved);
    score a hallucinated answer (expected blocked); attach the Answer BOM and the
    eval-trace record for each; stamp a content digest. The guard is any object
    exposing ``load_facts``, ``check``, ``answer_bom`` and ``eval_trace`` (i.e.
    :class:`~director_ai.guard.ProductionGuard`).
    """
    kb = dict(facts) if facts is not None else dict(DEMO_FACTS)
    guard.load_facts(kb)

    grounded = guard.check(question, grounded_answer)
    hallucinated = guard.check(question, hallucinated_answer)

    def _decision(result: Any, answer: str) -> dict[str, Any]:
        bom = guard.answer_bom(result, model=model, tenant=tenant_id)
        trace = guard.eval_trace(
            result, model=model, tenant_id=tenant_id, answer_id=bom.answer_id
        )
        return {
            "approved": bool(result.approved),
            "score": float(result.score),
            "answer_bom": bom.to_dict(),
            "eval_trace": trace,
        }

    content = {
        "schema_version": EVIDENCE_PACKET_VERSION,
        "knowledge_base_size": len(kb),
        "question": question,
        "grounded": _decision(grounded, grounded_answer),
        "hallucinated": _decision(hallucinated, hallucinated_answer),
        "checks": {
            "grounded_approved": bool(grounded.approved),
            "hallucinated_blocked": not bool(hallucinated.approved),
        },
    }
    return {
        "content": content,
        "integrity": {
            "algorithm": "sha256",
            "digest": _canonical_digest(content),
        },
    }


def verify_evidence_packet(packet: dict[str, Any]) -> tuple[bool, str]:
    """Verify a packet's integrity and demo expectations.

    Returns ``(ok, reason)``. ``ok`` is ``True`` only when the recomputed digest
    matches, the schema version is recognised, the grounded answer was approved,
    and the hallucinated answer was blocked.
    """
    if not isinstance(packet, dict) or "content" not in packet:
        return False, "malformed_packet"
    content = packet["content"]
    if content.get("schema_version") != EVIDENCE_PACKET_VERSION:
        return False, "unsupported_schema_version"
    integrity = packet.get("integrity", {})
    recomputed = _canonical_digest(content)
    if integrity.get("digest") != recomputed:
        return False, "digest_mismatch"
    checks = content.get("checks", {})
    if not checks.get("grounded_approved"):
        return False, "grounded_not_approved"
    if not checks.get("hallucinated_blocked"):
        return False, "hallucinated_not_blocked"
    return True, "ok"
