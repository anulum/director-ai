# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — protobuf <-> internal type converters

"""Small, hand-written adapters between the generated protobuf
messages in :mod:`director_ai.proto.director.v1` and the native
Director-AI dataclasses.

Converters live here (not on the dataclasses) so the generated
protobuf package stays mechanical and the domain types stay free of
protobuf imports.
"""

from __future__ import annotations

from typing import Any, cast

from director_ai.proto.director.v1 import director_pb2 as pb

__all__ = [
    "halt_reason_from_string",
    "halt_reason_to_string",
    "verdict_to_proto",
    "verdict_from_proto",
]


_HALT_REASON_BY_STRING: dict[str, int] = {
    "": pb.HALT_REASON_NONE,
    "none": pb.HALT_REASON_NONE,
    "coherence": pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD,
    "coherence_below_threshold": pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD,
    "injection": pb.HALT_REASON_INJECTION_DETECTED,
    "policy": pb.HALT_REASON_POLICY_VIOLATION,
    "token_timeout": pb.HALT_REASON_TOKEN_TIMEOUT,
    "total_timeout": pb.HALT_REASON_TOTAL_TIMEOUT,
    "callback_timeout": pb.HALT_REASON_CALLBACK_TIMEOUT,
}

_STRING_BY_HALT_REASON: dict[int, str] = {
    pb.HALT_REASON_UNSPECIFIED: "unspecified",
    pb.HALT_REASON_NONE: "none",
    pb.HALT_REASON_COHERENCE_BELOW_THRESHOLD: "coherence_below_threshold",
    pb.HALT_REASON_INJECTION_DETECTED: "injection",
    pb.HALT_REASON_POLICY_VIOLATION: "policy",
    pb.HALT_REASON_TOKEN_TIMEOUT: "token_timeout",
    pb.HALT_REASON_TOTAL_TIMEOUT: "total_timeout",
    pb.HALT_REASON_CALLBACK_TIMEOUT: "callback_timeout",
}


def halt_reason_from_string(value: str | None) -> int:
    """Map a Python halt-reason string to the generated enum.

    Unknown values map to ``HALT_REASON_UNSPECIFIED`` so decoders
    keep working across schema revisions.
    """
    if value is None:
        return pb.HALT_REASON_NONE
    key = value.strip().lower()
    return _HALT_REASON_BY_STRING.get(key, pb.HALT_REASON_UNSPECIFIED)


def halt_reason_to_string(value: int) -> str:
    """Inverse of :func:`halt_reason_from_string`."""
    return _STRING_BY_HALT_REASON.get(value, "unspecified")


def verdict_to_proto(
    *,
    score: float,
    halted: bool,
    halt_reason: str | None = None,
    hard_limit: float = 0.0,
    score_lower: float = 0.0,
    score_upper: float = 0.0,
    sources: list[dict[str, Any]] | None = None,
    message: str = "",
) -> pb.CoherenceVerdict:
    """Build a :class:`CoherenceVerdict` protobuf from loose fields.

    Scorer backends in the Python code-base all return different
    shapes; rather than introduce a single internal type the
    converter pulls out a minimal common denominator. `sources` may
    be ``None`` or a list of ``{"source_id", "similarity",
    "nli_support"}`` dicts.
    """
    verdict = pb.CoherenceVerdict(
        score=float(score),
        halted=bool(halted),
        halt_reason=cast("pb.HaltReason", halt_reason_from_string(halt_reason)),
        hard_limit=float(hard_limit),
        score_lower=float(score_lower),
        score_upper=float(score_upper),
        message=message,
    )
    if sources:
        for src in sources:
            verdict.sources.add(
                source_id=str(src.get("source_id", "")),
                similarity=float(src.get("similarity", 0.0)),
                nli_support=float(src.get("nli_support", 0.0)),
            )
    return verdict


def verdict_from_proto(verdict: pb.CoherenceVerdict) -> dict[str, Any]:
    """Reverse of :func:`verdict_to_proto` producing a plain dict."""
    return {
        "score": verdict.score,
        "halted": verdict.halted,
        "halt_reason": halt_reason_to_string(verdict.halt_reason),
        "hard_limit": verdict.hard_limit,
        "score_lower": verdict.score_lower,
        "score_upper": verdict.score_upper,
        "sources": [
            {
                "source_id": s.source_id,
                "similarity": s.similarity,
                "nli_support": s.nli_support,
            }
            for s in verdict.sources
        ],
        "message": verdict.message,
    }
