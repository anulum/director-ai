# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Answer BOM builder

"""Build an :class:`AnswerBOM` from a :class:`CoherenceScore`.

The scorer already produces claim-level provenance — the atomic claims and, for
each, the source sentence it was attributed to with a support/contradiction
divergence. This module folds that into the versioned manifest, mapping each
attribution to a :class:`ClaimRecord` with a verdict and a support strength, and
resolving the attributed source to an evidence identifier when the scoring
evidence carries one. The manifest's claim-level richness is exactly that of the
scorer's provenance: when no attributions are present, the manifest records the
model/scorer/threshold header with no per-claim rows rather than inventing them.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import ClaimAttribution, CoherenceScore, EvidenceChunk
from .manifest import AnswerBOM, ClaimRecord, new_answer_id

__all__ = ["build_answer_bom"]


def build_answer_bom(
    score: CoherenceScore,
    *,
    model: str,
    scorer: str,
    threshold: float,
    tenant: str = "",
    answer_id: str | None = None,
    timestamp: str | None = None,
    freshness: str = "",
    policy_refs: Sequence[str] = (),
    contradiction_threshold: float = 0.5,
) -> AnswerBOM:
    """Build the per-response manifest from a scorer result.

    Parameters
    ----------
    score:
        The :class:`CoherenceScore` returned by the scorer's ``review``.
    model, scorer:
        Identifiers recorded in the manifest header.
    threshold:
        The approval threshold the scorer applied.
    tenant:
        The tenant the answer was produced for; copied onto every claim.
    answer_id, timestamp:
        Optional overrides; an id and UTC timestamp are generated when omitted.
    freshness:
        Freshness marker applied to every supported claim's evidence.
    policy_refs:
        Policy identifiers applied to every claim.
    contradiction_threshold:
        Divergence at or above which an unsupported claim is recorded as
        ``contradicted`` rather than merely ``unsupported``.
    """
    chunks = _evidence_chunks(score)
    policy = tuple(policy_refs)
    claims = tuple(
        _record(
            attribution,
            chunks=chunks,
            tenant=tenant,
            freshness=freshness,
            policy_refs=policy,
            contradiction_threshold=contradiction_threshold,
        )
        for attribution in score.attributions
    )
    return AnswerBOM(
        answer_id=answer_id or new_answer_id(),
        model=model,
        scorer=scorer,
        threshold=threshold,
        tenant=tenant,
        claims=claims,
        timestamp=timestamp or "",
    )


def _evidence_chunks(score: CoherenceScore) -> list[EvidenceChunk]:
    if score.evidence is not None and score.evidence.chunks:
        return list(score.evidence.chunks)
    return []


def _record(
    attribution: ClaimAttribution,
    *,
    chunks: list[EvidenceChunk],
    tenant: str,
    freshness: str,
    policy_refs: tuple[str, ...],
    contradiction_threshold: float,
) -> ClaimRecord:
    support = max(0.0, min(1.0, 1.0 - attribution.divergence))
    evidence_ids: tuple[str, ...]
    if attribution.supported:
        verdict = "supported"
        evidence_ids = (_evidence_id(attribution, chunks),)
    elif attribution.divergence >= contradiction_threshold:
        verdict = "contradicted"
        evidence_ids = ()
    else:
        verdict = "unsupported"
        evidence_ids = ()
    return ClaimRecord(
        claim=attribution.claim,
        verdict=verdict,
        support=round(support, 4),
        evidence_ids=evidence_ids,
        freshness=freshness if attribution.supported else "",
        tenant=tenant,
        policy_refs=policy_refs,
    )


def _evidence_id(
    attribution: ClaimAttribution,
    chunks: list[EvidenceChunk],
) -> str:
    """Resolve the attributed source to an evidence id.

    Uses the source chunk's ``source`` field when the attribution's source index
    addresses a known evidence chunk; otherwise falls back to a positional
    reference so the link is never silently dropped.
    """
    index = attribution.source_index
    if 0 <= index < len(chunks) and chunks[index].source:
        return chunks[index].source
    return f"source:{index}"
