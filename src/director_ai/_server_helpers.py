# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Server Helper Functions
"""Serialisation helpers for the Director-AI REST server."""

from __future__ import annotations


def halt_evidence_to_dict(halt_ev) -> dict | None:
    """Convert HaltEvidence to JSON-safe dict."""
    if halt_ev is None:
        return None
    return {
        "reason": halt_ev.reason,
        "last_score": halt_ev.last_score,
        "evidence_chunks": [
            {"text": c.text, "distance": c.distance, "source": c.source}
            for c in halt_ev.evidence_chunks
        ],
        "nli_scores": halt_ev.nli_scores,
        "suggested_action": halt_ev.suggested_action,
    }


def evidence_to_dict(evidence) -> dict | None:
    """Convert ScoringEvidence to JSON-safe dict."""
    if evidence is None:
        return None
    d = {
        "chunks": [
            {"text": c.text, "distance": c.distance, "source": c.source}
            for c in evidence.chunks
        ],
        "nli_premise": evidence.nli_premise,
        "nli_hypothesis": evidence.nli_hypothesis,
        "nli_score": evidence.nli_score,
        "premise_chunk_count": evidence.premise_chunk_count,
        "hypothesis_chunk_count": evidence.hypothesis_chunk_count,
    }
    if evidence.claim_coverage is not None:
        d["claim_coverage"] = evidence.claim_coverage
        d["per_claim_divergences"] = evidence.per_claim_divergences
        d["claims"] = evidence.claims
    if evidence.attributions is not None:
        d["attributions"] = [
            {
                "claim": a.claim,
                "claim_index": a.claim_index,
                "source_sentence": a.source_sentence,
                "source_index": a.source_index,
                "divergence": a.divergence,
                "supported": a.supported,
            }
            for a in evidence.attributions
        ]
    if evidence.token_count is not None:
        d["token_count"] = evidence.token_count
        d["estimated_cost_usd"] = evidence.estimated_cost_usd
    return d
