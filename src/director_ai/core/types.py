# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Shared Types (Coherence Engine)

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

__all__ = [
    "ClaimAttribution",
    "CoherenceScore",
    "EvidenceChunk",
    "HaltEvidence",
    "ReviewResult",
    "ScoringEvidence",
]

_clamp_logger = logging.getLogger("DirectorAI.Types")


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi], replacing NaN/Inf with boundary values."""
    if math.isnan(value):
        _clamp_logger.warning("NaN detected in _clamp — replacing with %s", lo)
        return lo
    if math.isinf(value):
        replacement = hi if value > 0 else lo
        _clamp_logger.warning("Inf detected in _clamp — replacing with %s", replacement)
        return replacement
    return max(lo, min(hi, value))


@dataclass
class ClaimAttribution:
    """Maps a summary claim to the source sentence that best supports/contradicts it."""

    claim: str
    claim_index: int
    source_sentence: str
    source_index: int
    divergence: float
    supported: bool


@dataclass
class EvidenceChunk:
    """A single RAG retrieval result with relevance distance."""

    text: str
    distance: float  # lower = more relevant
    source: str = ""


@dataclass
class ScoringEvidence:
    """Evidence collected during coherence scoring."""

    chunks: list[EvidenceChunk]
    nli_premise: str
    nli_hypothesis: str
    nli_score: float
    chunk_scores: list[float] | None = None
    premise_chunk_count: int = 1
    hypothesis_chunk_count: int = 1
    claim_coverage: float | None = None
    per_claim_divergences: list[float] | None = None
    claims: list[str] | None = None
    attributions: list[ClaimAttribution] | None = None
    token_count: int | None = None
    estimated_cost_usd: float | None = None


@dataclass
class CoherenceScore:
    """Result of a coherence check on generated output."""

    score: float  # Composite coherence score (0.0 = incoherent, 1.0 = perfect)
    approved: bool  # Whether the output passes the threshold
    h_logical: float  # Logical divergence (NLI contradiction probability)
    h_factual: float  # Factual divergence (ground truth deviation)
    evidence: ScoringEvidence | None = None
    warning: bool = False
    cross_turn_divergence: float | None = None
    strict_mode_rejected: bool = False
    verdict_confidence: float | None = (
        None  # 0-1, guardrail confidence in its own verdict
    )
    nli_model_confidence: float | None = (
        None  # 0-1, NLI softmax entropy-based confidence
    )
    signal_agreement: float | None = (
        None  # 0-1, agreement between h_logical and h_factual
    )
    contradiction_index: float | None = (
        None  # 0-1, cross-turn self-contradiction severity
    )
    # Phase 5 — Multi-Signal Explainability
    detected_task_type: str | None = None  # dialogue/summarization/qa/rag/default
    escalated_to_judge: bool | None = None  # whether LLM judge was consulted
    nli_probs: dict[str, float] | None = None  # {entailment, neutral, contradiction}
    retrieval_confidence: float | None = (
        None  # best retrieval distance (0=no match, 1=exact)
    )

    # -- Claim-Level Provenance (Gem 2) ------------------------------------

    @property
    def claims(self) -> list[str]:
        """Atomic claims extracted from the scored response."""
        if self.evidence is not None and self.evidence.claims:
            return self.evidence.claims
        return []

    @property
    def attributions(self) -> list[ClaimAttribution]:
        """Per-claim source attribution with support/divergence."""
        if self.evidence is not None and self.evidence.attributions:
            return self.evidence.attributions
        return []

    @property
    def claim_coverage(self) -> float | None:
        """Fraction of claims supported by source material (0-1)."""
        if self.evidence is not None:
            return self.evidence.claim_coverage
        return None

    @property
    def unsupported_claims(self) -> list[ClaimAttribution]:
        """Claims not supported by any source — the hallucinated ones."""
        return [a for a in self.attributions if not a.supported]

    def claim_provenance(self) -> list[dict]:
        """Structured provenance for each claim.

        Returns a list of dicts, one per claim::

            [
                {
                    "claim": "Paris is the capital of France.",
                    "supported": True,
                    "source": "France is a country whose capital is Paris.",
                    "divergence": 0.12,
                    "source_index": 3,
                },
                ...
            ]
        """
        return [
            {
                "claim": a.claim,
                "supported": a.supported,
                "source": a.source_sentence,
                "divergence": a.divergence,
                "source_index": a.source_index,
            }
            for a in self.attributions
        ]


@dataclass
class HaltEvidence:
    """Structured evidence returned when the agent halts."""

    reason: str
    last_score: float
    evidence_chunks: list[EvidenceChunk]
    nli_scores: list[float] | None = None
    suggested_action: str = ""


@dataclass
class ReviewResult:
    """Full review outcome from the CoherenceAgent pipeline."""

    output: str  # Final output text (or halt message)
    coherence: CoherenceScore | None  # Score of the selected candidate
    halted: bool  # True if the system refused to emit output
    candidates_evaluated: int  # Number of candidates scored
    fallback_used: bool = False
    halt_evidence: HaltEvidence | None = None
