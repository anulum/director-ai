# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Shared Types (Coherence Engine)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

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


@dataclass
class CoherenceScore:
    """Result of a coherence check on generated output."""

    score: float  # Composite coherence score (0.0 = incoherent, 1.0 = perfect)
    approved: bool  # Whether the output passes the threshold
    h_logical: float  # Logical divergence (NLI contradiction probability)
    h_factual: float  # Factual divergence (ground truth deviation)
    evidence: ScoringEvidence | None = None
    warning: bool = False


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
