# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Shared Types (Coherence Engine)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import math
from dataclasses import dataclass


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi], converting NaN/Inf to boundary."""
    if math.isnan(value):
        return lo
    if math.isinf(value):
        return hi if value > 0 else lo
    return max(lo, min(hi, value))


@dataclass
class CoherenceScore:
    """Result of a coherence check on generated output."""

    score: float  # Composite coherence score (0.0 = incoherent, 1.0 = perfect)
    approved: bool  # Whether the output passes the threshold
    h_logical: float  # Logical divergence (NLI contradiction probability)
    h_factual: float  # Factual divergence (ground truth deviation)

    def __post_init__(self) -> None:
        self.score = _clamp(self.score)
        self.h_logical = _clamp(self.h_logical)
        self.h_factual = _clamp(self.h_factual)


@dataclass
class ReviewResult:
    """Full review outcome from the CoherenceAgent pipeline."""

    output: str  # Final output text (or halt message)
    coherence: CoherenceScore | None  # Score of the selected candidate
    halted: bool  # True if the system refused to emit output
    candidates_evaluated: int  # Number of candidates scored

    def __post_init__(self) -> None:
        if self.candidates_evaluated < 0:
            self.candidates_evaluated = 0
