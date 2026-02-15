# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Shared Types (Coherence Engine)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class CoherenceScore:
    """Result of a coherence check on generated output."""

    score: float           # Composite coherence score (0.0 = incoherent, 1.0 = perfect)
    approved: bool         # Whether the output passes the threshold
    h_logical: float       # Logical divergence (NLI contradiction probability)
    h_factual: float       # Factual divergence (ground truth deviation)


@dataclass
class ReviewResult:
    """Full review outcome from the CoherenceAgent pipeline."""

    output: str            # Final output text (or halt message)
    coherence: CoherenceScore | None  # Score of the selected candidate
    halted: bool           # True if the system refused to emit output
    candidates_evaluated: int  # Number of candidates scored
