# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Director-Lite heuristic coherence (vendored)
"""Heuristic-coherence combination, vendored verbatim from the full package.

Director-Lite is a standalone distribution with no ``director-ai`` dependency,
so the calibration used by the full product's no-model coherence path is copied
here byte-for-byte (``director_ai.core.scoring.heuristic_coherence``) to keep the
lite and full halt decisions identical on the same inputs. The full package
remains the single source of truth; this copy must track it.
"""

from __future__ import annotations

DIVERGENCE_NEUTRAL: float = 0.5


def combine_weighted_coherence(
    *,
    h_logic: float,
    h_factual: float,
    w_logic: float,
    w_fact: float,
    nli_available: bool,
    evidence_present: bool,
    dialogue_route: bool,
) -> float:
    """Combine component divergences into a calibrated coherence score in [0, 1].

    Mirrors ``director_ai.core.scoring.heuristic_coherence`` so a lite halt and a
    full-package heuristic halt agree on identical component scores.
    """
    total_divergence = w_logic * h_logic + w_fact * h_factual
    coherence = 1.0 - total_divergence

    fact_is_neutral = abs(h_factual - DIVERGENCE_NEUTRAL) < 1e-9
    if (
        nli_available
        and fact_is_neutral
        and not evidence_present
        and not dialogue_route
    ):
        lo = 1.0 - w_logic - w_fact * DIVERGENCE_NEUTRAL
        hi = 1.0 - w_fact * DIVERGENCE_NEUTRAL
        span = hi - lo
        if span > 1e-9:
            coherence = (coherence - lo) / span

    return max(0.0, min(1.0, coherence))


__all__ = ["DIVERGENCE_NEUTRAL", "combine_weighted_coherence"]
