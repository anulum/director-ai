# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Meta-confidence estimation for guardrail verdicts.

Computes how confident the guardrail is in its own approval/rejection
decision by combining three orthogonal signals:

1. **NLI model confidence** — softmax entropy of the NLI prediction.
   High entropy = model is uncertain about the entailment label.
2. **Margin** — distance between the coherence score and the threshold.
   Score 0.51 at threshold 0.50 = margin 0.01 = low verdict confidence.
3. **Signal agreement** — agreement between logical and factual divergence.
   When one says "hallucination" and the other says "fine", the verdict
   is less trustworthy.

The combined verdict_confidence lets users route low-confidence results
to human review instead of trusting the binary approved/rejected signal.
"""

from __future__ import annotations

__all__ = ["compute_meta_confidence"]


def _margin_confidence(score: float, threshold: float) -> float:
    """Map score-threshold distance to a confidence in [0, 1].

    Linear ramp: margin of 0.0 → confidence 0.0,
    margin >= 0.20 → confidence 1.0.
    """
    margin = abs(score - threshold)
    return min(margin / 0.20, 1.0)


def _signal_agreement(h_logical: float, h_factual: float) -> float:
    """Measure agreement between logical and factual divergence signals.

    Both divergences are in [0, 1] where 0 = no divergence, 1 = full divergence.
    Perfect agreement (both 0 or both 1) → 1.0.
    Maximum disagreement (one 0, other 1) → 0.0.
    """
    return 1.0 - abs(h_logical - h_factual)


def compute_meta_confidence(
    score: float,
    threshold: float,
    h_logical: float,
    h_factual: float,
    nli_confidence: float | None = None,
) -> tuple[float, float, float]:
    """Compute verdict meta-confidence from available signals.

    Returns (verdict_confidence, margin_conf, signal_agr).
    If nli_confidence is provided, it's included in the combination.
    """
    margin_conf = _margin_confidence(score, threshold)
    signal_agr = _signal_agreement(h_logical, h_factual)

    if nli_confidence is not None:
        verdict = min(nli_confidence, margin_conf, signal_agr)
    else:
        verdict = min(margin_conf, signal_agr)

    return verdict, margin_conf, signal_agr
