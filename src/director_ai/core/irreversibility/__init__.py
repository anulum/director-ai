# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — irreversibility forecaster package

"""Irreversibility impact forecaster (roadmap Tier 1 Batch 4 #8).

Estimate the probability that an action sequence crosses a
point-of-no-return — a state from which the safety invariant can
no longer be restored by any subsequent reversal.

* :class:`ReversibilityEstimator` — Protocol for anything that
  scores a single action's reversibility in ``[0, 1]`` (1.0 =
  trivially reversible, 0.0 = permanent). The shipped
  :class:`RuleReversibility` is a keyword-based stub useful for
  tests and bootstrap; model-backed estimators drop in later.
* :class:`IrreversibilityForecaster` — seeded Monte-Carlo sampler
  that walks ``N`` draws of ``K``-step sequences, records the
  fraction that crosses the caller-supplied irreversibility
  threshold, and returns a conformal-style credible interval
  (Wilson score, not a heavy prior — enough for go/no-go).

Foundation scope: independent-action sampling (no causal graph,
no HaltMonitor coupling). The Protocol boundary for the
estimator is stable so a causal-graph-aware estimator composed
with :mod:`~director_ai.core.causal_verifier` slots in later.
"""

from .forecaster import Forecast, IrreversibilityForecaster
from .reversibility import (
    ReversibilityEstimator,
    ReversibilityScore,
    RuleReversibility,
)

__all__ = [
    "Forecast",
    "IrreversibilityForecaster",
    "ReversibilityEstimator",
    "ReversibilityScore",
    "RuleReversibility",
]
