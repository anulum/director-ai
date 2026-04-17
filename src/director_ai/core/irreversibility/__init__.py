# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — irreversibility forecaster package

"""Irreversibility impact forecaster.

Estimate the probability that an action sequence crosses a
point-of-no-return — a state from which the safety invariant can
no longer be restored by any subsequent reversal.

* :class:`ReversibilityEstimator` — Protocol for anything that
  scores a single action's reversibility in ``[0, 1]`` (1.0 =
  trivially reversible, 0.0 = permanent). The shipped
  :class:`RuleReversibility` is a deterministic keyword-based
  estimator; model-backed and causal-graph-aware estimators
  drop in on the same Protocol.
* :class:`IrreversibilityForecaster` — seeded Monte-Carlo sampler
  that walks ``N`` draws of ``K``-step sequences, records the
  fraction that crosses the caller-supplied irreversibility
  threshold, and returns a Wilson-score credible interval —
  lighter than a full conformal prior and tight enough for
  go/no-go decisions.

Independent-action sampling is the default. A causal-graph-aware
estimator composed with :mod:`~director_ai.core.causal_verifier`
slots in through the :class:`ReversibilityEstimator` Protocol
without any API change here.
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
