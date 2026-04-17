# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — prompt-risk routing package

"""Predictive prompt-risk routing (roadmap 2026-2030, Tier 1 #2).

Ultra-fast input risk scoring so the gateway can pick the right
scoring backend and honour a per-tenant risk budget before the
expensive NLI model ever loads:

* :class:`PromptRiskScorer` — combines a length/complexity
  heuristic with :class:`~director_ai.core.safety.sanitizer.InputSanitizer`
  and optional :class:`~director_ai.core.safety.injection.InjectionDetector`
  signals into a single ``[0, 1]`` risk score.
* :class:`RiskBudget` — sliding-window per-tenant ledger that
  throttles tenants whose cumulative risk exceeds their allowance.
* :class:`RiskRouter` — picks a scorer backend name (``"rules"`` /
  ``"embed"`` / ``"nli"``) based on the risk score and returns a
  routing decision that includes budget status.

The package is deliberately self-contained — it imports the
sanitiser / injection detector lazily so callers that do not need
those signals can use :class:`PromptRiskScorer` with just the
length heuristic.
"""

from .budget import BudgetEntry, RiskBudget
from .router import RiskRouter, RoutingDecision
from .scorer import PromptRiskScorer, RiskComponents

__all__ = [
    "BudgetEntry",
    "PromptRiskScorer",
    "RiskBudget",
    "RiskComponents",
    "RiskRouter",
    "RoutingDecision",
]
