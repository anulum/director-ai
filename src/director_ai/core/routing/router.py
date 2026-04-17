# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RiskRouter

"""Policy glue that turns a :class:`RiskComponents` + per-tenant
budget into a routing decision the gateway can act on.

A :class:`RoutingDecision` carries three things:

* ``backend`` — which scorer backend to load (``"rules"`` /
  ``"embed"`` / ``"nli"``).
* ``action`` — ``"allow"`` when the request proceeds, ``"reject"``
  when the tenant has blown its risk budget. The gateway may
  choose to return 429 or 422 depending on deployment policy.
* ``risk`` / ``budget`` — the underlying :class:`RiskComponents`
  and :class:`BudgetEntry` for logging and observability.

The router is configuration-only — no state beyond the injected
scorer and budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .budget import BudgetEntry, RiskBudget
from .scorer import PromptRiskScorer, RiskComponents

Action = Literal["allow", "reject"]
Backend = Literal["rules", "embed", "nli"]


@dataclass(frozen=True)
class RoutingDecision:
    """Result of a routing evaluation.

    Consumers branch on ``action`` first; ``backend`` is only
    meaningful when ``action == "allow"``.
    """

    backend: Backend
    action: Action
    reason: str
    risk: RiskComponents
    budget: BudgetEntry


class RiskRouter:
    """Decide which scoring backend to run and whether to honour the
    request at all.

    Parameters
    ----------
    scorer :
        The :class:`PromptRiskScorer` instance used to score every
        incoming prompt.
    budget :
        :class:`RiskBudget` ledger; per-tenant bookkeeping.
    rules_threshold :
        Prompts whose risk stays below this go to the cheap rules
        backend.
    embed_threshold :
        Between ``rules_threshold`` and ``embed_threshold`` the
        router picks the embedding backend; above it we escalate to
        NLI.
    reject_threshold :
        Prompts above this are rejected even if the budget has
        headroom — used to block obvious attacks without spending
        budget on them.
    """

    def __init__(
        self,
        *,
        scorer: PromptRiskScorer,
        budget: RiskBudget,
        rules_threshold: float = 0.2,
        embed_threshold: float = 0.55,
        reject_threshold: float = 0.92,
    ) -> None:
        if not 0.0 < rules_threshold < embed_threshold < reject_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < rules < embed < reject <= 1; got "
                f"({rules_threshold}, {embed_threshold}, {reject_threshold})"
            )
        self._scorer = scorer
        self._budget = budget
        self._rules_threshold = rules_threshold
        self._embed_threshold = embed_threshold
        self._reject_threshold = reject_threshold

    def route(self, prompt: str, tenant_id: str = "") -> RoutingDecision:
        """Score ``prompt`` for ``tenant_id`` and return the
        :class:`RoutingDecision`. Passes ``""`` as tenant when the
        caller has no binding (single-tenant deployments).
        """
        risk = self._scorer.score(prompt)
        if risk.combined >= self._reject_threshold:
            snapshot = self._budget.snapshot(tenant_id)
            return RoutingDecision(
                backend="nli",
                action="reject",
                reason=f"risk={risk.combined:.3f} >= reject_threshold "
                f"({self._reject_threshold:.3f})",
                risk=risk,
                budget=snapshot,
            )
        budget_entry = self._budget.reserve(tenant_id, risk.combined)
        if budget_entry.exhausted:
            return RoutingDecision(
                backend=self._select_backend(risk.combined),
                action="reject",
                reason=(
                    f"risk budget exhausted "
                    f"(consumed={budget_entry.consumed:.2f} / "
                    f"allowance={budget_entry.allowance:.2f})"
                ),
                risk=risk,
                budget=budget_entry,
            )
        backend = self._select_backend(risk.combined)
        return RoutingDecision(
            backend=backend,
            action="allow",
            reason=self._route_reason(risk.combined, backend),
            risk=risk,
            budget=budget_entry,
        )

    def _select_backend(self, risk: float) -> Backend:
        if risk < self._rules_threshold:
            return "rules"
        if risk < self._embed_threshold:
            return "embed"
        return "nli"

    def _route_reason(self, risk: float, backend: Backend) -> str:
        if backend == "rules":
            return f"risk={risk:.3f} < rules_threshold ({self._rules_threshold:.3f})"
        if backend == "embed":
            return (
                f"{self._rules_threshold:.3f} <= risk={risk:.3f} "
                f"< embed_threshold ({self._embed_threshold:.3f})"
            )
        return (
            f"{self._embed_threshold:.3f} <= risk={risk:.3f} "
            f"< reject_threshold ({self._reject_threshold:.3f})"
        )
