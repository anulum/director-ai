# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — hallucination economics (cost-risk action selection)

"""Pick the guard action that minimises expected cost for one request.

Guarding is usually framed as a quality control; this frames it as an
*economic* decision. Every guard action has an operational cost (compute, added
latency, dollars) and a catch rate — the probability it stops a hallucination
that would otherwise reach the user. A hallucination that does reach the user
has a business cost (a wrong medical answer costs more than a wrong film
recommendation). For a request whose hallucination risk is ``risk`` the expected
total cost of an action is::

    expected_cost(action) = action.cost + risk · (1 - action.catch) · hallucination_cost

The cheapest action under that rule is the economically optimal one: a low-risk
request in a low-stakes domain should skip the expensive scorer, while a
high-risk request in a high-stakes domain should pay for escalation or human
review. The decision also reports the *value* of guarding — the expected loss it
avoids versus doing nothing — so guarding can be justified as a value driver, not
only a cost centre.

The computation is exact decision arithmetic over a small action menu; there is
no array hot-loop and therefore no polyglot kernel, matching the deterministic
verification modules in this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "GuardAction",
    "EconomicDecision",
    "HallucinationEconomics",
    "DEFAULT_ACTIONS",
]


@dataclass(frozen=True)
class GuardAction:
    """One guarding option with its operational cost and catch rate.

    Parameters
    ----------
    name:
        Identifier for the action (e.g. ``"nli"``, ``"escalate"``).
    cost:
        Operational cost of taking the action, in the same abstract unit as
        ``hallucination_cost`` (compute, latency budget, or currency — the caller
        decides, as long as both sides use one unit).
    catch:
        Probability in ``[0, 1]`` that the action stops a hallucination that
        would otherwise reach the user (its recall). ``0`` means it catches
        nothing (a no-op / skip); ``1`` means it catches everything.
    """

    name: str
    cost: float
    catch: float

    def __post_init__(self) -> None:
        """Reject a negative cost or a catch rate outside [0, 1]."""
        if self.cost < 0.0:
            raise ValueError(f"{self.name}: cost must be non-negative")
        if not 0.0 <= self.catch <= 1.0:
            raise ValueError(f"{self.name}: catch must be in [0, 1]")


# Illustrative default tiers; costs are abstract units (configure per workload).
DEFAULT_ACTIONS: tuple[GuardAction, ...] = (
    GuardAction("skip", cost=0.0, catch=0.0),
    GuardAction("heuristic", cost=0.01, catch=0.55),
    GuardAction("nli", cost=0.2, catch=0.9),
    GuardAction("escalate", cost=1.0, catch=0.97),
    GuardAction("human_review", cost=5.0, catch=0.99),
)


@dataclass(frozen=True)
class EconomicDecision:
    """The cost-minimising guard action for one request."""

    action: str
    risk: float
    expected_cost: float
    baseline_cost: float
    value: float
    worth_guarding: bool
    residual_risk: float
    breakdown: tuple[tuple[str, float], ...]
    rationale: tuple[str, ...]


@dataclass
class HallucinationEconomics:
    """Select the expected-cost-minimising guard action per request.

    Parameters
    ----------
    actions:
        The menu of available :class:`GuardAction` options. Defaults to
        :data:`DEFAULT_ACTIONS`. A zero-cost, zero-catch ``skip`` baseline is
        always considered even if it is not in the menu, so the *value* of
        guarding is always well defined.
    hallucination_cost:
        Default business cost of a hallucination reaching the user, in the same
        unit as the action costs. Override per call (e.g. per domain or tenant).
    """

    actions: Sequence[GuardAction] = field(default_factory=lambda: DEFAULT_ACTIONS)
    hallucination_cost: float = 1.0

    def __post_init__(self) -> None:
        """Reject an empty action menu or a negative hallucination cost."""
        if not self.actions:
            raise ValueError("at least one guard action is required")
        if self.hallucination_cost < 0.0:
            raise ValueError("hallucination_cost must be non-negative")

    @staticmethod
    def expected_cost(
        action: GuardAction, risk: float, hallucination_cost: float
    ) -> float:
        """Compute the expected total cost of *action* at the given *risk*."""
        return action.cost + risk * (1.0 - action.catch) * hallucination_cost

    def decide(
        self, risk: float, *, hallucination_cost: float | None = None
    ) -> EconomicDecision:
        """Return the cost-minimising :class:`EconomicDecision` for *risk*."""
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        hcost = (
            self.hallucination_cost
            if hallucination_cost is None
            else hallucination_cost
        )
        if hcost < 0.0:
            raise ValueError("hallucination_cost must be non-negative")

        # Doing nothing: no operational cost, no catch.
        baseline_cost = risk * hcost

        breakdown = tuple(
            (a.name, round(self.expected_cost(a, risk, hcost), 6)) for a in self.actions
        )
        best = min(self.actions, key=lambda a: self.expected_cost(a, risk, hcost))
        best_cost = self.expected_cost(best, risk, hcost)
        residual_risk = risk * (1.0 - best.catch)
        value = baseline_cost - best_cost
        worth_guarding = value > 0.0 and best.catch > 0.0

        rationale: list[str] = []
        if not worth_guarding:
            rationale.append("guarding not worth its cost for this request")
        elif best.catch >= 0.95:
            rationale.append("high-stakes/high-risk: pay for strong catch")
        else:
            rationale.append("cheap guard pays for itself")
        if risk >= 0.5:
            rationale.append("elevated hallucination risk")

        return EconomicDecision(
            action=best.name,
            risk=round(risk, 4),
            expected_cost=round(best_cost, 6),
            baseline_cost=round(baseline_cost, 6),
            value=round(value, 6),
            worth_guarding=worth_guarding,
            residual_risk=round(residual_risk, 6),
            breakdown=breakdown,
            rationale=tuple(rationale),
        )
