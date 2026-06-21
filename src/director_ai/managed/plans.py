# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Plans and quota decisions for the managed Director-AI service.

A plan is just two limits — a monthly request cap and a per-minute rate — plus a
name. The registry turns an account's plan and its usage-so-far into an
allow/deny :class:`QuotaDecision` the request gate can act on and surface in
response headers. The numeric caps here are PROVISIONAL: the final tiers and
prices are a pending CEO decision (see the D1 build notes), so they are kept in
one overridable registry rather than scattered through the gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Plan:
    """A managed tier: a monthly request cap and a per-minute rate limit.

    ``None`` means "no limit" for that dimension (the ``scale`` tier).
    """

    name: str
    monthly_request_cap: int | None
    requests_per_minute: int | None


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    """The verdict on whether one more request is within an account's plan."""

    allowed: bool
    reason: str
    limit: int | None
    used: int
    remaining: int | None


# Provisional caps — final tiers/prices are a pending CEO decision (TODO D1).
_DEFAULT_PLANS: dict[str, Plan] = {
    "free": Plan("free", monthly_request_cap=1_000, requests_per_minute=20),
    "pro": Plan("pro", monthly_request_cap=100_000, requests_per_minute=120),
    "scale": Plan("scale", monthly_request_cap=None, requests_per_minute=None),
}

_FALLBACK_PLAN = _DEFAULT_PLANS["free"]


class PlanRegistry:
    """Resolves plan names to limits and decides quota for a usage count."""

    def __init__(self, plans: dict[str, Plan] | None = None) -> None:
        self._plans = dict(plans) if plans is not None else dict(_DEFAULT_PLANS)

    def names(self) -> list[str]:
        """Return the registered plan names."""
        return list(self._plans)

    def get(self, plan_name: str) -> Plan:
        """Return the named plan, falling back to the free tier when unknown.

        An account on a retired or mistyped plan degrades to the most
        restrictive tier rather than to unlimited access.
        """
        return self._plans.get(plan_name, _FALLBACK_PLAN)

    def decide(self, plan_name: str, used_this_period: int) -> QuotaDecision:
        """Decide whether one more request fits the plan's monthly cap."""
        plan = self.get(plan_name)
        cap = plan.monthly_request_cap
        if cap is None:
            return QuotaDecision(
                allowed=True,
                reason="unlimited plan",
                limit=None,
                used=used_this_period,
                remaining=None,
            )
        if used_this_period >= cap:
            return QuotaDecision(
                allowed=False,
                reason="monthly request cap reached",
                limit=cap,
                used=used_this_period,
                remaining=0,
            )
        return QuotaDecision(
            allowed=True,
            reason="within plan",
            limit=cap,
            used=used_this_period,
            remaining=cap - used_this_period,
        )
