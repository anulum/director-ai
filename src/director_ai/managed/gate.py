# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""The managed request gate: authenticate, meter, and enforce plan quota.

One call decides a request: resolve the API key to an active account, count what
that account has used since the billing window opened, and let the plan registry
say whether one more request fits. The decision is framework-free — it returns a
:class:`GateOutcome` with the HTTP status and the rate-limit headers to set — so
it is fully testable without a web server; ``fastapi_dependency`` is only a thin
adapter that pulls the key off a request and raises the matching ``HTTPException``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .accounts import Account, AccountStore
from .plans import PlanRegistry, QuotaDecision
from .usage import UsageMeter

if TYPE_CHECKING:
    from collections.abc import Callable


def utc_month_start() -> str:
    """Return the ISO timestamp of the current UTC month's first instant."""
    now = datetime.now(UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """The verdict on one managed request, with the HTTP status and headers."""

    allowed: bool
    status: int
    reason: str
    account: Account | None
    headers: dict[str, str]


def _quota_headers(decision: QuotaDecision, period_start: str) -> dict[str, str]:
    limit = "unlimited" if decision.limit is None else str(decision.limit)
    remaining = "unlimited" if decision.remaining is None else str(decision.remaining)
    return {
        "X-RateLimit-Limit": limit,
        "X-RateLimit-Remaining": remaining,
        "X-Usage-Used": str(decision.used),
        "X-Usage-Period-Start": period_start,
    }


class ManagedGate:
    """Auth + metering + quota over one SQLite control-plane database."""

    def __init__(
        self,
        db_path: str | Path = "director_managed.db",
        plans: PlanRegistry | None = None,
    ) -> None:
        self.accounts = AccountStore(db_path)
        self.usage = UsageMeter(db_path)
        self.plans = plans if plans is not None else PlanRegistry()

    def check(self, api_key: str | None, *, endpoint: str = "") -> GateOutcome:
        """Decide one request and, when allowed, record it for the meter.

        Returns 401 for a missing/invalid key, 429 when the account is over its
        monthly cap, and 200 otherwise. The request is metered only on a 200, so
        rejected calls do not consume quota.
        """
        if not api_key:
            return GateOutcome(False, 401, "missing API key", None, {})
        account = self.accounts.authenticate(api_key)
        if account is None:
            return GateOutcome(False, 401, "invalid or revoked API key", None, {})

        period_start = utc_month_start()
        used = self.usage.request_count(account.account_id, since=period_start)
        decision = self.plans.decide(account.plan, used)
        headers = _quota_headers(decision, period_start)
        if not decision.allowed:
            return GateOutcome(False, 429, decision.reason, account, headers)

        self.usage.record(account.account_id, endpoint or "unknown", key_id=None)
        return GateOutcome(True, 200, decision.reason, account, headers)


def fastapi_dependency(gate: ManagedGate) -> Callable[..., Any]:
    """Return a FastAPI dependency that enforces ``gate`` on each request.

    The dependency extracts a bearer / ``X-API-Key`` token, applies the gate,
    raises ``HTTPException`` on 401/429 (carrying the rate-limit headers), and
    otherwise returns the resolved :class:`Account` and stamps the headers onto
    ``request.state`` for a response hook to copy out.
    """
    from fastapi import HTTPException, Request  # noqa: PLC0415 — optional dep

    async def _dependency(request: Request) -> Account:
        api_key = _extract_key(request)
        outcome = gate.check(api_key, endpoint=request.url.path)
        request.state.managed_quota_headers = outcome.headers
        if not outcome.allowed:
            raise HTTPException(
                status_code=outcome.status,
                detail=outcome.reason,
                headers=outcome.headers or None,
            )
        assert outcome.account is not None  # noqa: S101 — 200 always has account
        request.state.managed_account = outcome.account
        return outcome.account

    return _dependency


def _extract_key(request: Any) -> str | None:  # noqa: ANN401 — fastapi Request
    auth: str = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[len("Bearer ") :].strip()
    header_key: str = request.headers.get("X-API-Key", "")
    return header_key.strip() or None
