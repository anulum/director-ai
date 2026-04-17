# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — per-tenant risk budget

"""Sliding-window risk budget per tenant.

The :class:`PromptRiskScorer` assigns every prompt a ``[0, 1]``
risk score; the budget tracks a rolling sum of those scores per
tenant over a configurable window and refuses further reservations
once the window total exceeds the allowance.

The budget is intentionally simple: in-memory, single-process, no
persistence. A Redis-backed variant can plug in later by
implementing the same public contract — ``reserve``, ``snapshot``,
``reset`` — which is why :class:`RiskBudget` is kept free of any
Redis coupling today.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetEntry:
    """Snapshot of one tenant's current budget state.

    ``accepted`` distinguishes "reservation charged" from "over
    budget — ledger untouched". Without this flag a caller could
    not tell whether ``remaining > 0`` meant "reservation applied"
    or "reservation denied and ledger unchanged".
    """

    tenant_id: str
    window_seconds: float
    allowance: float
    consumed: float
    remaining: float
    events: int
    accepted: bool = True

    @property
    def exhausted(self) -> bool:
        """True when the last reservation was refused or the budget
        is empty — the caller should reject the request."""
        return not self.accepted or self.remaining <= 0.0


@dataclass
class _TenantLedger:
    """Private per-tenant deque of ``(timestamp, risk)`` entries."""

    entries: deque[tuple[float, float]] = field(default_factory=deque)
    total: float = 0.0

    def prune(self, cutoff: float) -> None:
        while self.entries and self.entries[0][0] < cutoff:
            _, expired = self.entries.popleft()
            self.total -= expired
            if self.total < 0.0:  # pragma: no cover — float drift guard
                self.total = 0.0

    def push(self, timestamp: float, risk: float) -> None:
        self.entries.append((timestamp, risk))
        self.total += risk

    def count(self) -> int:
        return len(self.entries)


class RiskBudget:
    """Sliding-window risk budget with per-tenant ledgers.

    Parameters
    ----------
    allowance :
        Total risk the tenant may consume in one window.
    window_seconds :
        Window length. Events older than this are pruned on the
        next access.
    clock :
        Injectable ``() -> float`` — tests pass a fake clock to make
        the window deterministic. Defaults to :func:`time.monotonic`.
    default_allowances :
        Optional per-tenant overrides; missing tenants fall back to
        the global ``allowance``.
    """

    def __init__(
        self,
        *,
        allowance: float = 10.0,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        default_allowances: dict[str, float] | None = None,
    ) -> None:
        if allowance <= 0.0:
            raise ValueError(f"allowance must be positive; got {allowance}")
        if window_seconds <= 0.0:
            raise ValueError(
                f"window_seconds must be positive; got {window_seconds}"
            )
        self._allowance = float(allowance)
        self._window = float(window_seconds)
        self._clock = clock
        self._overrides: dict[str, float] = dict(default_allowances or {})
        self._ledgers: dict[str, _TenantLedger] = {}
        self._lock = threading.Lock()

    def set_allowance(self, tenant_id: str, allowance: float) -> None:
        """Override the allowance for one tenant."""
        if allowance <= 0.0:
            raise ValueError(f"allowance must be positive; got {allowance}")
        with self._lock:
            self._overrides[tenant_id] = float(allowance)

    def allowance_for(self, tenant_id: str) -> float:
        """Return the effective allowance for ``tenant_id``."""
        with self._lock:
            return self._overrides.get(tenant_id, self._allowance)

    def reserve(self, tenant_id: str, risk: float) -> BudgetEntry:
        """Try to reserve ``risk`` for ``tenant_id``.

        Returns a :class:`BudgetEntry` reflecting the post-decision
        state. The reservation succeeds when ``entry.remaining >= 0``
        after the charge; callers should check ``entry.exhausted``
        to decide whether to reject / queue the request.

        ``risk`` is clamped to ``[0, 1]``. Pass ``0`` to check the
        current state without consuming budget.
        """
        if risk < 0.0:
            risk = 0.0
        elif risk > 1.0:
            risk = 1.0
        now = self._clock()
        cutoff = now - self._window
        with self._lock:
            ledger = self._ledgers.setdefault(tenant_id, _TenantLedger())
            ledger.prune(cutoff)
            allowance = self._overrides.get(tenant_id, self._allowance)
            tentative = ledger.total + risk
            if tentative <= allowance + 1e-9:
                if risk > 0.0:
                    ledger.push(now, risk)
                consumed = ledger.total
                remaining = allowance - consumed
                return BudgetEntry(
                    tenant_id=tenant_id,
                    window_seconds=self._window,
                    allowance=allowance,
                    consumed=consumed,
                    remaining=remaining,
                    events=ledger.count(),
                    accepted=True,
                )
            # Over budget — report without charging.
            consumed = ledger.total
            return BudgetEntry(
                tenant_id=tenant_id,
                window_seconds=self._window,
                allowance=allowance,
                consumed=consumed,
                remaining=allowance - consumed,
                events=ledger.count(),
                accepted=False,
            )

    def snapshot(self, tenant_id: str) -> BudgetEntry:
        """Read current ledger state without charging."""
        return self.reserve(tenant_id, 0.0)

    def reset(self, tenant_id: str | None = None) -> None:
        """Clear one tenant's ledger (``tenant_id`` given) or every
        ledger at once (``tenant_id=None``). Useful for tests and
        for operator-triggered 'new epoch' events."""
        with self._lock:
            if tenant_id is None:
                self._ledgers.clear()
            else:
                self._ledgers.pop(tenant_id, None)
