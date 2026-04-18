# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ComputeQuota

"""Per-tenant multi-day compute quota.

The quota tracks each tenant's cumulative consumption for every
day within its rolling window. Day boundaries come from
``int(timestamp // 86400)`` so timezone-agnostic deployments
get UTC days; callers who want local boundaries pass a
``day_resolver`` instead.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


class QuotaError(ValueError):
    """Raised when the quota rejects a consumption request."""


DayResolver = Callable[[float], int]


@dataclass(frozen=True)
class DailyUsage:
    """One day's usage record for one tenant."""

    tenant_id: str
    day: int
    consumed: float

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise QuotaError("tenant_id must be non-empty")
        if self.consumed < 0:
            raise QuotaError("consumed must be non-negative")


class ComputeQuota:
    """Multi-day quota with per-tenant daily limits.

    Parameters
    ----------
    daily_limit :
        Maximum consumption per tenant per day. Positive.
    window_days :
        How many days of usage history to retain. Default 14 —
        two weeks is enough for most forecasters and weekly
        seasonality.
    clock :
        Timestamp source.
    day_resolver :
        ``timestamp -> day_index``. Default resolves UTC days.
    """

    def __init__(
        self,
        *,
        daily_limit: float,
        window_days: int = 14,
        clock: Callable[[], float] | None = None,
        day_resolver: DayResolver | None = None,
    ) -> None:
        if daily_limit <= 0:
            raise QuotaError("daily_limit must be positive")
        if window_days <= 0:
            raise QuotaError("window_days must be positive")
        self._daily_limit = daily_limit
        self._window = window_days
        self._clock = clock or time.time
        self._day_resolver = day_resolver or _utc_day
        self._lock = threading.Lock()
        self._usage: dict[str, deque[tuple[int, float]]] = {}

    @property
    def daily_limit(self) -> float:
        return self._daily_limit

    def current_day(self) -> int:
        return self._day_resolver(float(self._clock()))

    def remaining_today(self, tenant_id: str) -> float:
        if not tenant_id:
            raise QuotaError("tenant_id must be non-empty")
        with self._lock:
            today = self.current_day()
            used = self._used_on_day_locked(tenant_id, today)
        return max(0.0, self._daily_limit - used)

    def consume(self, *, tenant_id: str, amount: float) -> DailyUsage:
        """Atomically debit ``amount`` from ``tenant_id``'s daily
        bucket. Raises when the draw would cross the daily limit."""
        if not tenant_id:
            raise QuotaError("tenant_id must be non-empty")
        if amount <= 0:
            raise QuotaError("amount must be positive")
        with self._lock:
            today = self.current_day()
            self._trim_locked(tenant_id)
            used_today = self._used_on_day_locked(tenant_id, today)
            projected = used_today + amount
            if projected > self._daily_limit:
                raise QuotaError(
                    f"tenant {tenant_id!r} would exceed daily limit: "
                    f"used {used_today:.2f} + requested {amount:.2f} "
                    f"> limit {self._daily_limit:.2f}"
                )
            bucket = self._usage.setdefault(tenant_id, deque())
            self._append_bucket(bucket, today, amount)
            return DailyUsage(tenant_id=tenant_id, day=today, consumed=amount)

    def usage_history(self, tenant_id: str) -> tuple[DailyUsage, ...]:
        """Return this tenant's usage as an ordered tuple oldest
        → newest day. Aggregates multiple consumes on the same
        day into a single record."""
        if not tenant_id:
            raise QuotaError("tenant_id must be non-empty")
        with self._lock:
            self._trim_locked(tenant_id)
            bucket = list(self._usage.get(tenant_id, ()))
        return tuple(
            DailyUsage(tenant_id=tenant_id, day=day, consumed=amount)
            for day, amount in bucket
        )

    def reset(self, tenant_id: str | None = None) -> None:
        with self._lock:
            if tenant_id is None:
                self._usage.clear()
            elif tenant_id in self._usage:
                del self._usage[tenant_id]

    def _trim_locked(self, tenant_id: str) -> None:
        bucket = self._usage.get(tenant_id)
        if bucket is None:
            return
        cutoff = self.current_day() - self._window + 1
        while bucket and bucket[0][0] < cutoff:
            bucket.popleft()

    def _used_on_day_locked(self, tenant_id: str, day: int) -> float:
        bucket = self._usage.get(tenant_id)
        if not bucket:
            return 0.0
        for recorded_day, consumed in bucket:
            if recorded_day == day:
                return consumed
        return 0.0

    def _append_bucket(
        self,
        bucket: deque[tuple[int, float]],
        day: int,
        amount: float,
    ) -> None:
        if bucket and bucket[-1][0] == day:
            current_day, current_amount = bucket.pop()
            bucket.append((current_day, current_amount + amount))
            return
        bucket.append((day, amount))


def _utc_day(timestamp: float) -> int:
    return int(timestamp // 86_400)
