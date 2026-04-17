# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ResourcePool + AgentEconomicState

"""Shared-resource accounting with regeneration and a
per-agent consumption ledger.

The pool exposes a monotonic clock-driven regeneration rule:
between two calls it refills at ``regeneration_rate`` units per
second up to ``capacity``. Consumption records are immutable so
downstream analysers (bargaining, tragedy detection) can query
the last-N or since-T window without worrying about mutation.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


class PoolError(ValueError):
    """Raised when a consumption request is malformed or the pool
    rejects the draw."""


@dataclass(frozen=True)
class ConsumptionRecord:
    """One draw from the pool."""

    agent_id: str
    amount: float
    timestamp: float

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise PoolError("agent_id must be non-empty")
        if self.amount <= 0:
            raise PoolError("amount must be positive")
        if self.timestamp < 0:
            raise PoolError("timestamp must be non-negative")


@dataclass(frozen=True)
class AgentEconomicState:
    """Per-agent economic snapshot.

    ``credit_balance`` is the agent's remaining quota (reset by
    the orchestrator on its own cadence). ``valuation`` is the
    caller-supplied marginal utility per unit of the resource —
    used by :class:`NashBargainingSolver`. ``recent_consumption``
    is the running total the pool itself tracks; callers treat
    it as informational.
    """

    agent_id: str
    credit_balance: float
    valuation: float
    recent_consumption: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise PoolError("agent_id must be non-empty")
        if self.credit_balance < 0:
            raise PoolError("credit_balance must be non-negative")
        if self.valuation < 0:
            raise PoolError("valuation must be non-negative")
        if self.recent_consumption < 0:
            raise PoolError("recent_consumption must be non-negative")


class ResourcePool:
    """Shared resource with a regeneration rule.

    Parameters
    ----------
    capacity :
        Maximum balance the pool can hold. Must be positive.
    regeneration_rate :
        Units added per second up to ``capacity``. Zero means a
        non-regenerating pool (hard cap). Must be non-negative.
    initial_balance :
        Starting balance. Defaults to ``capacity``.
    ledger_size :
        Max retained consumption records (FIFO). Default 10 000.
    clock :
        Timestamp source. Injection point for tests.
    """

    def __init__(
        self,
        *,
        capacity: float,
        regeneration_rate: float = 0.0,
        initial_balance: float | None = None,
        ledger_size: int = 10_000,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if capacity <= 0:
            raise PoolError("capacity must be positive")
        if regeneration_rate < 0:
            raise PoolError("regeneration_rate must be non-negative")
        if ledger_size <= 0:
            raise PoolError("ledger_size must be positive")
        if initial_balance is None:
            initial_balance = capacity
        if initial_balance < 0 or initial_balance > capacity:
            raise PoolError(
                "initial_balance must be in [0, capacity]"
            )
        self._capacity = capacity
        self._regen_rate = regeneration_rate
        self._balance = initial_balance
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._last_refilled_at = float(self._clock())
        self._ledger: deque[ConsumptionRecord] = deque(maxlen=ledger_size)

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def regeneration_rate(self) -> float:
        return self._regen_rate

    def balance(self) -> float:
        with self._lock:
            self._refill_locked()
            return self._balance

    def consume(self, *, agent_id: str, amount: float) -> ConsumptionRecord:
        """Atomically draw ``amount`` from the pool. Raises
        :class:`PoolError` when the request cannot be satisfied."""
        if amount <= 0:
            raise PoolError("amount must be positive")
        with self._lock:
            self._refill_locked()
            if amount > self._balance:
                raise PoolError(
                    f"insufficient balance: requested {amount:.3f}, "
                    f"available {self._balance:.3f}"
                )
            self._balance -= amount
            record = ConsumptionRecord(
                agent_id=agent_id,
                amount=amount,
                timestamp=float(self._clock()),
            )
            self._ledger.append(record)
            return record

    def ledger(self) -> tuple[ConsumptionRecord, ...]:
        with self._lock:
            return tuple(self._ledger)

    def recent(
        self, *, since_seconds: float
    ) -> tuple[ConsumptionRecord, ...]:
        if since_seconds <= 0:
            raise PoolError("since_seconds must be positive")
        cutoff = float(self._clock()) - since_seconds
        with self._lock:
            return tuple(r for r in self._ledger if r.timestamp >= cutoff)

    def _refill_locked(self) -> None:
        if self._regen_rate == 0.0:
            return
        now = float(self._clock())
        elapsed = max(0.0, now - self._last_refilled_at)
        if elapsed == 0.0:
            return
        replenished = min(
            self._capacity - self._balance, elapsed * self._regen_rate
        )
        self._balance = min(self._capacity, self._balance + replenished)
        self._last_refilled_at = now

    def reset(self, *, balance: float | None = None) -> None:
        """Set the balance back to a caller-chosen level (defaults
        to capacity) and clear the ledger. Operators use this
        between billing windows."""
        target = balance if balance is not None else self._capacity
        if target < 0 or target > self._capacity:
            raise PoolError("balance must be in [0, capacity]")
        with self._lock:
            self._balance = target
            self._ledger.clear()
            self._last_refilled_at = float(self._clock())
