# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - physical action budget limiter

"""Per-tenant budgets for physical action checks."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

__all__ = [
    "PhysicalBudgetExceededError",
    "PhysicalBudgetLimits",
    "TenantPhysicalBudget",
]


_COUNTERS = frozenset(
    {"action_validations", "inverse_kinematics", "simulation_checks", "sensor_fusion"}
)


@dataclass(frozen=True)
class PhysicalBudgetLimits:
    """Fixed-window physical check limits for one tenant."""

    window_seconds: float = 60.0
    max_action_validations: int = 120
    max_inverse_kinematics: int = 30
    max_simulation_checks: int = 60
    max_sensor_fusion: int = 60

    def __post_init__(self) -> None:
        """Reject non-positive windows and per-counter limits."""
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        for name in (
            "max_action_validations",
            "max_inverse_kinematics",
            "max_simulation_checks",
            "max_sensor_fusion",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def limit_for(self, counter: str) -> int:
        """Return the configured per-window limit for ``counter``."""
        if counter == "action_validations":
            return self.max_action_validations
        if counter == "inverse_kinematics":
            return self.max_inverse_kinematics
        if counter == "simulation_checks":
            return self.max_simulation_checks
        if counter == "sensor_fusion":
            return self.max_sensor_fusion
        raise ValueError(f"unknown physical budget counter {counter!r}")


@dataclass(frozen=True)
class PhysicalBudgetExceededError(Exception):
    """Raised when a tenant exhausts a physical action budget."""

    tenant_id: str
    counter: str
    limit: int
    window_seconds: float

    def __str__(self) -> str:
        """Render the exhausted counter with its limit and window."""
        return (
            f"physical budget exceeded for {self.counter}: "
            f"limit {self.limit} per {self.window_seconds:.3f}s window"
        )


@dataclass
class _UsageWindow:
    started_at: float
    action_validations: int = 0
    inverse_kinematics: int = 0
    simulation_checks: int = 0
    sensor_fusion: int = 0


class TenantPhysicalBudget:
    """Thread-safe fixed-window budget tracker keyed by tenant id."""

    def __init__(
        self,
        limits: PhysicalBudgetLimits | None = None,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.limits = limits or PhysicalBudgetLimits()
        self._clock = clock
        self._lock = threading.Lock()
        self._windows: dict[str, _UsageWindow] = {}

    def consume(self, tenant_id: str, counter: str) -> None:
        """Consume one unit from a named tenant counter."""
        if counter not in _COUNTERS:
            raise ValueError(f"unknown physical budget counter {counter!r}")
        tenant_key = tenant_id or "default"
        now = self._clock()
        with self._lock:
            window = self._windows.get(tenant_key)
            if window is None or now - window.started_at >= self.limits.window_seconds:
                window = _UsageWindow(started_at=now)
                self._windows[tenant_key] = window
            current = getattr(window, counter)
            limit = self.limits.limit_for(counter)
            if current >= limit:
                raise PhysicalBudgetExceededError(
                    tenant_id=tenant_key,
                    counter=counter,
                    limit=limit,
                    window_seconds=self.limits.window_seconds,
                )
            setattr(window, counter, current + 1)

    def snapshot(self, tenant_id: str) -> dict[str, int]:
        """Return the current tenant counters for tests and telemetry."""
        tenant_key = tenant_id or "default"
        with self._lock:
            window = self._windows.get(tenant_key)
            if window is None:
                return {
                    "action_validations": 0,
                    "inverse_kinematics": 0,
                    "simulation_checks": 0,
                    "sensor_fusion": 0,
                }
            return {
                "action_validations": window.action_validations,
                "inverse_kinematics": window.inverse_kinematics,
                "simulation_checks": window.simulation_checks,
                "sensor_fusion": window.sensor_fusion,
            }
