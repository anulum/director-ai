# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PrivacyAccountant

"""Track the cumulative ``(ε, δ)`` budget spent across queries.

Two composition modes:

* **Basic** (default) — sum the per-query ``ε`` and ``δ`` values.
  Tight for small query counts; conservative beyond that.
* **Advanced** — Dwork-Rothblum-Vadhan bound: for ``k``
  applications of an ``ε_0``-DP mechanism, the composition
  satisfies ``(ε, δ)``-DP with
  ``ε = √(2k · ln(1/δ_add)) · ε_0 + k · ε_0 · (e^{ε_0} − 1)``
  for any ``δ_add > 0``. The accountant exposes the advanced
  bound via :meth:`epsilon_advanced` so callers can swap between
  tight-low-k and tight-high-k regimes.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class AccountantEntry:
    """One recorded query."""

    label: str
    epsilon: float
    delta: float
    sensitivity: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


class PrivacyAccountant:
    """Thread-safe cumulative ``(ε, δ)`` tracker.

    Parameters
    ----------
    max_epsilon :
        Budget ceiling. :meth:`charge` raises when the next entry
        would cross this value under the active composition mode.
    max_delta :
        Optional ``δ`` ceiling. ``None`` means no δ cap (common
        for pure-ε deployments).
    mode :
        ``"basic"`` or ``"advanced"``. Can be switched at any time
        via :meth:`use_basic` / :meth:`use_advanced`.
    """

    def __init__(
        self,
        *,
        max_epsilon: float,
        max_delta: float | None = None,
        mode: str = "basic",
    ) -> None:
        if max_epsilon <= 0:
            raise ValueError("max_epsilon must be positive")
        if max_delta is not None and max_delta <= 0:
            raise ValueError("max_delta must be positive when supplied")
        if mode not in {"basic", "advanced"}:
            raise ValueError("mode must be 'basic' or 'advanced'")
        self._max_epsilon = max_epsilon
        self._max_delta = max_delta
        self._mode = mode
        self._lock = threading.Lock()
        self._entries: list[AccountantEntry] = []

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def use_basic(self) -> None:
        with self._lock:
            self._mode = "basic"

    def use_advanced(self) -> None:
        with self._lock:
            self._mode = "advanced"

    def charge(self, entry: AccountantEntry) -> None:
        """Record ``entry`` and raise :class:`ValueError` when the
        post-charge total would exceed the budget under the
        active composition mode."""
        if entry.epsilon < 0 or entry.delta < 0:
            raise ValueError("entry epsilon / delta must be non-negative")
        with self._lock:
            projected_epsilon = self._project_epsilon([*self._entries, entry])
            projected_delta = self._project_delta([*self._entries, entry])
            if projected_epsilon > self._max_epsilon:
                raise ValueError(
                    f"charging {entry.label!r} would push epsilon to "
                    f"{projected_epsilon:.4f} > {self._max_epsilon:.4f}"
                )
            if self._max_delta is not None and projected_delta > self._max_delta:
                raise ValueError(
                    f"charging {entry.label!r} would push delta to "
                    f"{projected_delta:.4e} > {self._max_delta:.4e}"
                )
            self._entries.append(entry)

    def cumulative_epsilon(self) -> float:
        with self._lock:
            return self._project_epsilon(self._entries)

    def cumulative_delta(self) -> float:
        with self._lock:
            return self._project_delta(self._entries)

    def entries(self) -> tuple[AccountantEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def epsilon_advanced(self, *, target_delta: float) -> float:
        """Compute the advanced-composition ``ε`` for the
        currently-recorded homogeneous epsilon entries.

        Raises :class:`ValueError` when the entries are not all
        identical in ``ε`` (advanced composition requires
        homogeneous mechanisms — callers wanting heterogeneous
        bounds fall back to the RDP / zCDP accountants which are
        out of scope here)."""
        if not 0.0 < target_delta < 1.0:
            raise ValueError("target_delta must be in (0, 1)")
        with self._lock:
            entries = tuple(self._entries)
        if not entries:
            return 0.0
        eps_values = {e.epsilon for e in entries}
        if len(eps_values) != 1:
            raise ValueError(
                "advanced composition requires homogeneous epsilon "
                "across charged entries"
            )
        eps_0 = next(iter(eps_values))
        k = len(entries)
        return _advanced_epsilon(eps_0=eps_0, k=k, delta_add=target_delta)

    def _project_epsilon(self, entries: list[AccountantEntry]) -> float:
        if self._mode == "basic":
            return sum(e.epsilon for e in entries)
        eps_values = {e.epsilon for e in entries}
        if len(eps_values) != 1:
            return sum(e.epsilon for e in entries)
        eps_0 = next(iter(eps_values))
        k = len(entries)
        # The advanced bound needs an auxiliary δ_add; use half of the
        # max_delta (or 1e-9 when δ cap is unset) so the projected
        # epsilon stays comparable to max_epsilon.
        delta_add = (self._max_delta / 2.0) if self._max_delta else 1e-9
        return _advanced_epsilon(eps_0=eps_0, k=k, delta_add=delta_add)

    def _project_delta(self, entries: list[AccountantEntry]) -> float:
        return sum(e.delta for e in entries)


def _advanced_epsilon(*, eps_0: float, k: int, delta_add: float) -> float:
    """Dwork-Rothblum-Vadhan advanced composition bound."""
    if eps_0 == 0.0:
        return 0.0
    return math.sqrt(2.0 * k * math.log(1.0 / delta_add)) * eps_0 + k * eps_0 * (
        math.exp(eps_0) - 1.0
    )
