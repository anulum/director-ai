# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rényi Differential Privacy accountant

"""Rényi-DP (RDP) accounting for tight composition of Gaussian mechanisms.

The :class:`~director_ai.core.federated_privacy.accountant.PrivacyAccountant`
tracks ``(ε, δ)`` under basic or advanced (Dwork-Rothblum-Vadhan) composition.
Both are loose for many compositions of the Gaussian mechanism: advanced
composition grows ``ε`` like ``√k`` but with a large constant, and basic
composition grows it linearly. RDP closes that gap.

This accountant tracks the Rényi divergence at a grid of orders ``α`` and
converts to ``(ε, δ)``-DP at release time, which is the standard tight route for
the Gaussian mechanism used by DP decoding and DP score release.

Mathematics (Mironov, *Rényi Differential Privacy*, CSF 2017):

* **Gaussian RDP** — a mechanism with L2-sensitivity ``Δ`` perturbed by
  ``N(0, σ²)`` is ``(α, α·Δ² / (2σ²))``-RDP for every ``α > 1`` (Mironov 2017,
  Prop. 7 / Cor. 3). With *noise multiplier* ``z = σ / Δ`` this is ``α / (2z²)``.
* **Composition** — RDP at a fixed order is additive: composing mechanisms that
  are ``(α, ε_i)``-RDP yields ``(α, Σ ε_i)``-RDP (Mironov 2017, Prop. 1).
* **Conversion** — an ``(α, ε_RDP)``-RDP mechanism is
  ``(ε_RDP + ln(1/δ) / (α − 1), δ)``-DP for any ``δ ∈ (0, 1)`` (Mironov 2017,
  Prop. 3). The reported ``ε`` is the minimum of that bound over the order grid.

No fabricated constants: every formula above is the published RDP result. The
default order grid matches the de-facto standard used by RDP accountants
(fractional orders ``1.1 … 10.9`` plus integer orders ``11 … 63``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

try:
    from backfire_kernel import rust_sum_f64

    _RUST_RDP = True
except Exception:  # pragma: no cover - mandatory dependency
    _RUST_RDP = True

    def rust_sum_f64(_values: list[float]) -> float:
        raise RuntimeError("backfire_kernel rust_sum_f64 is unavailable")


def _default_orders() -> tuple[float, ...]:
    """The standard RDP order grid: dense fractional orders plus integers."""
    fractional = [1.0 + x / 10.0 for x in range(1, 100)]
    integer = [float(x) for x in range(11, 64)]
    return tuple(fractional + integer)


def gaussian_rdp(order: float, noise_multiplier: float) -> float:
    """RDP of the Gaussian mechanism at a single order.

    Parameters
    ----------
    order:
        The Rényi order ``α``; must be strictly greater than one.
    noise_multiplier:
        ``z = σ / Δ`` — the Gaussian noise standard deviation in units of the
        query's L2 sensitivity. ``z = 0`` (no noise) has unbounded RDP, returned
        as ``math.inf``.

    Returns
    -------
    float
        ``α / (2 z²)`` — the Gaussian-mechanism RDP at order ``α``.
    """
    if order <= 1.0:
        raise ValueError("order must be > 1")
    if noise_multiplier < 0.0:
        raise ValueError("noise_multiplier must be non-negative")
    if noise_multiplier == 0.0:
        return math.inf
    return order / (2.0 * noise_multiplier * noise_multiplier)


@dataclass(frozen=True)
class DPGuarantee:
    """The ``(ε, δ)``-DP guarantee converted from the accumulated RDP."""

    epsilon: float
    delta: float
    order: float

    def to_dict(self) -> dict[str, float]:
        """Tenant-safe view of the converted guarantee."""
        return {"epsilon": self.epsilon, "delta": self.delta, "order": self.order}


class RenyiAccountant:
    """Accumulate RDP across mechanisms and convert to ``(ε, δ)``-DP.

    The accountant holds one running RDP value per order in its grid. Each
    composed mechanism adds its per-order RDP; :meth:`epsilon` then converts the
    accumulated curve to the tightest ``(ε, δ)``-DP bound over the grid.

    Parameters
    ----------
    orders:
        The Rényi orders to track. Defaults to the standard grid. All orders
        must be strictly greater than one.
    """

    def __init__(self, *, orders: Sequence[float] | None = None) -> None:
        grid = tuple(orders) if orders is not None else _default_orders()
        if not grid:
            raise ValueError("orders must be non-empty")
        cleaned = sorted({float(a) for a in grid})
        if cleaned[0] <= 1.0:
            raise ValueError("all orders must be > 1")
        self._orders: tuple[float, ...] = tuple(cleaned)
        self._rdp: list[float] = [0.0] * len(self._orders)

    @property
    def orders(self) -> tuple[float, ...]:
        """The Rényi orders tracked by this accountant."""
        return self._orders

    def rdp_curve(self) -> tuple[float, ...]:
        """The accumulated RDP value at each tracked order."""
        return tuple(self._rdp)

    def rdp_at(self, order: float) -> float:
        """The accumulated RDP at a specific tracked order."""
        try:
            idx = self._orders.index(float(order))
        except ValueError:
            raise KeyError(f"order {order} is not tracked") from None
        return self._rdp[idx]

    def compose_gaussian(
        self, *, noise_multiplier: float, steps: int = 1
    ) -> RenyiAccountant:
        """Compose ``steps`` applications of the Gaussian mechanism.

        Adds ``steps · α / (2 z²)`` to the running RDP at every order ``α``.
        Returns ``self`` so compositions can be chained.
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps == 0:
            return self
        for i, order in enumerate(self._orders):
            self._rdp[i] += steps * gaussian_rdp(order, noise_multiplier)
        return self

    def compose_rdp(self, rdp_per_order: Sequence[float]) -> RenyiAccountant:
        """Compose a mechanism given its RDP at each tracked order.

        ``rdp_per_order`` must align with :attr:`orders`. Use this for any
        mechanism whose per-order RDP is known (additive composition).
        """
        if len(rdp_per_order) != len(self._orders):
            raise ValueError(
                f"rdp_per_order must have {len(self._orders)} entries "
                f"(one per tracked order), got {len(rdp_per_order)}"
            )
        for i, value in enumerate(rdp_per_order):
            if value < 0.0:
                raise ValueError("RDP values must be non-negative")
            self._rdp[i] += float(value)
        return self

    def epsilon(self, *, delta: float) -> DPGuarantee:
        """Convert the accumulated RDP to the tightest ``(ε, δ)``-DP bound.

        For each order ``α`` the Mironov (2017, Prop. 3) bound is
        ``ε_RDP(α) + ln(1/δ) / (α − 1)``; the reported guarantee is the minimum
        over the grid (and the order achieving it).

        An accountant with no composed mechanism (an all-zero RDP curve) is
        trivially ``(0, δ)``-DP — nothing was released — so ``ε = 0`` is reported
        directly rather than the spurious positive ``ln(1/δ) / (α − 1)`` residual
        the conversion formula would give at zero RDP mass.
        """
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if all(rdp == 0.0 for rdp in self._rdp):
            return DPGuarantee(epsilon=0.0, delta=delta, order=self._orders[-1])
        log_inv_delta = math.log(1.0 / delta)
        best_eps = math.inf
        best_order = self._orders[0]
        for order, rdp in zip(self._orders, self._rdp, strict=True):
            candidate = rdp + log_inv_delta / (order - 1.0)
            if candidate < best_eps:
                best_eps = candidate
                best_order = order
        return DPGuarantee(epsilon=best_eps, delta=delta, order=best_order)

    def total_rdp_mass(self) -> float:
        """Sum of the RDP curve — a scalar progress signal for budget displays.

        Not a privacy quantity on its own (RDP is per-order); exposed only so a
        dashboard can show monotone budget consumption without leaking the curve.
        """
        return _sum_float(list(self._rdp))


def _sum_float(values: list[float]) -> float:
    if not values:
        return 0.0
    if _RUST_RDP:
        try:
            return float(rust_sum_f64(values))
        except Exception as exc:
            _logger.debug(
                "Rust RDP sum unavailable; using Python fallback: %s",
                exc,
            )
    return sum(values)
