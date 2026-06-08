# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rényi-DP accountant tests
"""Multi-angle tests for the Rényi-DP accountant.

Covers the Gaussian RDP formula and its edge cases, the order grid and its
validation, additive composition (single multi-step vs repeated single steps),
generic per-order composition, the Mironov ``(ε, δ)`` conversion checked against
an independent closed-form reference, best-order selection, the tightness of RDP
versus basic/advanced composition for many Gaussian steps, and the tenant-safe
scalar progress signal.
"""

from __future__ import annotations

import math

import pytest

from director_ai.core.federated_privacy import (
    DPGuarantee,
    PrivacyAccountant,
    RenyiAccountant,
    gaussian_rdp,
)
from director_ai.core.federated_privacy import rdp_accountant as rdp_accountant_mod
from director_ai.core.federated_privacy.accountant import AccountantEntry
from director_ai.core.federated_privacy.rdp_accountant import (
    _default_orders,
    _sum_float,
)


def _reference_epsilon(rdp_per_order, orders, delta):
    """Independent Mironov-conversion reference: min_α rdp(α) + ln(1/δ)/(α-1)."""
    log_inv = math.log(1.0 / delta)
    return min(
        rdp + log_inv / (order - 1.0)
        for order, rdp in zip(orders, rdp_per_order, strict=True)
    )


class TestGaussianRdp:
    def test_matches_closed_form(self):
        # α / (2 z²)
        assert gaussian_rdp(2.0, 1.0) == pytest.approx(2.0 / 2.0)
        assert gaussian_rdp(10.0, 2.0) == pytest.approx(10.0 / (2.0 * 4.0))

    def test_monotone_increasing_in_order(self):
        z = 1.5
        vals = [gaussian_rdp(a, z) for a in (1.1, 2.0, 5.0, 20.0)]
        assert vals == sorted(vals)
        assert len(set(vals)) == len(vals)

    def test_decreases_with_more_noise(self):
        # More noise (larger z) → smaller RDP at a fixed order.
        assert gaussian_rdp(5.0, 1.0) > gaussian_rdp(5.0, 4.0)

    def test_zero_noise_is_infinite(self):
        assert gaussian_rdp(3.0, 0.0) == math.inf

    def test_order_must_exceed_one(self):
        with pytest.raises(ValueError, match="order must be > 1"):
            gaussian_rdp(1.0, 1.0)
        with pytest.raises(ValueError, match="order must be > 1"):
            gaussian_rdp(0.5, 1.0)

    def test_negative_noise_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            gaussian_rdp(2.0, -1.0)


class TestOrderGrid:
    def test_default_grid_is_fractional_plus_integer(self):
        orders = _default_orders()
        assert orders[0] == pytest.approx(1.1)
        assert 5.8 in orders
        assert 11.0 in orders
        assert 63.0 in orders
        # All strictly greater than one and strictly increasing.
        assert all(a > 1.0 for a in orders)
        assert list(orders) == sorted(orders)

    def test_custom_orders_sorted_and_deduplicated(self):
        acc = RenyiAccountant(orders=[5.0, 2.0, 2.0, 9.0])
        assert acc.orders == (2.0, 5.0, 9.0)

    def test_empty_orders_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            RenyiAccountant(orders=[])

    def test_order_at_or_below_one_rejected(self):
        with pytest.raises(ValueError, match="must be > 1"):
            RenyiAccountant(orders=[1.0, 2.0])
        with pytest.raises(ValueError, match="must be > 1"):
            RenyiAccountant(orders=[0.5, 3.0])


class TestComposition:
    def test_fresh_accountant_has_zero_curve(self):
        acc = RenyiAccountant(orders=[2.0, 4.0])
        assert acc.rdp_curve() == (0.0, 0.0)

    def test_gaussian_step_adds_per_order_rdp(self):
        acc = RenyiAccountant(orders=[2.0, 10.0])
        acc.compose_gaussian(noise_multiplier=1.0, steps=1)
        assert acc.rdp_at(2.0) == pytest.approx(gaussian_rdp(2.0, 1.0))
        assert acc.rdp_at(10.0) == pytest.approx(gaussian_rdp(10.0, 1.0))

    def test_multistep_equals_repeated_single_steps(self):
        bulk = RenyiAccountant().compose_gaussian(noise_multiplier=2.0, steps=10)
        repeated = RenyiAccountant()
        for _ in range(10):
            repeated.compose_gaussian(noise_multiplier=2.0, steps=1)
        for a, b in zip(bulk.rdp_curve(), repeated.rdp_curve(), strict=True):
            assert a == pytest.approx(b, rel=1e-12)

    def test_compose_returns_self_for_chaining(self):
        acc = RenyiAccountant(orders=[2.0])
        assert acc.compose_gaussian(noise_multiplier=1.0) is acc
        assert acc.compose_rdp([0.5]) is acc

    def test_zero_steps_is_noop(self):
        acc = RenyiAccountant(orders=[2.0, 3.0])
        acc.compose_gaussian(noise_multiplier=1.0, steps=0)
        assert acc.rdp_curve() == (0.0, 0.0)

    def test_negative_steps_rejected(self):
        with pytest.raises(ValueError, match="steps must be non-negative"):
            RenyiAccountant().compose_gaussian(noise_multiplier=1.0, steps=-1)

    def test_compose_rdp_alignment_required(self):
        acc = RenyiAccountant(orders=[2.0, 3.0])
        with pytest.raises(ValueError, match="one per tracked order"):
            acc.compose_rdp([0.1])

    def test_compose_rdp_rejects_negative(self):
        acc = RenyiAccountant(orders=[2.0, 3.0])
        with pytest.raises(ValueError, match="non-negative"):
            acc.compose_rdp([0.1, -0.2])

    def test_compose_rdp_adds_curve(self):
        acc = RenyiAccountant(orders=[2.0, 3.0])
        acc.compose_rdp([0.1, 0.2]).compose_rdp([0.3, 0.4])
        assert acc.rdp_curve() == pytest.approx((0.4, 0.6))


class TestEpsilonConversion:
    def test_empty_composition_is_zero_epsilon(self):
        # No mechanism composed → trivially (0, δ)-DP, not the spurious
        # ln(1/δ)/(α-1) residual the conversion formula gives at zero RDP mass.
        acc = RenyiAccountant()
        guarantee = acc.epsilon(delta=1e-5)
        assert guarantee.epsilon == 0.0
        assert guarantee.delta == 1e-5

    def test_zero_step_gaussian_is_zero_epsilon(self):
        acc = RenyiAccountant().compose_gaussian(noise_multiplier=2.0, steps=0)
        assert acc.epsilon(delta=1e-6).epsilon == 0.0

    def test_matches_independent_reference(self):
        acc = RenyiAccountant()
        acc.compose_gaussian(noise_multiplier=1.0, steps=1)
        guarantee = acc.epsilon(delta=1e-5)
        ref = _reference_epsilon(acc.rdp_curve(), acc.orders, 1e-5)
        assert isinstance(guarantee, DPGuarantee)
        assert guarantee.epsilon == pytest.approx(ref)
        assert guarantee.delta == 1e-5

    def test_known_single_gaussian_value(self):
        # z = 1, δ = 1e-5: analytic optimum ε ≈ 5.2985 at α ≈ 5.8.
        acc = RenyiAccountant()
        acc.compose_gaussian(noise_multiplier=1.0, steps=1)
        guarantee = acc.epsilon(delta=1e-5)
        assert guarantee.epsilon == pytest.approx(5.2985, abs=1e-3)
        assert guarantee.order == pytest.approx(5.8, abs=0.2)

    def test_reported_order_achieves_the_minimum(self):
        acc = RenyiAccountant()
        acc.compose_gaussian(noise_multiplier=1.5, steps=4)
        guarantee = acc.epsilon(delta=1e-6)
        achieved = acc.rdp_at(guarantee.order) + math.log(1.0 / 1e-6) / (
            guarantee.order - 1.0
        )
        assert guarantee.epsilon == pytest.approx(achieved)

    def test_more_steps_increase_epsilon(self):
        few = RenyiAccountant().compose_gaussian(noise_multiplier=2.0, steps=1)
        many = RenyiAccountant().compose_gaussian(noise_multiplier=2.0, steps=50)
        assert many.epsilon(delta=1e-5).epsilon > few.epsilon(delta=1e-5).epsilon

    def test_smaller_delta_increases_epsilon(self):
        acc = RenyiAccountant().compose_gaussian(noise_multiplier=2.0, steps=5)
        assert acc.epsilon(delta=1e-7).epsilon > acc.epsilon(delta=1e-3).epsilon

    def test_delta_must_be_open_unit(self):
        acc = RenyiAccountant().compose_gaussian(noise_multiplier=1.0)
        for bad in (0.0, 1.0, -0.1, 2.0):
            with pytest.raises(ValueError, match="delta must be in"):
                acc.epsilon(delta=bad)

    def test_guarantee_to_dict_is_tenant_safe(self):
        acc = RenyiAccountant().compose_gaussian(noise_multiplier=2.0, steps=3)
        payload = acc.epsilon(delta=1e-5).to_dict()
        assert set(payload) == {"epsilon", "delta", "order"}
        assert all(isinstance(v, float) for v in payload.values())


class TestTightness:
    def test_rdp_tighter_than_basic_composition_for_many_steps(self):
        # 100 Gaussian rounds at z = 4. RDP should give a far smaller ε than the
        # accountant's *basic* (linear-sum) composition of the equivalent
        # per-round (ε, δ) Gaussian guarantee.
        z = 4.0
        steps = 100
        delta = 1e-5
        rdp = RenyiAccountant().compose_gaussian(noise_multiplier=z, steps=steps)
        rdp_eps = rdp.epsilon(delta=delta).epsilon

        # Per-round (ε, δ)-DP of one Gaussian step via RDP on a single step.
        per_round = (
            RenyiAccountant()
            .compose_gaussian(noise_multiplier=z, steps=1)
            .epsilon(delta=delta / steps)
            .epsilon
        )
        basic = PrivacyAccountant(max_epsilon=1e9)
        for _ in range(steps):
            basic.charge(
                AccountantEntry(label="round", epsilon=per_round, delta=delta / steps)
            )
        assert rdp_eps < basic.cumulative_epsilon()


class TestProgressSignal:
    def test_total_rdp_mass_is_monotone_and_starts_zero(self):
        acc = RenyiAccountant(orders=[2.0, 3.0, 4.0])
        assert acc.total_rdp_mass() == 0.0
        acc.compose_gaussian(noise_multiplier=2.0, steps=1)
        first = acc.total_rdp_mass()
        assert first > 0.0
        acc.compose_gaussian(noise_multiplier=2.0, steps=1)
        assert acc.total_rdp_mass() > first

    def test_rdp_at_unknown_order_raises(self):
        acc = RenyiAccountant(orders=[2.0, 3.0])
        with pytest.raises(KeyError, match="not tracked"):
            acc.rdp_at(7.0)


class TestRustSum:
    def test_empty_values_sum_to_zero(self):
        assert _sum_float([]) == 0.0

    def test_rust_sum_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(rdp_accountant_mod, "_RUST_RDP", True)
        called = {"count": 0}

        def _sum(values: list[float]) -> float:
            called["count"] += 1
            return sum(values)

        monkeypatch.setattr(rdp_accountant_mod, "rust_sum_f64", _sum, raising=True)
        assert _sum_float([0.1, 0.2, 0.3]) == pytest.approx(0.6)
        assert called["count"] == 1

    def test_rust_sum_error_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(rdp_accountant_mod, "_RUST_RDP", True)
        monkeypatch.setattr(
            rdp_accountant_mod,
            "rust_sum_f64",
            lambda _values: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        assert _sum_float([1.0, 2.0, 4.0]) == pytest.approx(7.0)
