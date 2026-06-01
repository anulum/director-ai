# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — federated privacy tests

"""Multi-angle coverage: Laplace / Gaussian calibration,
accountant basic + advanced composition, additive secret
sharing reconstruction, SecureAggregator multi-party total,
FederatedCounter + FederatedHistogram releases with accountant
budget guards, concurrent submissions."""

from __future__ import annotations

import math
import statistics
import threading

import pytest

import director_ai.core.federated_privacy.accountant as accountant_mod
import director_ai.core.federated_privacy.aggregator as aggregator_mod
from director_ai.core.federated_privacy import (
    AccountantEntry,
    FederatedCounter,
    FederatedHistogram,
    FederatedSafetySignalAggregator,
    GaussianMechanism,
    LaplaceMechanism,
    PrivacyAccountant,
    SecretShare,
    SecureAggregator,
    ShareError,
)
from director_ai.core.federated_privacy.secret_sharing import (
    DEFAULT_MODULUS,
    _split_with_rng,
    reconstruct,
    split,
    split_many,
)
from director_ai.core.safety_event import SafetyEvent
from director_ai.core.safety_protocol import director_safety_signal_from_event

# --- LaplaceMechanism ---------------------------------------------


class TestLaplace:
    def test_scale_is_sensitivity_over_epsilon(self):
        m = LaplaceMechanism(
            epsilon=0.5,
            sensitivity=2.0,
            seed=0,
            allow_insecure_seed=True,
        )
        assert m.scale == pytest.approx(4.0)

    def test_zero_sensitivity_produces_zero_scale(self):
        m = LaplaceMechanism(
            epsilon=0.5,
            sensitivity=0.0,
            seed=0,
            allow_insecure_seed=True,
        )
        assert m.scale == 0.0
        assert m.noise() == 0.0

    def test_bad_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            LaplaceMechanism(epsilon=0, sensitivity=1.0)

    def test_bad_sensitivity(self):
        with pytest.raises(ValueError, match="sensitivity"):
            LaplaceMechanism(epsilon=1.0, sensitivity=-0.5)

    def test_noise_has_mean_near_zero(self):
        m = LaplaceMechanism(
            epsilon=1.0,
            sensitivity=1.0,
            seed=42,
            allow_insecure_seed=True,
        )
        samples = [m.noise() for _ in range(5_000)]
        assert abs(statistics.mean(samples)) < 0.1

    def test_noise_variance_matches_laplace(self):
        """Laplace(0, b) has variance 2 b^2. We tolerate 20% error
        on 5 000 samples."""
        m = LaplaceMechanism(
            epsilon=2.0,
            sensitivity=1.0,
            seed=7,
            allow_insecure_seed=True,
        )
        samples = [m.noise() for _ in range(5_000)]
        expected_var = 2 * (m.scale**2)
        assert statistics.pvariance(samples) == pytest.approx(expected_var, rel=0.2)

    def test_seed_requires_explicit_insecure_test_mode(self):
        with pytest.raises(ValueError, match="allow_insecure_seed"):
            LaplaceMechanism(epsilon=1.0, sensitivity=1.0, seed=42)


# --- GaussianMechanism --------------------------------------------


class TestGaussian:
    def test_sigma_formula(self):
        m = GaussianMechanism(
            epsilon=0.5,
            delta=1e-5,
            sensitivity=1.0,
            seed=0,
            allow_insecure_seed=True,
        )
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 0.5
        assert m.sigma == pytest.approx(expected)

    def test_zero_sensitivity_zero_sigma(self):
        m = GaussianMechanism(
            epsilon=0.5,
            delta=1e-5,
            sensitivity=0.0,
            seed=0,
            allow_insecure_seed=True,
        )
        assert m.sigma == 0.0
        assert m.noise() == 0.0

    def test_epsilon_must_be_under_one(self):
        with pytest.raises(ValueError, match="epsilon"):
            GaussianMechanism(epsilon=1.1, delta=1e-5, sensitivity=1.0)

    def test_delta_must_be_unit_interval(self):
        with pytest.raises(ValueError, match="delta"):
            GaussianMechanism(epsilon=0.5, delta=0.0, sensitivity=1.0)

    def test_sensitivity_must_be_non_negative(self):
        with pytest.raises(ValueError, match="sensitivity"):
            GaussianMechanism(epsilon=0.5, delta=1e-5, sensitivity=-0.1)

    def test_properties_and_apply_use_calibrated_gaussian_noise(self):
        m = GaussianMechanism(
            epsilon=0.5,
            delta=1e-5,
            sensitivity=1.0,
            seed=123,
            allow_insecure_seed=True,
        )

        assert m.epsilon == pytest.approx(0.5)
        assert m.delta == pytest.approx(1e-5)
        assert m.sensitivity == pytest.approx(1.0)
        assert m.apply(10.0) != 10.0

    def test_gaussian_seed_requires_explicit_insecure_test_mode(self):
        with pytest.raises(ValueError, match="allow_insecure_seed"):
            GaussianMechanism(
                epsilon=0.5,
                delta=1e-5,
                sensitivity=1.0,
                seed=42,
            )

    def test_laplace_apply_and_sign_direction_are_deterministic_with_seed(self):
        m = LaplaceMechanism(
            epsilon=1.0,
            sensitivity=1.0,
            seed=3,
            allow_insecure_seed=True,
        )

        released = m.apply(5.0)

        assert released != 5.0
        assert isinstance(released, float)

    def test_laplace_midpoint_rng_draw_releases_original_value(self):
        class MidpointRng:
            def uniform(self, low, high):
                assert (low, high) == (-0.5, 0.5)
                return 0.0

        m = LaplaceMechanism(
            epsilon=1.0,
            sensitivity=1.0,
            seed=3,
            allow_insecure_seed=True,
        )
        m._rng = MidpointRng()

        assert m.noise() == 0.0
        assert m.apply(7.0) == 7.0


# --- PrivacyAccountant --------------------------------------------


class TestAccountant:
    def test_basic_composition_sums_epsilon(self):
        acc = PrivacyAccountant(max_epsilon=10.0)
        acc.charge(AccountantEntry(label="q1", epsilon=0.3, delta=0.0))
        acc.charge(AccountantEntry(label="q2", epsilon=0.2, delta=0.0))
        assert acc.cumulative_epsilon() == pytest.approx(0.5)

    def test_budget_ceiling_enforced(self):
        acc = PrivacyAccountant(max_epsilon=0.4)
        acc.charge(AccountantEntry(label="q1", epsilon=0.3, delta=0.0))
        with pytest.raises(ValueError, match="epsilon"):
            acc.charge(AccountantEntry(label="q2", epsilon=0.2, delta=0.0))

    def test_delta_ceiling(self):
        acc = PrivacyAccountant(max_epsilon=10.0, max_delta=1e-5)
        acc.charge(AccountantEntry(label="q1", epsilon=0.1, delta=5e-6))
        with pytest.raises(ValueError, match="delta"):
            acc.charge(AccountantEntry(label="q2", epsilon=0.1, delta=1e-5))

    def test_cumulative_delta_and_entries_snapshot(self):
        acc = PrivacyAccountant(max_epsilon=10.0)
        first = AccountantEntry(
            label="q1",
            epsilon=0.1,
            delta=2e-6,
            sensitivity=1.0,
            metadata={"tenant": "a"},
        )
        second = AccountantEntry(label="q2", epsilon=0.2, delta=3e-6)

        acc.charge(first)
        acc.charge(second)

        assert acc.cumulative_delta() == pytest.approx(5e-6)
        assert acc.entries() == (first, second)

    def test_negative_entries_rejected(self):
        acc = PrivacyAccountant(max_epsilon=10.0)
        with pytest.raises(ValueError, match="non-negative"):
            acc.charge(AccountantEntry(label="q", epsilon=-0.1, delta=0.0))

    def test_mode_switch(self):
        acc = PrivacyAccountant(max_epsilon=10.0)
        assert acc.mode == "basic"
        acc.use_advanced()
        assert acc.mode == "advanced"
        acc.use_basic()
        assert acc.mode == "basic"

    def test_advanced_bound_requires_homogeneity(self):
        acc = PrivacyAccountant(max_epsilon=10.0)
        acc.charge(AccountantEntry(label="q1", epsilon=0.1, delta=0.0))
        acc.charge(AccountantEntry(label="q2", epsilon=0.2, delta=0.0))
        with pytest.raises(ValueError, match="homogeneous"):
            acc.epsilon_advanced(target_delta=1e-6)

    def test_advanced_bound_value(self):
        acc = PrivacyAccountant(max_epsilon=10.0)
        for _ in range(100):
            acc.charge(AccountantEntry(label="q", epsilon=0.1, delta=0.0))
        bound = acc.epsilon_advanced(target_delta=1e-6)
        # Basic composition gives 10 for 100 ε_0=0.1 queries; advanced
        # gives a tighter bound at small ε_0.
        basic = 0.1 * 100
        assert bound < basic

    def test_advanced_bound_empty_and_zero_epsilon(self):
        acc = PrivacyAccountant(max_epsilon=1.0, mode="advanced")

        assert acc.epsilon_advanced(target_delta=1e-6) == 0.0

        acc.charge(AccountantEntry(label="public-stat", epsilon=0.0, delta=0.0))
        assert acc.cumulative_epsilon() == 0.0
        assert acc.epsilon_advanced(target_delta=1e-6) == 0.0

    def test_advanced_mode_homogeneous_projection_uses_delta_cap(self):
        acc = PrivacyAccountant(max_epsilon=10.0, max_delta=1e-4, mode="advanced")
        acc.charge(AccountantEntry(label="q1", epsilon=0.01, delta=1e-6))
        acc.charge(AccountantEntry(label="q2", epsilon=0.01, delta=1e-6))

        advanced = acc.cumulative_epsilon()
        expected = math.sqrt(2.0 * 2 * math.log(1.0 / 5e-5)) * 0.01 + 2 * 0.01 * (
            math.exp(0.01) - 1.0
        )

        assert advanced == pytest.approx(expected)
        assert advanced > 0.02

    def test_advanced_mode_heterogeneous_projection_falls_back_to_basic_sum(self):
        acc = PrivacyAccountant(max_epsilon=10.0, mode="advanced")
        acc.charge(AccountantEntry(label="q1", epsilon=0.1, delta=0.0))
        acc.charge(AccountantEntry(label="q2", epsilon=0.2, delta=0.0))

        assert acc.cumulative_epsilon() == pytest.approx(0.3)

    def test_target_delta_validation(self):
        acc = PrivacyAccountant(max_epsilon=10.0)
        with pytest.raises(ValueError, match="target_delta"):
            acc.epsilon_advanced(target_delta=0.0)

    def test_bad_mode(self):
        with pytest.raises(ValueError, match="mode"):
            PrivacyAccountant(max_epsilon=10.0, mode="weird")

    def test_bad_ceilings(self):
        with pytest.raises(ValueError, match="max_epsilon"):
            PrivacyAccountant(max_epsilon=0.0)
        with pytest.raises(ValueError, match="max_delta"):
            PrivacyAccountant(max_epsilon=1.0, max_delta=0.0)


class TestAccountantRustSums:
    def test_rust_sum_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(accountant_mod, "_RUST_ACCOUNTANT", True)
        called = {"count": 0}

        def _sum(values: list[float]) -> float:
            called["count"] += 1
            return sum(values)

        monkeypatch.setattr(accountant_mod, "rust_sum_f64", _sum, raising=True)
        acc = PrivacyAccountant(max_epsilon=10.0)
        acc.charge(AccountantEntry(label="q1", epsilon=0.1, delta=1e-6))
        acc.charge(AccountantEntry(label="q2", epsilon=0.2, delta=2e-6))
        assert acc.cumulative_epsilon() == pytest.approx(0.3)
        assert called["count"] >= 1

    def test_rust_sum_type_error_falls_back_to_python(self, monkeypatch):
        monkeypatch.setattr(accountant_mod, "_RUST_ACCOUNTANT", True)
        monkeypatch.setattr(
            accountant_mod,
            "rust_sum_f64",
            lambda _values: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        acc = PrivacyAccountant(max_epsilon=10.0)
        acc.charge(AccountantEntry(label="q1", epsilon=0.1, delta=1e-6))
        acc.charge(AccountantEntry(label="q2", epsilon=0.2, delta=2e-6))
        assert acc.cumulative_delta() == pytest.approx(3e-6)


# --- SecretShare + SecureAggregator -------------------------------


class TestSecretSharing:
    def test_reconstruct_roundtrip(self):
        share = split(42, party_count=3, seed=0, allow_insecure_seed=True)
        assert reconstruct(share) == 42

    def test_share_has_correct_party_count(self):
        share = split(7, party_count=5, seed=1, allow_insecure_seed=True)
        assert share.party_count == 5

    def test_share_values_bounded(self):
        share = split(
            100,
            party_count=4,
            seed=2,
            modulus=1000,
            allow_insecure_seed=True,
        )
        for v in share.values:
            assert 0 <= v < 1000

    def test_aggregator_sums_secrets(self):
        aggregator = SecureAggregator(party_count=3)
        s1 = split(10, party_count=3, seed=1, allow_insecure_seed=True)
        s2 = split(20, party_count=3, seed=2, allow_insecure_seed=True)
        s3 = split(15, party_count=3, seed=3, allow_insecure_seed=True)
        aggregator.submit(s1)
        aggregator.submit(s2)
        aggregator.submit(s3)
        assert aggregator.reconstruct() == 45
        assert aggregator.submissions == 3

    def test_aggregator_rejects_mismatched_party_count(self):
        aggregator = SecureAggregator(party_count=3)
        mismatched = split(10, party_count=4, seed=1, allow_insecure_seed=True)
        with pytest.raises(ShareError, match="parties"):
            aggregator.submit(mismatched)

    def test_aggregator_rejects_mismatched_modulus(self):
        aggregator = SecureAggregator(party_count=3, modulus=1_000_003)
        other_modulus = split(
            10,
            party_count=3,
            seed=1,
            modulus=999_983,
            allow_insecure_seed=True,
        )
        with pytest.raises(ShareError, match="modulus"):
            aggregator.submit(other_modulus)

    def test_aggregator_no_submissions(self):
        aggregator = SecureAggregator(party_count=2)
        with pytest.raises(ShareError, match="no submissions"):
            aggregator.reconstruct()

    def test_aggregator_reset(self):
        aggregator = SecureAggregator(party_count=2)
        aggregator.submit(split(5, party_count=2, seed=0, allow_insecure_seed=True))
        aggregator.reset()
        assert aggregator.submissions == 0
        with pytest.raises(ShareError):
            aggregator.reconstruct()

    def test_bad_party_count(self):
        with pytest.raises(ShareError, match="party_count"):
            SecureAggregator(party_count=1)
        with pytest.raises(ShareError, match="party_count"):
            split(1, party_count=1)

    def test_bad_modulus(self):
        with pytest.raises(ShareError, match="modulus"):
            SecureAggregator(party_count=2, modulus=0)
        with pytest.raises(ShareError, match="modulus"):
            split(1, party_count=2, modulus=0)
        with pytest.raises(ShareError, match="modulus"):
            SecretShare(values=(0, 0), modulus=0)

    def test_internal_split_revalidates_party_count_and_modulus(self):
        class _ZeroRng:
            def randrange(self, _modulus):
                return 0

        with pytest.raises(ShareError, match="party_count"):
            _split_with_rng(1, party_count=1, modulus=DEFAULT_MODULUS, rng=_ZeroRng())
        with pytest.raises(ShareError, match="modulus"):
            _split_with_rng(1, party_count=2, modulus=0, rng=_ZeroRng())

    def test_secret_share_requires_at_least_two_parties(self):
        with pytest.raises(ShareError, match="at least two parties"):
            SecretShare(values=(1,), modulus=DEFAULT_MODULUS)

    def test_seed_requires_explicit_insecure_opt_in(self):
        with pytest.raises(ShareError, match="allow_insecure_seed"):
            split(1, party_count=2, seed=123)

    def test_split_many_rejects_empty_secret_list(self):
        with pytest.raises(ShareError, match="non-empty"):
            split_many([], party_count=2)

    def test_share_negative_rejected(self):
        with pytest.raises(ShareError, match="outside"):
            SecretShare(values=(1, -1), modulus=DEFAULT_MODULUS)

    def test_share_too_large_rejected(self):
        with pytest.raises(ShareError, match="outside"):
            SecretShare(values=(1, 1 << 200), modulus=DEFAULT_MODULUS)

    def test_split_many(self):
        shares = split_many(
            [1, 2, 3],
            party_count=3,
            seed=0,
            allow_insecure_seed=True,
        )
        assert len(shares) == 3
        for share, expected in zip(shares, [1, 2, 3], strict=False):
            assert reconstruct(share) == expected

    def test_split_many_unseeded_does_not_downgrade_to_deterministic_rng(
        self,
        monkeypatch,
    ):
        import director_ai.core.federated_privacy.secret_sharing as module

        class FakeSystemRandom:
            def __init__(self):
                self._next = 0

            def randrange(self, modulus):
                self._next += 1
                return self._next % modulus

        def forbidden_random(_seed):
            raise AssertionError("production split_many must not seed random.Random")

        monkeypatch.setattr(module.random, "SystemRandom", FakeSystemRandom)
        monkeypatch.setattr(module.random, "Random", forbidden_random)

        shares = split_many([1, 2, 3], party_count=3)

        assert [reconstruct(share) for share in shares] == [1, 2, 3]

    def test_split_many_empty(self):
        with pytest.raises(ShareError, match="secrets"):
            split_many([], party_count=3)

    def test_reproducible_with_seed(self):
        a = split(123, party_count=5, seed=777, allow_insecure_seed=True)
        b = split(123, party_count=5, seed=777, allow_insecure_seed=True)
        assert a.values == b.values

    def test_secret_sharing_seed_requires_explicit_insecure_test_mode(self):
        with pytest.raises(ShareError, match="allow_insecure_seed"):
            split(123, party_count=5, seed=777)
        with pytest.raises(ShareError, match="allow_insecure_seed"):
            split_many([1, 2], party_count=3, seed=777)


# --- FederatedCounter ---------------------------------------------


class TestFederatedCounter:
    def test_submits_and_releases(self):
        acc = PrivacyAccountant(max_epsilon=5.0)
        counter = FederatedCounter(
            epsilon=0.5,
            sensitivity=1.0,
            accountant=acc,
            seed=0,
            allow_insecure_seed=True,
        )
        counter.submit(tenant_id="t1", count=3)
        counter.submit(tenant_id="t2", count=7)
        release = counter.release()
        assert release.raw_sum == 10
        assert release.epsilon_spent == pytest.approx(0.5)
        assert acc.cumulative_epsilon() == pytest.approx(0.5)

    def test_release_resets_state(self):
        counter = FederatedCounter(epsilon=0.5, seed=0, allow_insecure_seed=True)
        counter.submit(tenant_id="t1", count=1)
        counter.release()
        second = counter.release()
        assert second.raw_sum == 0

    def test_budget_guard(self):
        acc = PrivacyAccountant(max_epsilon=0.4)
        counter = FederatedCounter(
            epsilon=0.5,
            accountant=acc,
            seed=0,
            allow_insecure_seed=True,
        )
        with pytest.raises(ValueError, match="epsilon"):
            counter.release()

    def test_bad_tenant_or_count(self):
        counter = FederatedCounter(epsilon=0.5, seed=0, allow_insecure_seed=True)
        with pytest.raises(ValueError, match="tenant_id"):
            counter.submit(tenant_id="", count=1)
        with pytest.raises(ValueError, match="count"):
            counter.submit(tenant_id="t", count=-1)

    def test_bad_label(self):
        with pytest.raises(ValueError, match="label"):
            FederatedCounter(epsilon=0.5, label="")

    def test_counter_seed_requires_explicit_insecure_test_mode(self):
        with pytest.raises(ValueError, match="allow_insecure_seed"):
            FederatedCounter(epsilon=0.5, seed=0)

    def test_concurrent_submits(self):
        counter = FederatedCounter(epsilon=0.5, seed=0, allow_insecure_seed=True)

        def writer(tag: str) -> None:
            for _ in range(100):
                counter.submit(tenant_id=tag, count=1)

        threads = [threading.Thread(target=writer, args=(f"t{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        release = counter.release()
        assert release.raw_sum == 800


# --- FederatedHistogram -------------------------------------------


class TestFederatedHistogram:
    def test_submit_and_release(self):
        acc = PrivacyAccountant(max_epsilon=5.0)
        hist = FederatedHistogram(
            categories=("spam", "phishing", "safe"),
            epsilon=0.9,
            accountant=acc,
            seed=0,
            allow_insecure_seed=True,
        )
        hist.submit(tenant_id="t1", category="spam", count=3)
        hist.submit(tenant_id="t2", category="phishing", count=1)
        hist.submit(tenant_id="t3", category="safe", count=5)
        release = hist.release()
        assert release.raw_counts["spam"] == 3
        assert release.raw_counts["phishing"] == 1
        assert release.raw_counts["safe"] == 5
        assert release.epsilon_spent == pytest.approx(0.9)
        assert acc.cumulative_epsilon() == pytest.approx(0.9)

    def test_unknown_category_rejected(self):
        hist = FederatedHistogram(
            categories=("a", "b"),
            epsilon=0.5,
            seed=0,
            allow_insecure_seed=True,
        )
        with pytest.raises(KeyError, match="ghost"):
            hist.submit(tenant_id="t", category="ghost")

    def test_duplicate_categories_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            FederatedHistogram(categories=("a", "a"), epsilon=0.5)

    def test_empty_categories_rejected(self):
        with pytest.raises(ValueError, match="categories"):
            FederatedHistogram(categories=(), epsilon=0.5)

    def test_empty_category_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            FederatedHistogram(categories=("", "x"), epsilon=0.5)

    def test_bad_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            FederatedHistogram(categories=("a",), epsilon=0.0)

    def test_histogram_seed_requires_explicit_insecure_test_mode(self):
        with pytest.raises(ValueError, match="allow_insecure_seed"):
            FederatedHistogram(categories=("a",), epsilon=0.5, seed=0)

    def test_empty_label(self):
        with pytest.raises(ValueError, match="label"):
            FederatedHistogram(categories=("a",), epsilon=0.5, label="")

    def test_bad_submit(self):
        hist = FederatedHistogram(
            categories=("a",),
            epsilon=0.5,
            seed=0,
            allow_insecure_seed=True,
        )
        with pytest.raises(ValueError, match="tenant_id"):
            hist.submit(tenant_id="", category="a")
        with pytest.raises(ValueError, match="count"):
            hist.submit(tenant_id="t", category="a", count=-1)

    def test_reset_clears_pending_counts_without_charging_accountant(self):
        acc = PrivacyAccountant(max_epsilon=5.0)
        hist = FederatedHistogram(
            categories=("a",),
            epsilon=0.5,
            accountant=acc,
            seed=0,
            allow_insecure_seed=True,
        )
        hist.submit(tenant_id="t", category="a")

        hist.reset()
        assert acc.cumulative_epsilon() == 0.0
        release = hist.release()

        assert release.raw_counts["a"] == 0
        assert acc.cumulative_epsilon() == pytest.approx(0.5)


class TestAggregatorRustSums:
    def test_rust_sum_i64_kernel_is_used_when_available(self, monkeypatch):
        monkeypatch.setattr(aggregator_mod, "_RUST_AGGREGATOR", True)
        called = {"count": 0}

        def _sum(values: list[int]) -> int:
            called["count"] += 1
            return sum(values)

        monkeypatch.setattr(aggregator_mod, "rust_sum_i64", _sum, raising=True)
        counter = FederatedCounter(epsilon=0.5, seed=0, allow_insecure_seed=True)
        counter.submit(tenant_id="t1", count=2)
        counter.submit(tenant_id="t2", count=3)
        release = counter.release()
        assert release.raw_sum == 5
        assert called["count"] >= 1

    def test_rust_sum_i64_type_error_falls_back(self, monkeypatch):
        monkeypatch.setattr(aggregator_mod, "_RUST_AGGREGATOR", True)
        monkeypatch.setattr(
            aggregator_mod,
            "rust_sum_i64",
            lambda _values: (_ for _ in ()).throw(TypeError("ffi signature mismatch")),
            raising=True,
        )
        hist = FederatedHistogram(
            categories=("spam", "safe"),
            epsilon=0.5,
            seed=0,
            allow_insecure_seed=True,
        )
        hist.submit(tenant_id="t1", category="spam", count=2)
        hist.submit(tenant_id="t2", category="safe", count=3)
        release = hist.release()
        assert release.submissions == 5


# --- FederatedSafetySignalAggregator ------------------------------


def _signal(*, tenant_id: str, decision: str = "halt", reason: str = "coherence"):
    event = SafetyEvent.from_policy_decision(
        hook_id="stream",
        hook_scope="streaming",
        policy_decision=decision,
        halt_reason=reason,
        tenant_safe_explanation="Tenant-safe halt summary.",
        tenant_id=tenant_id,
        observed_score=0.2,
        threshold=0.5,
        evidence_refs=("chunk:0",),
    )
    return director_safety_signal_from_event(
        event,
        producer_id="producer-a",
        framework="test",
    )


class TestFederatedSafetySignalAggregator:
    def test_releases_anonymised_noisy_signal_histogram_without_raw_defaults(self):
        accountant = PrivacyAccountant(max_epsilon=5.0)
        aggregator = FederatedSafetySignalAggregator(
            epsilon=0.9,
            accountant=accountant,
            min_tenants=2,
            seed=0,
            allow_insecure_seed=True,
        )
        aggregator.submit_signal(_signal(tenant_id="tenant-a", decision="halt"))
        aggregator.submit_signal(_signal(tenant_id="tenant-b", decision="warn"))

        release = aggregator.release()
        payload = release.to_dict()

        assert release.signal_count == 2
        assert release.distinct_tenants == 2
        assert release.raw_counts["decision:halt"] == 1
        assert release.raw_counts["decision:warn"] == 1
        assert "raw_counts" not in payload
        assert "tenant-a" not in str(payload)
        assert "tenant-b" not in str(payload)
        assert payload["privacy"]["payload_classification"] == "anonymous_dp_aggregate"
        assert payload["epsilon_spent"] == pytest.approx(0.9)
        assert accountant.cumulative_epsilon() == pytest.approx(0.9)

    def test_per_tenant_category_cap_bounds_histogram_sensitivity(self):
        aggregator = FederatedSafetySignalAggregator(
            epsilon=0.9,
            min_tenants=2,
            seed=0,
            allow_insecure_seed=True,
        )
        first = _signal(tenant_id="tenant-a", decision="halt")
        duplicate = _signal(tenant_id="tenant-a", decision="halt")
        other = _signal(tenant_id="tenant-b", decision="halt")

        assert aggregator.submit_signal(first) == ("decision:halt", "scope:streaming")
        assert aggregator.submit_signal(duplicate) == ()
        assert aggregator.submit_signal(other) == ("decision:halt", "scope:streaming")

        release = aggregator.release()

        assert release.raw_counts["decision:halt"] == 2
        assert release.raw_counts["scope:streaming"] == 2
        assert release.signal_count == 2

    def test_release_requires_minimum_distinct_tenants_without_charging_budget(self):
        accountant = PrivacyAccountant(max_epsilon=5.0)
        aggregator = FederatedSafetySignalAggregator(
            epsilon=0.9,
            accountant=accountant,
            min_tenants=2,
            seed=0,
            allow_insecure_seed=True,
        )
        aggregator.submit_signal(_signal(tenant_id="tenant-a", decision="halt"))

        with pytest.raises(ValueError, match="min_tenants"):
            aggregator.release()

        assert accountant.cumulative_epsilon() == 0.0

    def test_rejects_transport_payloads_that_include_raw_data(self):
        aggregator = FederatedSafetySignalAggregator(
            epsilon=0.9,
            min_tenants=1,
            seed=0,
            allow_insecure_seed=True,
        )
        payload = _signal(tenant_id="tenant-a", decision="halt").to_transport_dict()
        payload["privacy"]["raw_payload_included"] = True

        with pytest.raises(ValueError, match="raw payloads"):
            aggregator.submit_transport(payload)
