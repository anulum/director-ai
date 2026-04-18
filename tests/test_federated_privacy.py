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

from director_ai.core.federated_privacy import (
    AccountantEntry,
    FederatedCounter,
    FederatedHistogram,
    GaussianMechanism,
    LaplaceMechanism,
    PrivacyAccountant,
    SecretShare,
    SecureAggregator,
    ShareError,
)
from director_ai.core.federated_privacy.secret_sharing import (
    DEFAULT_MODULUS,
    reconstruct,
    split,
    split_many,
)

# --- LaplaceMechanism ---------------------------------------------


class TestLaplace:
    def test_scale_is_sensitivity_over_epsilon(self):
        m = LaplaceMechanism(epsilon=0.5, sensitivity=2.0, seed=0)
        assert m.scale == pytest.approx(4.0)

    def test_zero_sensitivity_produces_zero_scale(self):
        m = LaplaceMechanism(epsilon=0.5, sensitivity=0.0, seed=0)
        assert m.scale == 0.0
        assert m.noise() == 0.0

    def test_bad_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            LaplaceMechanism(epsilon=0, sensitivity=1.0)

    def test_bad_sensitivity(self):
        with pytest.raises(ValueError, match="sensitivity"):
            LaplaceMechanism(epsilon=1.0, sensitivity=-0.5)

    def test_noise_has_mean_near_zero(self):
        m = LaplaceMechanism(epsilon=1.0, sensitivity=1.0, seed=42)
        samples = [m.noise() for _ in range(5_000)]
        assert abs(statistics.mean(samples)) < 0.1

    def test_noise_variance_matches_laplace(self):
        """Laplace(0, b) has variance 2 b^2. We tolerate 20% error
        on 5 000 samples."""
        m = LaplaceMechanism(epsilon=2.0, sensitivity=1.0, seed=7)
        samples = [m.noise() for _ in range(5_000)]
        expected_var = 2 * (m.scale**2)
        assert statistics.pvariance(samples) == pytest.approx(expected_var, rel=0.2)


# --- GaussianMechanism --------------------------------------------


class TestGaussian:
    def test_sigma_formula(self):
        m = GaussianMechanism(epsilon=0.5, delta=1e-5, sensitivity=1.0, seed=0)
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 0.5
        assert m.sigma == pytest.approx(expected)

    def test_zero_sensitivity_zero_sigma(self):
        m = GaussianMechanism(epsilon=0.5, delta=1e-5, sensitivity=0.0, seed=0)
        assert m.sigma == 0.0
        assert m.noise() == 0.0

    def test_epsilon_must_be_under_one(self):
        with pytest.raises(ValueError, match="epsilon"):
            GaussianMechanism(epsilon=1.1, delta=1e-5, sensitivity=1.0)

    def test_delta_must_be_unit_interval(self):
        with pytest.raises(ValueError, match="delta"):
            GaussianMechanism(epsilon=0.5, delta=0.0, sensitivity=1.0)


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


# --- SecretShare + SecureAggregator -------------------------------


class TestSecretSharing:
    def test_reconstruct_roundtrip(self):
        share = split(42, party_count=3, seed=0)
        assert reconstruct(share) == 42

    def test_share_has_correct_party_count(self):
        share = split(7, party_count=5, seed=1)
        assert share.party_count == 5

    def test_share_values_bounded(self):
        share = split(100, party_count=4, seed=2, modulus=1000)
        for v in share.values:
            assert 0 <= v < 1000

    def test_aggregator_sums_secrets(self):
        aggregator = SecureAggregator(party_count=3)
        s1 = split(10, party_count=3, seed=1)
        s2 = split(20, party_count=3, seed=2)
        s3 = split(15, party_count=3, seed=3)
        aggregator.submit(s1)
        aggregator.submit(s2)
        aggregator.submit(s3)
        assert aggregator.reconstruct() == 45
        assert aggregator.submissions == 3

    def test_aggregator_rejects_mismatched_party_count(self):
        aggregator = SecureAggregator(party_count=3)
        mismatched = split(10, party_count=4, seed=1)
        with pytest.raises(ShareError, match="parties"):
            aggregator.submit(mismatched)

    def test_aggregator_rejects_mismatched_modulus(self):
        aggregator = SecureAggregator(party_count=3, modulus=1_000_003)
        other_modulus = split(10, party_count=3, seed=1, modulus=999_983)
        with pytest.raises(ShareError, match="modulus"):
            aggregator.submit(other_modulus)

    def test_aggregator_no_submissions(self):
        aggregator = SecureAggregator(party_count=2)
        with pytest.raises(ShareError, match="no submissions"):
            aggregator.reconstruct()

    def test_aggregator_reset(self):
        aggregator = SecureAggregator(party_count=2)
        aggregator.submit(split(5, party_count=2, seed=0))
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

    def test_share_negative_rejected(self):
        with pytest.raises(ShareError, match="outside"):
            SecretShare(values=(1, -1), modulus=DEFAULT_MODULUS)

    def test_share_too_large_rejected(self):
        with pytest.raises(ShareError, match="outside"):
            SecretShare(values=(1, 1 << 200), modulus=DEFAULT_MODULUS)

    def test_split_many(self):
        shares = split_many([1, 2, 3], party_count=3, seed=0)
        assert len(shares) == 3
        for share, expected in zip(shares, [1, 2, 3], strict=False):
            assert reconstruct(share) == expected

    def test_split_many_empty(self):
        with pytest.raises(ShareError, match="secrets"):
            split_many([], party_count=3)

    def test_reproducible_with_seed(self):
        a = split(123, party_count=5, seed=777)
        b = split(123, party_count=5, seed=777)
        assert a.values == b.values


# --- FederatedCounter ---------------------------------------------


class TestFederatedCounter:
    def test_submits_and_releases(self):
        acc = PrivacyAccountant(max_epsilon=5.0)
        counter = FederatedCounter(epsilon=0.5, sensitivity=1.0, accountant=acc, seed=0)
        counter.submit(tenant_id="t1", count=3)
        counter.submit(tenant_id="t2", count=7)
        release = counter.release()
        assert release.raw_sum == 10
        assert release.epsilon_spent == pytest.approx(0.5)
        assert acc.cumulative_epsilon() == pytest.approx(0.5)

    def test_release_resets_state(self):
        counter = FederatedCounter(epsilon=0.5, seed=0)
        counter.submit(tenant_id="t1", count=1)
        counter.release()
        second = counter.release()
        assert second.raw_sum == 0

    def test_budget_guard(self):
        acc = PrivacyAccountant(max_epsilon=0.4)
        counter = FederatedCounter(epsilon=0.5, accountant=acc, seed=0)
        with pytest.raises(ValueError, match="epsilon"):
            counter.release()

    def test_bad_tenant_or_count(self):
        counter = FederatedCounter(epsilon=0.5, seed=0)
        with pytest.raises(ValueError, match="tenant_id"):
            counter.submit(tenant_id="", count=1)
        with pytest.raises(ValueError, match="count"):
            counter.submit(tenant_id="t", count=-1)

    def test_bad_label(self):
        with pytest.raises(ValueError, match="label"):
            FederatedCounter(epsilon=0.5, label="")

    def test_concurrent_submits(self):
        counter = FederatedCounter(epsilon=0.5, seed=0)

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
        hist = FederatedHistogram(categories=("a", "b"), epsilon=0.5, seed=0)
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

    def test_empty_label(self):
        with pytest.raises(ValueError, match="label"):
            FederatedHistogram(categories=("a",), epsilon=0.5, label="")

    def test_bad_submit(self):
        hist = FederatedHistogram(categories=("a",), epsilon=0.5, seed=0)
        with pytest.raises(ValueError, match="tenant_id"):
            hist.submit(tenant_id="", category="a")
        with pytest.raises(ValueError, match="count"):
            hist.submit(tenant_id="t", category="a", count=-1)
