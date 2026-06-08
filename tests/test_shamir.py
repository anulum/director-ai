# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Shamir threshold secret-sharing tests

"""Multi-angle tests for Shamir t-of-n secret sharing and secure summation.

Covers split/reconstruct round-trips, threshold sufficiency and dropout
tolerance, the additive-homomorphism secure sum, share validation, and the
ProductionGuard secure-aggregator wiring.
"""

from __future__ import annotations

import pytest

from director_ai.core.federated_privacy import (
    ShamirShare,
    ShareError,
    shamir_reconstruct,
    shamir_split,
    shamir_sum_shares,
)


def _split(secret, *, n=5, t=3, seed=0):
    return shamir_split(
        secret, party_count=n, threshold=t, seed=seed, allow_insecure_seed=True
    )


class TestSplitReconstruct:
    def test_round_trip_any_threshold_subset(self):
        shares = _split(987654, n=5, t=3)
        assert shamir_reconstruct(list(shares[:3])) == 987654
        assert shamir_reconstruct([shares[0], shares[2], shares[4]]) == 987654

    def test_dropout_tolerance(self):
        # 3-of-5: lose two parties, still reconstruct.
        shares = _split(42, n=5, t=3)
        survivors = [shares[1], shares[3], shares[4]]
        assert shamir_reconstruct(survivors) == 42

    def test_more_than_threshold_ok(self):
        shares = _split(7, n=4, t=2)
        assert shamir_reconstruct(list(shares)) == 7

    def test_threshold_one_is_constant(self):
        shares = _split(55, n=3, t=1)
        assert all(s.y == 55 for s in shares)
        assert shamir_reconstruct([shares[0]]) == 55

    def test_secret_reduced_mod_modulus(self):
        shares = _split(0, n=3, t=2)
        assert shamir_reconstruct(list(shares[:2])) == 0


class TestSplitValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"party_count": 1, "threshold": 1}, "party_count"),
            ({"party_count": 3, "threshold": 0}, "threshold"),
            ({"party_count": 3, "threshold": 4}, "threshold"),
            ({"party_count": 3, "threshold": 2, "modulus": 0}, "modulus"),
        ],
    )
    def test_invalid_split_params(self, kwargs, match):
        with pytest.raises(ShareError, match=match):
            shamir_split(1, **kwargs)

    def test_seed_requires_opt_in(self):
        with pytest.raises(ShareError, match="allow_insecure_seed"):
            shamir_split(1, party_count=3, threshold=2, seed=5)


class TestShareValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"modulus": 0}, "modulus"),
            ({"x": 0}, "x must be positive"),
            ({"y": -1}, "outside"),
            ({"threshold": 0}, "threshold"),
        ],
    )
    def test_invalid_share(self, kwargs, match):
        base = {"x": 1, "y": 1, "threshold": 2}
        base.update(kwargs)
        with pytest.raises(ShareError, match=match):
            ShamirShare(**base)


class TestReconstructValidation:
    def test_empty_rejected(self):
        with pytest.raises(ShareError, match="at least one share"):
            shamir_reconstruct([])

    def test_insufficient_shares_rejected(self):
        shares = _split(1, n=4, t=3)
        with pytest.raises(ShareError, match="need at least 3"):
            shamir_reconstruct(list(shares[:2]))

    def test_mismatched_modulus_rejected(self):
        a = ShamirShare(x=1, y=1, threshold=2)
        b = ShamirShare(x=2, y=1, threshold=2, modulus=7)
        with pytest.raises(ShareError, match="modulus"):
            shamir_reconstruct([a, b])

    def test_mismatched_threshold_rejected(self):
        a = ShamirShare(x=1, y=1, threshold=2)
        b = ShamirShare(x=2, y=1, threshold=3)
        with pytest.raises(ShareError, match="threshold"):
            shamir_reconstruct([a, b])

    def test_duplicate_x_rejected(self):
        a = ShamirShare(x=1, y=1, threshold=2)
        b = ShamirShare(x=1, y=2, threshold=2)
        with pytest.raises(ShareError, match="distinct"):
            shamir_reconstruct([a, b])


class TestSecureSum:
    def test_homomorphic_secure_sum(self):
        secrets = [100, 250, 77, 3]
        groups = [_split(s, n=5, t=3, seed=i) for i, s in enumerate(secrets)]
        summed = shamir_sum_shares(groups)
        # Any threshold subset of the summed shares reconstructs the total.
        assert shamir_reconstruct(list(summed[:3])) == sum(secrets)
        assert shamir_reconstruct([summed[1], summed[3], summed[4]]) == sum(secrets)

    def test_empty_rejected(self):
        with pytest.raises(ShareError, match="at least one secret"):
            shamir_sum_shares([])

    def test_mismatched_party_count_rejected(self):
        g1 = _split(1, n=4, t=2)
        g2 = _split(2, n=3, t=2)
        with pytest.raises(ShareError, match="same parties"):
            shamir_sum_shares([g1, g2])

    def test_misaligned_x_rejected(self):
        a = (
            ShamirShare(x=1, y=1, threshold=2),
            ShamirShare(x=2, y=2, threshold=2),
        )
        b = (
            ShamirShare(x=2, y=3, threshold=2),  # swapped x order
            ShamirShare(x=1, y=4, threshold=2),
        )
        with pytest.raises(ShareError, match="aligned"):
            shamir_sum_shares([a, b])


class TestGuardWiring:
    def test_production_guard_secure_aggregator(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.core.federated_privacy import SecretShare, SecureAggregator
        from director_ai.core.federated_privacy.secret_sharing import split
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        agg = guard.secure_aggregator(party_count=3)
        assert isinstance(agg, SecureAggregator)
        for value in (10, 20, 12):
            agg.submit(
                split(value, party_count=3, seed=value, allow_insecure_seed=True)
            )
        assert agg.reconstruct() == 42
        assert isinstance(
            split(1, party_count=3, seed=1, allow_insecure_seed=True), SecretShare
        )
