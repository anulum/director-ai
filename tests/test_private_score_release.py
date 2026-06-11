# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Private Score Release Tests

import pytest

from director_ai.core import DifferentialPrivacyScoreReleaser, PrivacyScoreRelease
from director_ai.core.federated_privacy import PrivacyAccountant


def test_zero_sensitivity_releases_score_without_noise_and_charges_budget():
    accountant = PrivacyAccountant(max_epsilon=1.0)
    releaser = DifferentialPrivacyScoreReleaser(
        epsilon=0.25,
        sensitivity=0.0,
        accountant=accountant,
    )

    release = releaser.release_score(0.42, label="public-dashboard")

    assert isinstance(release, PrivacyScoreRelease)
    assert release.released_score == 0.42
    assert release.raw_score_included is False
    assert release.epsilon_spent == 0.25
    assert accountant.cumulative_epsilon() == pytest.approx(0.25)


def test_laplace_release_is_clamped_and_reproducible_for_test_seed():
    first = DifferentialPrivacyScoreReleaser(
        epsilon=0.5,
        sensitivity=1.0,
        seed=123,
        allow_insecure_seed=True,
    )
    second = DifferentialPrivacyScoreReleaser(
        epsilon=0.5,
        sensitivity=1.0,
        seed=123,
        allow_insecure_seed=True,
    )

    release_a = first.release_score(0.95, label="score")
    release_b = second.release_score(0.95, label="score")

    assert 0.0 <= release_a.released_score <= 1.0
    assert release_a.released_score == release_b.released_score
    assert release_a.noise != 0.0


def test_budget_exhaustion_prevents_release_before_noise_is_returned():
    accountant = PrivacyAccountant(max_epsilon=0.4)
    releaser = DifferentialPrivacyScoreReleaser(
        epsilon=0.25,
        sensitivity=0.0,
        accountant=accountant,
    )

    releaser.release_score(0.5, label="q1")

    with pytest.raises(ValueError, match="epsilon"):
        releaser.release_score(0.5, label="q2")


def test_report_payload_excludes_raw_score_and_carries_privacy_metadata():
    releaser = DifferentialPrivacyScoreReleaser(epsilon=0.5, sensitivity=0.0)

    release = releaser.release_score(
        0.73,
        label="tenant-score",
        tenant_id="tenant-a",
        threshold=0.7,
    )
    payload = release.to_dict()

    assert payload["released_score"] == 0.73
    assert payload["raw_score"] is None
    assert payload["raw_score_included"] is False
    assert payload["approved_at_threshold"] is True
    assert payload["privacy"]["mechanism"] == "laplace"
    assert payload["privacy"]["epsilon"] == 0.5


def test_release_without_threshold_has_no_approval_decision():
    releaser = DifferentialPrivacyScoreReleaser(epsilon=0.5, sensitivity=0.0)

    release = releaser.release_score(0.3, label="analytics")

    assert release.threshold is None
    assert release.approved_at_threshold is None
    assert release.to_dict()["approved_at_threshold"] is None


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError, match="epsilon"):
        DifferentialPrivacyScoreReleaser(epsilon=0.0)
    with pytest.raises(ValueError, match="sensitivity"):
        DifferentialPrivacyScoreReleaser(epsilon=0.5, sensitivity=-1.0)

    releaser = DifferentialPrivacyScoreReleaser(epsilon=0.5)
    with pytest.raises(ValueError, match="score"):
        releaser.release_score(1.5, label="bad")
    with pytest.raises(ValueError, match="label"):
        releaser.release_score(0.5, label="")
    with pytest.raises(ValueError, match="threshold"):
        releaser.release_score(0.5, label="bad-threshold", threshold=float("nan"))
