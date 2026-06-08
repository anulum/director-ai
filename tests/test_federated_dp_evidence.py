# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Federated-DP evidence tests
"""Multi-angle tests for the federated-DP evidence packet.

Covers the formal (ε, δ) bound (effective noise multiplier z = m/2, round
monotonicity, no-noise → unbounded, default-rounds resolution), the certified
poisoning bound (closed-form shift, validation, zero-attacker edge), the combined
evidence packet, and the poisoning simulation (noise cancellation, observed shift
strictly within and exactly at the certified bound, deterministic baseline).
"""

from __future__ import annotations

import math

import pytest

from director_ai.core.federated_dp import (
    FederatedCalibrationRound,
    FederatedDPEvidence,
    FederatedDPEvidencePacket,
    PoisoningBound,
    PoisoningSimulation,
)
from director_ai.core.federated_privacy.rdp_accountant import DPGuarantee


def _round(**kwargs) -> FederatedCalibrationRound:
    params = {
        "clip_norm": 0.1,
        "noise_multiplier": 1.0,
        "min_cohort": 3,
        "learning_rate": 1.0,
    }
    params.update(kwargs)
    return FederatedCalibrationRound(0.6, **params)


class TestEffectiveNoise:
    def test_effective_noise_multiplier_is_half(self):
        ev = FederatedDPEvidence(_round(noise_multiplier=1.0))
        assert ev.effective_noise_multiplier == 0.5

    def test_effective_noise_multiplier_scales(self):
        ev = FederatedDPEvidence(_round(noise_multiplier=3.0))
        assert ev.effective_noise_multiplier == 1.5


class TestEpsilonBound:
    def test_returns_guarantee(self):
        ev = FederatedDPEvidence(_round())
        guarantee = ev.epsilon_bound(delta=1e-5, rounds=10)
        assert isinstance(guarantee, DPGuarantee)
        assert guarantee.delta == 1e-5
        assert guarantee.epsilon > 0.0

    def test_more_rounds_increase_epsilon(self):
        ev = FederatedDPEvidence(_round(noise_multiplier=2.0))
        few = ev.epsilon_bound(delta=1e-5, rounds=1).epsilon
        many = ev.epsilon_bound(delta=1e-5, rounds=50).epsilon
        assert many > few

    def test_zero_rounds_is_zero_epsilon(self):
        ev = FederatedDPEvidence(_round())
        assert ev.epsilon_bound(delta=1e-5, rounds=0).epsilon == 0.0

    def test_no_noise_is_unbounded(self):
        ev = FederatedDPEvidence(_round(noise_multiplier=0.0))
        assert ev.epsilon_bound(delta=1e-5, rounds=5).epsilon == math.inf

    def test_rounds_default_to_rounds_applied(self):
        cal = _round()
        for _ in range(2):
            for i in range(3):
                cal.submit_update(tenant_id=f"t{i}", update=0.05)
            cal.aggregate()
        ev = FederatedDPEvidence(cal)
        explicit = ev.epsilon_bound(delta=1e-5, rounds=2).epsilon
        default = ev.epsilon_bound(delta=1e-5).epsilon
        assert default == explicit

    def test_negative_rounds_rejected(self):
        ev = FederatedDPEvidence(_round())
        with pytest.raises(ValueError, match="rounds must be non-negative"):
            ev.epsilon_bound(delta=1e-5, rounds=-1)

    def test_delta_validation_propagates(self):
        ev = FederatedDPEvidence(_round())
        with pytest.raises(ValueError, match="delta must be in"):
            ev.epsilon_bound(delta=0.0, rounds=5)


class TestPoisoningBound:
    def test_closed_form_shift(self):
        ev = FederatedDPEvidence(_round(clip_norm=0.1, learning_rate=1.0))
        bound = ev.poisoning_bound(num_malicious=2, cohort_size=10, rounds=10)
        # per_round = lr * 2 * f * C / n = 1 * 2 * 2 * 0.1 / 10 = 0.04
        assert bound.per_round_shift == pytest.approx(0.04)
        assert bound.total_shift == pytest.approx(0.4)
        assert bound.fraction_malicious == pytest.approx(0.2)

    def test_learning_rate_scales_shift(self):
        ev = FederatedDPEvidence(_round(learning_rate=0.5))
        bound = ev.poisoning_bound(num_malicious=1, cohort_size=4, rounds=1)
        assert bound.per_round_shift == pytest.approx(0.5 * 2 * 1 * 0.1 / 4)

    def test_zero_attackers_zero_shift(self):
        ev = FederatedDPEvidence(_round())
        bound = ev.poisoning_bound(num_malicious=0, cohort_size=5, rounds=10)
        assert bound.total_shift == 0.0
        assert bound.fraction_malicious == 0.0

    def test_cohort_size_must_be_positive(self):
        ev = FederatedDPEvidence(_round())
        with pytest.raises(ValueError, match="cohort_size must be >= 1"):
            ev.poisoning_bound(num_malicious=0, cohort_size=0)

    def test_num_malicious_non_negative(self):
        ev = FederatedDPEvidence(_round())
        with pytest.raises(ValueError, match="num_malicious must be non-negative"):
            ev.poisoning_bound(num_malicious=-1, cohort_size=5)

    def test_num_malicious_cannot_exceed_cohort(self):
        ev = FederatedDPEvidence(_round())
        with pytest.raises(ValueError, match="cannot exceed cohort_size"):
            ev.poisoning_bound(num_malicious=6, cohort_size=5)

    def test_to_dict_tenant_safe(self):
        ev = FederatedDPEvidence(_round())
        payload = ev.poisoning_bound(
            num_malicious=2, cohort_size=10, rounds=3
        ).to_dict()
        assert set(payload) == {
            "num_malicious",
            "cohort_size",
            "fraction_malicious",
            "clip_norm",
            "learning_rate",
            "rounds",
            "per_round_shift",
            "total_shift",
        }


class TestEvidencePacket:
    def test_combines_bounds(self):
        ev = FederatedDPEvidence(_round(noise_multiplier=2.0))
        packet = ev.evidence_packet(
            delta=1e-5, num_malicious=1, cohort_size=8, rounds=5
        )
        assert isinstance(packet, FederatedDPEvidencePacket)
        assert packet.rounds == 5
        assert packet.noise_multiplier == 2.0
        assert packet.effective_noise_multiplier == 1.0
        assert packet.epsilon == ev.epsilon_bound(delta=1e-5, rounds=5).epsilon
        assert isinstance(packet.poisoning, PoisoningBound)
        assert packet.delta == 1e-5

    def test_to_dict_nested_tenant_safe(self):
        ev = FederatedDPEvidence(_round())
        payload = ev.evidence_packet(
            delta=1e-5, num_malicious=1, cohort_size=8, rounds=3
        ).to_dict()
        assert set(payload) == {
            "rounds",
            "noise_multiplier",
            "effective_noise_multiplier",
            "epsilon",
            "delta",
            "rdp_order",
            "poisoning",
        }
        assert "total_shift" in payload["poisoning"]


class TestPoisoningSimulation:
    def test_observed_within_bound_moderate(self):
        # Honest = 0 (mid-range), attackers at +clip: observed < certified bound.
        ev = FederatedDPEvidence(_round(clip_norm=0.1, learning_rate=1.0))
        sim = ev.simulate_poisoning(
            num_malicious=2, cohort_size=10, honest_update=0.0, rounds=10, seed=1
        )
        assert isinstance(sim, PoisoningSimulation)
        assert sim.within_bound
        # observed = R*lr*f*(C - honest)/n = 10*1*2*0.1/10 = 0.2 < 0.4 bound.
        assert sim.observed_shift == pytest.approx(0.2)
        assert sim.observed_shift < sim.bound.total_shift

    def test_worst_case_hits_bound(self):
        # Honest at -clip, attackers at +clip: observed exactly the certified bound.
        ev = FederatedDPEvidence(_round(clip_norm=0.1, learning_rate=1.0))
        sim = ev.simulate_poisoning(
            num_malicious=2,
            cohort_size=10,
            honest_update=-0.1,
            attacker_update=0.1,
            rounds=10,
            seed=7,
        )
        assert sim.observed_shift == pytest.approx(sim.bound.total_shift)
        assert sim.within_bound

    def test_noise_cancels_so_zero_attackers_zero_shift(self):
        ev = FederatedDPEvidence(_round(noise_multiplier=5.0))
        sim = ev.simulate_poisoning(
            num_malicious=0, cohort_size=6, honest_update=0.05, rounds=8, seed=3
        )
        assert sim.observed_shift == pytest.approx(0.0)
        assert sim.baseline_value == pytest.approx(sim.attacked_value)

    def test_rounds_must_be_positive(self):
        ev = FederatedDPEvidence(_round())
        with pytest.raises(ValueError, match="rounds must be >= 1"):
            ev.simulate_poisoning(
                num_malicious=1, cohort_size=5, honest_update=0.0, rounds=0
            )

    def test_simulation_to_dict_tenant_safe(self):
        ev = FederatedDPEvidence(_round())
        payload = ev.simulate_poisoning(
            num_malicious=1, cohort_size=5, honest_update=0.0, rounds=3, seed=1
        ).to_dict()
        assert set(payload) == {
            "bound",
            "observed_shift",
            "baseline_value",
            "attacked_value",
            "within_bound",
        }
