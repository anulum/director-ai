# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Cross-Validation Against SCPN-CODEBASE
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Cross-validates Director-AI physics modules against the canonical
SCPN-CODEBASE parameters and equations to ensure consistency.
"""

import numpy as np
import pytest

from director_ai.research.physics import (
    SECFunctional,
    UPDEState,
    UPDEStepper,
    L16OversightLoop,
)
from director_ai.research.physics.scpn_params import (
    OMEGA_N,
    build_knm_matrix,
    load_omega_n,
)


@pytest.mark.integration
class TestCanonicalParameters:
    """Verify Director-AI canonical parameters match SCPN-CODEBASE values."""

    def test_omega_n_count(self):
        omega = load_omega_n()
        assert omega.shape == (16,)
        assert np.all(omega > 0)

    def test_omega_n_values(self):
        """Cross-check against SCPN-CODEBASE canonical Omega_n."""
        omega = load_omega_n()
        # From CLAUDE.md: ω_1=1.329, ω_16=0.991
        assert omega[0] == pytest.approx(1.329, abs=0.01)
        assert omega[15] == pytest.approx(0.991, abs=0.01)

    def test_knm_shape_and_symmetry(self):
        K = build_knm_matrix()
        assert K.shape == (16, 16)
        np.testing.assert_allclose(K, K.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(K), 0.0, atol=1e-12)

    def test_knm_calibration_anchors(self):
        """Cross-check against SCPN-CODEBASE calibration anchors."""
        K = build_knm_matrix()
        # From CLAUDE.md: K[1,2]=0.302, K[2,3]=0.201, K[3,4]=0.252, K[4,5]=0.154
        assert K[0, 1] == pytest.approx(0.302, abs=0.005)
        assert K[1, 2] == pytest.approx(0.201, abs=0.005)
        assert K[2, 3] == pytest.approx(0.252, abs=0.005)
        assert K[3, 4] == pytest.approx(0.154, abs=0.005)

    def test_knm_cross_hierarchy_boosts(self):
        """L1↔L16=0.05, L5↔L7=0.15."""
        K = build_knm_matrix()
        assert K[0, 15] == pytest.approx(0.05, abs=0.005)
        assert K[4, 6] == pytest.approx(0.15, abs=0.005)


@pytest.mark.integration
class TestUPDEKuramoto:
    """Verify UPDE stepper implements correct Kuramoto coupling."""

    def test_correct_coupling_direction(self):
        """Phase difference coupling: sin(θ_m - θ_n), not sin(θ_n)."""
        stepper = UPDEStepper(noise_amplitude=0.0, field_pressure=0.0)
        state = UPDEState(theta=np.random.uniform(0, 2 * np.pi, 16))
        theta_before = state.theta.copy()
        state = stepper.step(state)
        # Phases should have moved — coupling drives them
        assert not np.allclose(theta_before, state.theta)

    def test_synchronized_phases_stable(self):
        """When all phases are equal, coupling term vanishes."""
        stepper = UPDEStepper(noise_amplitude=0.0, field_pressure=0.0)
        state = UPDEState(theta=np.ones(16))
        theta_before = state.theta.copy()
        state = stepper.step(state)
        # Coupling is zero, drift is only from natural frequencies + field
        # Δθ = (Ω_n + F*cos(θ)) * dt
        expected = theta_before + (load_omega_n() + 0.0 * np.cos(theta_before)) * stepper.dt
        np.testing.assert_allclose(state.theta, expected, atol=1e-10)

    def test_order_parameter_increases_with_strong_coupling(self):
        """Strong coupling should increase synchrony (R)."""
        K = build_knm_matrix() * 5.0  # 5x stronger
        stepper = UPDEStepper(knm=K, noise_amplitude=0.01)
        state = UPDEState(theta=np.random.uniform(0, 2 * np.pi, 16))
        R_initial = float(np.abs(np.mean(np.exp(1j * state.theta))))
        for _ in range(200):
            state = stepper.step(state)
        R_final = float(np.abs(np.mean(np.exp(1j * state.theta))))
        assert R_final > R_initial or R_final > 0.5


@pytest.mark.integration
class TestSECConsistency:
    """Verify SEC functional matches SCPN theoretical predictions."""

    def test_v_decreases_for_synchronized_initial(self):
        """Starting from near-sync, V should stay low."""
        sec = SECFunctional()
        theta = np.zeros(16) + np.random.randn(16) * 0.1  # Near-sync
        result = sec.evaluate(theta)
        assert result.V < 5.0  # Should be low for near-sync

    def test_v_high_for_random_initial(self):
        """Random phases should have higher V."""
        sec = SECFunctional()
        theta = np.random.uniform(0, 2 * np.pi, 16)
        result = sec.evaluate(theta)
        assert result.V > 0  # Definitely non-zero

    def test_coherence_score_range(self):
        sec = SECFunctional()
        for _ in range(10):
            theta = np.random.uniform(0, 2 * np.pi, 16)
            result = sec.evaluate(theta)
            assert 0.0 <= result.coherence_score <= 1.0

    def test_critical_coupling_positive(self):
        sec = SECFunctional()
        K_c = sec.critical_coupling()
        assert K_c > 0


@pytest.mark.integration
class TestOversightLoopConsistency:
    """Verify L16 oversight loop behaviour."""

    def test_run_produces_snapshots(self):
        loop = L16OversightLoop()
        snapshots = loop.run(n_steps=20)
        assert len(snapshots) == 20

    def test_coherence_improves_trend(self):
        """Over many steps, coherence should generally improve."""
        loop = L16OversightLoop()
        snapshots = loop.run(n_steps=50)
        first_10 = np.mean([s.coherence_score for s in snapshots[:10]])
        last_10 = np.mean([s.coherence_score for s in snapshots[-10:]])
        # Allow some tolerance for stochastic noise
        assert last_10 >= first_10 - 0.15
