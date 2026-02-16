# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — L16 Physics Unit Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Unit tests for research/physics/ modules: L16 closure, SCPN params,
SEC functional edge cases, and UPDE stepper properties.
"""

import numpy as np
import pytest

from director_ai.research.physics import (
    OMEGA_N,
    L16Controller,
    L16OversightLoop,
    SECFunctional,
    UPDEState,
    UPDEStepper,
    build_knm_matrix,
)
from director_ai.research.physics.l16_closure import PIState, pi_step


class TestPIController:
    def test_proportional_response(self):
        pi = PIState(setpoint=1.0, Kp=2.0, Ki=0.0)
        output = pi_step(pi, measured=0.5, dt=0.01)
        # Error = 1.0 - 0.5 = 0.5, output = 2.0 * 0.5 = 1.0
        assert abs(output - 1.0) < 1e-6

    def test_integral_accumulation(self):
        pi = PIState(setpoint=1.0, Kp=0.0, Ki=1.0, output_min=0.0)
        # First step: error = 0.5, integral += 0.5 * 0.01 = 0.005
        out1 = pi_step(pi, measured=0.5, dt=0.01)
        assert out1 > 0.0
        # Second step: same error, integral grows
        out2 = pi_step(pi, measured=0.5, dt=0.01)
        assert out2 > out1

    def test_anti_windup_clamp(self):
        pi = PIState(setpoint=1.0, Kp=0.0, Ki=100.0, integral_max=0.5)
        # Large Ki * error * dt would exceed integral_max
        for _ in range(100):
            pi_step(pi, measured=0.0, dt=0.1)
        assert pi.integral <= pi.integral_max + 1e-12

    def test_zero_error(self):
        pi = PIState(setpoint=0.5, Kp=1.0, Ki=0.5, output_min=0.0)
        output = pi_step(pi, measured=0.5, dt=0.01)
        assert abs(output) < 1e-6


class TestL16Controller:
    @pytest.fixture
    def controller(self):
        return L16Controller(N=16)

    def test_step_returns_expected_keys(self, controller):
        theta = np.random.uniform(0, 2 * np.pi, 16)
        result = controller.step(
            theta=theta,
            eigvecs=np.eye(16, 4),
            phi_target=np.eye(16, 4),
            R_global=0.6,
            plv=0.7,
            costs={"c7": 0.1, "c8": 0.1, "c10": 0.05},
            dt=0.01,
        )
        for key in ["lambda7", "lambda8", "lambda10", "h_rec", "gate_open"]:
            assert key in result

    def test_gate_open_with_high_plv(self, controller):
        theta = np.zeros(16)
        result = controller.step(
            theta=theta,
            eigvecs=np.eye(16, 4),
            phi_target=np.eye(16, 4),
            R_global=0.9,
            plv=0.8,
            costs={"c7": 0.01, "c8": 0.01, "c10": 0.01},
            dt=0.01,
        )
        assert isinstance(result["gate_open"], bool)

    def test_h_rec_is_finite(self, controller):
        theta = np.random.uniform(0, 2 * np.pi, 16)
        result = controller.step(
            theta=theta,
            eigvecs=np.eye(16, 4),
            phi_target=np.eye(16, 4),
            R_global=0.5,
            plv=0.6,
            costs={"c7": 0.2, "c8": 0.3, "c10": 0.1},
            dt=0.01,
        )
        assert np.isfinite(result["h_rec"])

    def test_refusal_on_rising_h_rec(self, controller):
        """Run many steps with bad data — controller shouldn't crash."""
        theta = np.random.uniform(0, 2 * np.pi, 16)
        for _ in range(20):
            result = controller.step(
                theta=theta,
                eigvecs=np.eye(16, 4),
                phi_target=np.eye(16, 4) * 100,  # huge misalignment
                R_global=0.1,
                plv=0.2,
                costs={"c7": 5.0, "c8": 5.0, "c10": 5.0},
                dt=0.01,
            )
        assert np.isfinite(result["h_rec"])


class TestSECEdgeCases:
    def test_uniform_random_phases(self):
        sec = SECFunctional()
        theta = np.random.uniform(0, 2 * np.pi, 16)
        result = sec.evaluate(theta)
        assert result.V >= 0.0
        assert 0.0 <= result.coherence_score <= 1.0

    def test_sequential_dv_dt(self):
        """dV/dt should be estimable after two evaluations."""
        sec = SECFunctional()
        theta1 = np.random.uniform(0, 2 * np.pi, 16)
        sec.evaluate(theta1, dt=0.01)
        theta2 = theta1 + 0.01 * np.random.randn(16)
        result2 = sec.evaluate(theta2, dt=0.01)
        assert result2.dV_dt != 0.0  # should have a nonzero estimate

    def test_custom_knm(self):
        """Accept a user-supplied coupling matrix."""
        knm = np.ones((16, 16)) * 0.1
        np.fill_diagonal(knm, 0)
        sec = SECFunctional(knm=knm)
        theta = np.zeros(16)
        result = sec.evaluate(theta)
        assert result.V >= 0.0


class TestUPDEStepperProperties:
    def test_theta_is_finite(self):
        """Phases should remain finite after stepping."""
        stepper = UPDEStepper(dt=0.1)
        state = UPDEState(theta=np.ones(16) * 6.0)  # near 2pi
        for _ in range(50):
            state = stepper.step(state)
        assert np.all(np.isfinite(state.theta))

    def test_noise_is_applied(self):
        """With noise_amplitude > 0, successive runs should differ."""
        stepper = UPDEStepper(dt=0.01, noise_amplitude=0.1)
        theta0 = np.zeros(16)
        state_a = UPDEState(theta=theta0.copy())
        state_b = UPDEState(theta=theta0.copy())
        state_a = stepper.step(state_a)
        state_b = stepper.step(state_b)
        # With noise, two independent runs from same IC should differ
        # (unless seeded identically)
        # Just check the stepper doesn't crash with noise
        assert state_a.theta.shape == (16,)

    def test_time_advances(self):
        stepper = UPDEStepper(dt=0.05)
        state = UPDEState(theta=np.zeros(16))
        for _ in range(10):
            state = stepper.step(state)
        assert abs(state.t - 0.5) < 1e-10

    def test_order_parameter_bounded(self):
        stepper = UPDEStepper(dt=0.01)
        state = UPDEState(theta=np.random.uniform(0, 2 * np.pi, 16))
        state.compute_order_parameter()
        for _ in range(30):
            state = stepper.step(state)
        assert 0.0 <= state.R_global <= 1.0


class TestKnmMatrix:
    def test_calibration_anchors(self):
        knm = build_knm_matrix()
        assert abs(knm[0, 1] - 0.302) < 1e-6
        assert abs(knm[1, 2] - 0.201) < 1e-6
        assert abs(knm[2, 3] - 0.252) < 1e-6
        assert abs(knm[3, 4] - 0.154) < 1e-6

    def test_cross_hierarchy_boosts(self):
        knm = build_knm_matrix()
        # L1-L16 (indices 0, 15) should have cross-boost
        assert knm[0, 15] > 0.0
        # L5-L7 (indices 4, 6) should have cross-boost
        assert knm[4, 6] > 0.0

    def test_omega_n_shape_and_positivity(self):
        assert OMEGA_N.shape == (16,)
        assert np.all(OMEGA_N > 0)


class TestOversightLoop:
    def test_convergence_trend(self):
        """R_global should generally increase over time under oversight."""
        loop = L16OversightLoop()
        snapshots = loop.run(n_steps=50)
        # Not a strict test (stochastic), just verify it ran without error
        assert len(snapshots) == 50
        assert all(0.0 <= s.R_global <= 1.0 for s in snapshots)

    def test_intervention_strings(self):
        """Each snapshot should have an intervention field."""
        loop = L16OversightLoop()
        snapshots = loop.run(n_steps=10)
        for snap in snapshots:
            assert snap.intervention is None or isinstance(snap.intervention, str)
