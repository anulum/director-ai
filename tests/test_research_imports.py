# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Research Module Import & Smoke Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Tests that all research modules import correctly and basic smoke tests
for the physics and consciousness subpackages.
"""

import numpy as np
import pytest


class TestPhysicsImports:
    def test_scpn_params(self):
        from director_ai.research.physics import (
            N_LAYERS,
            OMEGA_N,
            LAYER_NAMES,
            load_omega_n,
            build_knm_matrix,
        )

        assert N_LAYERS == 16
        assert OMEGA_N.shape == (16,)
        assert len(LAYER_NAMES) == 16
        omega = load_omega_n()
        assert omega.shape == (16,)
        knm = build_knm_matrix()
        assert knm.shape == (16, 16)

    def test_knm_properties(self):
        from director_ai.research.physics import build_knm_matrix

        knm = build_knm_matrix()
        # Symmetric
        np.testing.assert_allclose(knm, knm.T, atol=1e-12)
        # Zero diagonal
        np.testing.assert_allclose(np.diag(knm), 0.0)
        # Non-negative
        assert np.all(knm >= 0)
        # Calibration anchor check
        assert abs(knm[0, 1] - 0.302) < 1e-6

    def test_sec_functional(self):
        from director_ai.research.physics import SECFunctional, SECResult

        sec = SECFunctional()
        theta = np.random.uniform(0, 2 * np.pi, 16)
        result = sec.evaluate(theta)
        assert isinstance(result, SECResult)
        assert result.V >= 0.0
        assert 0.0 <= result.V_normalised <= 1.0
        assert 0.0 <= result.coherence_score <= 1.0
        assert 0.0 <= result.R_global <= 1.0

    def test_sec_stability_synchronized(self):
        from director_ai.research.physics import SECFunctional

        sec = SECFunctional()
        # Perfectly synchronised → low V
        theta_sync = np.zeros(16)
        result = sec.evaluate(theta_sync)
        assert result.V < 1.0  # low Lyapunov value for synchronised state
        assert result.coherence_score > 0.5

    def test_sec_critical_coupling(self):
        from director_ai.research.physics import SECFunctional

        sec = SECFunctional()
        Kc = sec.critical_coupling()
        assert Kc > 0.0

    def test_upde_stepper(self):
        from director_ai.research.physics import UPDEState, UPDEStepper

        stepper = UPDEStepper(dt=0.01)
        state = UPDEState(theta=np.random.uniform(0, 2 * np.pi, 16))
        state.compute_order_parameter()
        new_state = stepper.step(state)
        assert new_state.theta.shape == (16,)
        assert new_state.t > state.t
        assert 0.0 <= new_state.R_global <= 1.0

    def test_oversight_loop(self):
        from director_ai.research.physics import L16OversightLoop

        loop = L16OversightLoop()
        snapshots = loop.run(n_steps=20)
        assert len(snapshots) == 20
        for snap in snapshots:
            assert 0.0 <= snap.R_global <= 1.0
            assert isinstance(snap.is_stable, bool)

    def test_l16_controller(self):
        from director_ai.research.physics import L16Controller

        ctrl = L16Controller(N=16)
        theta = np.random.uniform(0, 2 * np.pi, 16)
        eigvecs = np.eye(16, 4)
        phi_target = np.eye(16, 4)
        result = ctrl.step(
            theta=theta,
            eigvecs=eigvecs,
            phi_target=phi_target,
            R_global=0.5,
            plv=0.7,
            costs={"c7": 0.1, "c8": 0.2, "c10": 0.05},
            dt=0.01,
        )
        assert "lambda7" in result
        assert "h_rec" in result
        assert isinstance(result["gate_open"], bool)

    def test_pi_controller(self):
        from director_ai.research.physics import PIState, pi_step

        pi = PIState(setpoint=0.5, Kp=1.0, Ki=0.1)
        output = pi_step(pi, measured=0.3, dt=0.01)
        assert output > 0.0


class TestConsciousnessImports:
    def test_tcbo_observer(self):
        from director_ai.research.consciousness import TCBOObserver, TCBOConfig

        cfg = TCBOConfig(window_size=20, embed_dim=3, tau_delay=1)
        obs = TCBOObserver(N=8, config=cfg)
        # Feed some synthetic data
        for _ in range(30):
            theta = np.random.uniform(0, 2 * np.pi, 8)
            obs.push_and_compute(theta, force=True)
        assert 0.0 <= obs.p_h1 <= 1.0
        state = obs.get_state()
        assert "p_h1" in state
        assert "is_conscious" in state

    def test_tcbo_controller(self):
        from director_ai.research.consciousness import TCBOController

        ctrl = TCBOController()
        kappa = ctrl.step(p_h1=0.5, current_kappa=0.3, dt=0.01)
        assert kappa >= 0.0
        assert isinstance(ctrl.is_gate_open(0.5), bool)
        assert ctrl.is_gate_open(0.8) is True
        assert ctrl.is_gate_open(0.5) is False

    def test_pgbo_engine(self):
        from director_ai.research.consciousness import PGBOEngine

        engine = PGBOEngine(N=16)
        theta1 = np.random.uniform(0, 2 * np.pi, 16)
        theta2 = theta1 + 0.1 * np.random.randn(16)
        engine.compute(theta1, dt=0.01)
        u, h = engine.compute(theta2, dt=0.01)
        assert u.shape == (16,)
        assert h.shape == (16, 16)
        # h_munu should be symmetric
        np.testing.assert_allclose(h, h.T, atol=1e-12)
        # h_munu should be PSD
        eigvals = np.linalg.eigvalsh(h)
        assert np.all(eigvals >= -1e-12)

    def test_pgbo_standalone(self):
        from director_ai.research.consciousness import phase_geometry_bridge

        D = 8
        dphi = np.random.randn(D)
        A = np.zeros(D)
        u, h = phase_geometry_bridge(dphi, A, alpha=0.5, kappa=0.3)
        assert u.shape == (D,)
        assert h.shape == (D, D)
        np.testing.assert_allclose(h, h.T, atol=1e-12)

    def test_pgbo_traceless(self):
        from director_ai.research.consciousness import phase_geometry_bridge

        D = 4
        dphi = np.random.randn(D)
        A = np.zeros(D)
        g0 = np.eye(D)
        g0_inv = np.eye(D)
        u, h = phase_geometry_bridge(
            dphi,
            A,
            alpha=0.5,
            kappa=0.3,
            g0=g0,
            g0_inv=g0_inv,
            traceless=True,
        )
        assert abs(np.trace(h)) < 1e-10

    def test_benchmarks_run(self):
        from director_ai.research.consciousness import run_all_benchmarks

        results = run_all_benchmarks(N=8)
        assert len(results) == 4
        for r in results:
            assert hasattr(r, "name")
            assert hasattr(r, "passed")
            assert isinstance(r.passed, bool)


class TestResearchTopLevelImports:
    def test_research_package(self):
        from director_ai.research import (
            ConsiliumAgent,
            EthicalFunctional,
            SystemState,
            SECFunctional,
            TCBOObserver,
            PGBOEngine,
            L16Controller,
        )

        assert ConsiliumAgent is not None
        assert EthicalFunctional is not None
        assert SECFunctional is not None
        assert TCBOObserver is not None
        assert PGBOEngine is not None
        assert L16Controller is not None

    def test_top_level_optional_imports(self):
        """Research classes should be available at director_ai level."""
        import director_ai

        assert hasattr(director_ai, "ConsiliumAgent")
        assert hasattr(director_ai, "EthicalFunctional")
