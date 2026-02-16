# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — SSGF Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import numpy as np
import pytest

from director_ai.research.physics import SSGFConfig, SSGFEngine, SSGFState


@pytest.mark.physics
class TestSSGFEngine:
    def test_instantiation(self):
        engine = SSGFEngine()
        assert engine.cfg.N == 16
        assert engine.theta.shape == (16,)
        assert engine.W.shape == (16, 16)

    def test_w_is_symmetric(self):
        engine = SSGFEngine()
        np.testing.assert_allclose(engine.W, engine.W.T, atol=1e-12)

    def test_w_is_non_negative(self):
        engine = SSGFEngine()
        assert np.all(engine.W >= 0)

    def test_w_zero_diagonal(self):
        engine = SSGFEngine()
        np.testing.assert_allclose(np.diag(engine.W), 0.0, atol=1e-12)

    def test_single_step(self):
        engine = SSGFEngine(config=SSGFConfig(n_micro=5))
        state = engine.step()
        assert isinstance(state, SSGFState)
        assert 0.0 <= state.R_global <= 1.0
        assert 0.0 <= state.p_h1 <= 1.0
        assert state.step == 1

    def test_run_convergence(self):
        cfg = SSGFConfig(N=8, n_micro=5, lr_z=0.005)
        engine = SSGFEngine(config=cfg)
        states = engine.run(n_steps=20)
        assert len(states) == 20
        # U_total should generally not explode
        u_values = [s.U_total for s in states]
        assert all(np.isfinite(u) for u in u_values)

    def test_w_stays_valid_after_run(self):
        cfg = SSGFConfig(N=8, n_micro=3)
        engine = SSGFEngine(config=cfg)
        engine.run(n_steps=10)
        # W must remain symmetric, non-negative, zero-diagonal
        np.testing.assert_allclose(engine.W, engine.W.T, atol=1e-12)
        assert np.all(engine.W >= 0)
        np.testing.assert_allclose(np.diag(engine.W), 0.0, atol=1e-12)

    def test_fiedler_value(self):
        engine = SSGFEngine()
        fiedler = engine.fiedler_value
        assert np.isfinite(fiedler)
        assert fiedler >= -1e-10  # Second eigenvalue of Laplacian ≥ 0

    def test_cost_history_tracked(self):
        cfg = SSGFConfig(N=8, n_micro=3)
        engine = SSGFEngine(config=cfg)
        engine.run(n_steps=5)
        history = engine.get_cost_history()
        assert len(history) == 5
        for costs in history:
            assert "U_total" in costs
            assert "c_micro" in costs
            assert "R_global" in costs

    def test_custom_config(self):
        cfg = SSGFConfig(
            N=4,
            n_micro=2,
            dt=0.005,
            lr_z=0.001,
            sigma_g=0.5,
        )
        engine = SSGFEngine(config=cfg)
        assert engine.theta.shape == (4,)
        state = engine.step()
        assert state.theta.shape == (4,)
