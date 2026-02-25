# ─────────────────────────────────────────────────────────────────────
# Phase 3 Hardening Tests (H28-H45)
# ─────────────────────────────────────────────────────────────────────
"""
Tests for Phase 3 hardening fixes:
  H28  ROB-5: NLI assert → RuntimeError
  H29  CON-1: batch asyncio.get_running_loop()
  H30  ROB-1: batch coherence None guard
  H31  ROB-3: l16_closure eigvecs ndim guard
  H32  ROB-4: ssgf_cycle spectral bridge fallback
  H33  ROB-6: gpu_upde NaN guard
  H34  SEC-3: actor response.text truncation
  H35  SEC-5: config _coerce error message
  H36  API-2: config server_port/workers validation
  H37  RES-3: config from_yaml UTF-8
  H38  ROB-7: pgbo singularity check
  H39  API-1: cli --port safety
  H40  ROB-9: consilium vestigial history removed
  H41  ROB-10: SSGF history cap
  H42  CON-2: scorer history thread lock
  H43  TYP-4: server type-ignore removed
  H44  ROB-8: scorer setLevel removed
"""

import json
import os
import tempfile
import threading

import numpy as np
import pytest

# ── H28: NLI assert → RuntimeError ──────────────────────────────────


class TestH28NLIAssert:
    """NLI scorer should raise RuntimeError (not AssertionError) when model is None."""

    def test_nli_scorer_no_model_raises_runtime_error(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        # _model_score should raise RuntimeError if called when model isn't loaded
        with pytest.raises(RuntimeError, match="NLI model not loaded"):
            scorer._model_score("premise", "hypothesis")

    def test_nli_scorer_heuristic_fallback(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        # score() should fall back to heuristic, not crash
        result = scorer.score("The sky is blue", "consistent with reality")
        assert 0.0 <= result <= 1.0

    def test_nli_scorer_batch(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        results = scorer.score_batch(
            [
                ("a", "consistent with reality"),
                ("b", "opposite is true"),
            ]
        )
        assert len(results) == 2
        assert results[0] < results[1]  # consistent < contradiction


# ── H29: batch asyncio.get_running_loop ──────────────────────────────


class TestH29AsyncLoop:
    """Batch async should use get_running_loop (not deprecated get_event_loop)."""

    def test_process_batch_async_uses_running_loop(self):
        import inspect

        from director_ai.core.batch import BatchProcessor

        source = inspect.getsource(BatchProcessor.process_batch_async)
        assert "get_running_loop" in source
        assert "get_event_loop" not in source


# ── H30: batch coherence None guard ──────────────────────────────────


class TestH30CoherenceNoneGuard:
    """Batch _process_one should not crash when coherence is None."""

    def test_process_one_none_coherence(self):
        from unittest.mock import MagicMock

        from director_ai.core.batch import BatchProcessor
        from director_ai.core.types import ReviewResult

        mock_backend = MagicMock()
        mock_backend.process.return_value = ReviewResult(
            output="test", halted=True, candidates_evaluated=1, coherence=None
        )
        proc = BatchProcessor(mock_backend)
        result = proc._process_one(0, "test")
        assert result.halted is True
        assert result.coherence is None


# ── H31: l16_closure eigvecs ndim guard ──────────────────────────────


class TestH31EigvecsNdim:
    """L16 compute_h_rec should handle 1-D eigvecs gracefully."""

    def test_1d_eigvecs_fallback(self):
        from director_ai.research.physics.l16_closure import L16Controller

        ctrl = L16Controller(N=4)
        theta = np.random.uniform(0, 2 * np.pi, 4)
        eigvecs_1d = np.array([1.0, 2.0, 3.0, 4.0])
        phi_1d = np.array([1.0, 0.0, 0.0, 0.0])

        h = ctrl.compute_h_rec(theta, eigvecs_1d, phi_1d, R_global=0.5)
        assert np.isfinite(h)
        assert h >= 0.0

    def test_2d_eigvecs_normal(self):
        from director_ai.research.physics.l16_closure import L16Controller

        ctrl = L16Controller(N=4)
        theta = np.random.uniform(0, 2 * np.pi, 4)
        eigvecs = np.eye(4, 4)
        phi_target = np.eye(4, 4)

        h = ctrl.compute_h_rec(theta, eigvecs, phi_target, R_global=0.8)
        assert np.isfinite(h)
        assert h >= 0.0


# ── H32: SSGF spectral bridge fallback ──────────────────────────────


class TestH32SpectralBridgeFallback:
    """SSGF step should not crash on degenerate W matrices."""

    def test_ssgf_step_with_zero_w(self):
        from director_ai.research.physics.ssgf_cycle import SSGFConfig, SSGFEngine

        cfg = SSGFConfig(N=4, n_micro=2)
        engine = SSGFEngine(config=cfg)
        # Force W to zero (degenerate)
        engine.W[:] = 0.0
        # Should not raise
        state = engine.step()
        assert np.isfinite(state.U_total)


# ── H33: GPU UPDE NaN guard ─────────────────────────────────────────


class TestH33GpuUpdeNanGuard:
    """TorchUPDEStepper._step_numpy should reject NaN/Inf input."""

    def test_nan_input_raises(self):
        from director_ai.research.physics.gpu_upde import (
            TorchUPDEConfig,
            TorchUPDEStepper,
        )
        from director_ai.research.physics.l16_mechanistic import UPDEState

        stepper = TorchUPDEStepper(config=TorchUPDEConfig(device="cpu"))
        theta = np.full(16, np.nan)
        state = UPDEState(theta=theta)

        with pytest.raises(ValueError, match="NaN or Inf"):
            stepper._step_numpy(state)

    def test_inf_input_raises(self):
        from director_ai.research.physics.gpu_upde import (
            TorchUPDEConfig,
            TorchUPDEStepper,
        )
        from director_ai.research.physics.l16_mechanistic import UPDEState

        stepper = TorchUPDEStepper(config=TorchUPDEConfig(device="cpu"))
        theta = np.full(16, np.inf)
        state = UPDEState(theta=theta)

        with pytest.raises(ValueError, match="NaN or Inf"):
            stepper._step_numpy(state)

    def test_valid_input_passes(self):
        from director_ai.research.physics.gpu_upde import (
            TorchUPDEConfig,
            TorchUPDEStepper,
        )
        from director_ai.research.physics.l16_mechanistic import UPDEState

        stepper = TorchUPDEStepper(config=TorchUPDEConfig(device="cpu"))
        theta = np.random.uniform(0, 2 * np.pi, 16)
        state = UPDEState(theta=theta)

        new_state = stepper._step_numpy(state)
        assert np.all(np.isfinite(new_state.theta))


# ── H34: actor response.text truncation ─────────────────────────────


class TestH34ResponseTruncation:
    """LLMGenerator error log should truncate response.text to 500 chars."""

    def test_log_truncation_in_source(self):
        import inspect

        from director_ai.core.actor import LLMGenerator

        source = inspect.getsource(LLMGenerator.generate_candidates)
        assert "response.text[:500]" in source


# ── H35: config _coerce error message ───────────────────────────────


class TestH35CoerceError:
    """_coerce ValueError should name the offending env var."""

    def test_invalid_env_var_reports_key(self):
        from director_ai.core.config import DirectorConfig

        env = {"DIRECTOR_COHERENCE_THRESHOLD": "not_a_float"}
        original = os.environ.copy()
        try:
            os.environ.update(env)
            with pytest.raises(ValueError, match="DIRECTOR_COHERENCE_THRESHOLD"):
                DirectorConfig.from_env()
        finally:
            os.environ.clear()
            os.environ.update(original)


# ── H36: config server_port / server_workers validation ──────────────


class TestH36ServerValidation:
    """DirectorConfig should reject invalid server_port and server_workers."""

    def test_port_zero_rejected(self):
        from director_ai.core.config import DirectorConfig

        with pytest.raises(ValueError, match="server_port"):
            DirectorConfig(server_port=0)

    def test_port_65536_rejected(self):
        from director_ai.core.config import DirectorConfig

        with pytest.raises(ValueError, match="server_port"):
            DirectorConfig(server_port=65536)

    def test_valid_port_accepted(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(server_port=8080)
        assert cfg.server_port == 8080

    def test_workers_zero_rejected(self):
        from director_ai.core.config import DirectorConfig

        with pytest.raises(ValueError, match="server_workers"):
            DirectorConfig(server_workers=0)

    def test_workers_positive_accepted(self):
        from director_ai.core.config import DirectorConfig

        cfg = DirectorConfig(server_workers=4)
        assert cfg.server_workers == 4


# ── H37: config from_yaml UTF-8 ─────────────────────────────────────


class TestH37YamlUtf8:
    """from_yaml should open files with encoding='utf-8'."""

    def test_yaml_utf8_encoding_in_source(self):
        import inspect

        from director_ai.core.config import DirectorConfig

        source = inspect.getsource(DirectorConfig.from_yaml)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source

    def test_yaml_with_unicode(self):
        from director_ai.core.config import DirectorConfig

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump({"profile": "default", "log_level": "DEBUG"}, f)
            f.flush()
            path = f.name

        try:
            cfg = DirectorConfig.from_yaml(path)
            assert cfg.log_level == "DEBUG"
        finally:
            os.unlink(path)


# ── H38: PGBO singularity check ─────────────────────────────────────


class TestH38PGBOSingularity:
    """PGBOEngine.set_background_metric should reject singular matrices."""

    def test_singular_metric_raises(self):
        from director_ai.research.consciousness.pgbo import PGBOEngine

        engine = PGBOEngine(N=4)
        singular = np.zeros((4, 4))  # All zeros → singular
        with pytest.raises(ValueError, match="singular"):
            engine.set_background_metric(singular)

    def test_valid_metric_accepted(self):
        from director_ai.research.consciousness.pgbo import PGBOEngine

        engine = PGBOEngine(N=4)
        valid = np.eye(4)
        engine.set_background_metric(valid)
        np.testing.assert_array_equal(engine.g0, valid)


# ── H39: CLI --port safety ──────────────────────────────────────────


class TestH39CLIPort:
    """CLI --port should handle non-integer gracefully."""

    def test_cli_port_in_source(self):
        import inspect

        from director_ai.cli import _cmd_serve

        source = inspect.getsource(_cmd_serve)
        # Should have try/except around int() parsing
        assert "ValueError" in source

    def test_cli_batch_utf8(self):
        import inspect

        from director_ai.cli import _cmd_batch

        source = inspect.getsource(_cmd_batch)
        assert 'encoding="utf-8"' in source or "encoding='utf-8'" in source


# ── H40: consilium vestigial history removed ─────────────────────────


class TestH40ConsiliumHistory:
    """ConsiliumAgent should not have vestigial self.history attribute."""

    def test_no_history_attribute(self):
        from director_ai.research.consilium.director_core import ConsiliumAgent

        agent = ConsiliumAgent()
        assert not hasattr(agent, "history")


# ── H41: SSGF history cap ───────────────────────────────────────────


class TestH41SSGFHistoryCap:
    """SSGF cost/state history should be capped to prevent unbounded growth."""

    def test_history_capped_at_500(self):
        from director_ai.research.physics.ssgf_cycle import SSGFConfig, SSGFEngine

        cfg = SSGFConfig(N=4, n_micro=1)
        engine = SSGFEngine(config=cfg)
        assert engine._MAX_HISTORY == 500

    def test_history_does_not_exceed_cap(self):
        from director_ai.research.physics.ssgf_cycle import SSGFConfig, SSGFEngine

        cfg = SSGFConfig(N=4, n_micro=1)
        engine = SSGFEngine(config=cfg)
        engine._MAX_HISTORY = 5  # Reduce for fast test

        for _ in range(10):
            engine.step()

        assert len(engine._cost_history) <= 5
        assert len(engine._state_history) <= 5


# ── H42: scorer history thread lock ─────────────────────────────────


class TestH42ScorerThreadLock:
    """CoherenceScorer should protect history mutations with a lock."""

    def test_scorer_has_lock(self):
        from director_ai.core.scorer import CoherenceScorer

        s = CoherenceScorer(use_nli=False)
        assert hasattr(s, "_history_lock")
        assert isinstance(s._history_lock, type(threading.Lock()))

    def test_concurrent_reviews(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        errors = []

        def review_many():
            try:
                for i in range(50):
                    scorer.review(f"prompt {i}", "consistent with reality")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=review_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(scorer.history) <= scorer.window


# ── H44: scorer setLevel removed ────────────────────────────────────


class TestH44ScorerSetLevel:
    """CoherenceScorer should not call setLevel on its logger."""

    def test_no_set_level_in_init(self):
        import inspect

        from director_ai.core.scorer import CoherenceScorer

        source = inspect.getsource(CoherenceScorer.__init__)
        assert "setLevel" not in source


# ── L16 Controller comprehensive tests ──────────────────────────────


class TestL16Controller:
    """Dedicated L16 closure tests for coverage gap."""

    def test_pi_step_bounds(self):
        from director_ai.research.physics.l16_closure import PIState, pi_step

        pi = PIState(setpoint=1.0, Kp=1.0, Ki=0.1)
        out = pi_step(pi, 0.5, 0.01)
        assert pi.output_min <= out <= pi.output_max

    def test_plv_gate_closed_initially(self):
        from director_ai.research.physics.l16_closure import L16Controller

        ctrl = L16Controller()
        assert not ctrl.plv_gate_open()

    def test_plv_gate_opens(self):
        from director_ai.research.physics.l16_closure import L16Controller

        ctrl = L16Controller(plv_threshold=0.5)
        for _ in range(10):
            ctrl.update_plv(0.8)
        assert ctrl.plv_gate_open()

    def test_refusal_activates_on_rising_h_rec(self):
        from director_ai.research.physics.l16_closure import L16Controller

        ctrl = L16Controller(h_rec_window=3)
        ctrl.state.H_rec_history = [1.0, 2.0, 3.0, 4.0]
        assert ctrl.check_refusal()
        assert ctrl.state.refusal_active

    def test_refusal_deactivates_on_falling_h_rec(self):
        from director_ai.research.physics.l16_closure import L16Controller

        ctrl = L16Controller(h_rec_window=3)
        ctrl.state.H_rec_history = [4.0, 3.0, 2.0, 1.0]
        assert not ctrl.check_refusal()
        assert not ctrl.state.refusal_active

    def test_full_step(self):
        from director_ai.research.physics.l16_closure import L16Controller

        ctrl = L16Controller(N=4)
        theta = np.random.uniform(0, 2 * np.pi, 4)
        eigvecs = np.eye(4, 4)
        phi_target = np.eye(4, 4)
        costs = {"c7": 0.1, "c8": 0.2, "c10": 0.05, "p_h1": 0.5, "h_frob": 0.1}

        result = ctrl.step(
            theta, eigvecs, phi_target, R_global=0.6, plv=0.7, costs=costs, dt=0.01
        )
        assert "lambda7" in result
        assert "h_rec" in result
        assert "gate_open" in result


# ── PGBO comprehensive tests ────────────────────────────────────────


class TestPGBOEngine:
    """Dedicated PGBO engine tests for coverage gap."""

    def test_compute_returns_symmetric_h(self):
        from director_ai.research.consciousness.pgbo import PGBOConfig, PGBOEngine

        engine = PGBOEngine(N=4, config=PGBOConfig(kappa=0.5))
        theta0 = np.array([0.0, 0.5, 1.0, 1.5])
        engine.compute(theta0, dt=0.01)
        theta1 = np.array([0.1, 0.6, 1.1, 1.6])
        u, h = engine.compute(theta1, dt=0.01)
        np.testing.assert_allclose(h, h.T, atol=1e-12)

    def test_h_munu_positive_semidefinite(self):
        from director_ai.research.consciousness.pgbo import PGBOConfig, PGBOEngine

        engine = PGBOEngine(N=4, config=PGBOConfig(kappa=0.5))
        engine.compute(np.array([0.0, 1.0, 2.0, 3.0]), dt=0.01)
        engine.compute(np.array([0.1, 1.1, 2.1, 3.1]), dt=0.01)
        eigs = np.linalg.eigvalsh(engine.h_munu)
        assert np.all(eigs >= -1e-12)

    def test_traceless_mode(self):
        from director_ai.research.consciousness.pgbo import PGBOConfig, PGBOEngine

        engine = PGBOEngine(N=4, config=PGBOConfig(kappa=0.5, traceless=True))
        engine.compute(np.array([0.0, 1.0, 2.0, 3.0]), dt=0.01)
        engine.compute(np.array([0.1, 1.1, 2.1, 3.1]), dt=0.01)
        # After traceless projection, Tr(g_inv @ h) should be near zero
        actual_trace = float(np.einsum("mn,mn->", engine.g0_inv, engine.h_munu))
        assert abs(actual_trace) < 1e-10

    def test_reset(self):
        from director_ai.research.consciousness.pgbo import PGBOEngine

        engine = PGBOEngine(N=4)
        engine.compute(np.array([0.0, 1.0, 2.0, 3.0]), dt=0.01)
        engine.reset()
        assert engine.u_norm == 0.0
        assert engine.h_frob == 0.0
        assert engine._prev_theta is None

    def test_saturation(self):
        from director_ai.research.consciousness.pgbo import PGBOConfig, PGBOEngine

        engine = PGBOEngine(N=4, config=PGBOConfig(u_cap=1.0, kappa=1.0))
        # Create large phase jumps
        engine.compute(np.zeros(4), dt=0.01)
        engine.compute(np.ones(4) * 100.0, dt=0.001)  # Very fast
        assert engine.u_norm < engine.cfg.u_cap * 2  # Bounded

    def test_standalone_function(self):
        from director_ai.research.consciousness.pgbo import phase_geometry_bridge

        dphi = np.array([1.0, 2.0, 3.0])
        a_mu = np.array([0.1, 0.2, 0.3])
        u, h = phase_geometry_bridge(dphi, a_mu, alpha=0.5, kappa=0.3)
        assert u.shape == (3,)
        assert h.shape == (3, 3)
        np.testing.assert_allclose(h, h.T, atol=1e-12)

    def test_standalone_batch(self):
        from director_ai.research.consciousness.pgbo import phase_geometry_bridge

        dphi = np.random.randn(5, 3)
        a_mu = np.random.randn(5, 3)
        u, h = phase_geometry_bridge(dphi, a_mu, alpha=0.5, kappa=0.3)
        assert u.shape == (5, 3)
        assert h.shape == (5, 3, 3)

    def test_get_state(self):
        from director_ai.research.consciousness.pgbo import PGBOEngine

        engine = PGBOEngine(N=4)
        state = engine.get_state()
        assert "alpha" in state
        assert "kappa" in state
        assert "D" in state
        assert state["D"] == 4
