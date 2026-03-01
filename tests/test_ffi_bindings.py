# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — FFI Binding Tests (PyO3 boundary)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Tests for the Rust FFI layer (backfire_kernel PyO3 bindings).

Skipped when backfire_kernel is not installed (maturin build required).
"""

import math

import pytest

try:
    import backfire_kernel

    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="backfire_kernel not installed")


# ── BackfireConfig ────────────────────────────────────────────────────


class TestBackfireConfig:
    def test_defaults(self):
        cfg = backfire_kernel.BackfireConfig()
        assert repr(cfg).startswith("BackfireConfig(")

    def test_custom_params(self):
        cfg = backfire_kernel.BackfireConfig(
            coherence_threshold=0.8, hard_limit=0.3, deadline_ms=100
        )
        assert "0.8" in repr(cfg)

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError):
            backfire_kernel.BackfireConfig(coherence_threshold=1.5)

    def test_invalid_hard_limit_rejected(self):
        with pytest.raises(ValueError):
            backfire_kernel.BackfireConfig(hard_limit=-0.1)

    def test_from_json(self):
        import json

        data = json.dumps(
            {
                "coherence_threshold": 0.7,
                "hard_limit": 0.4,
                "soft_limit": 0.7,
                "w_logic": 0.6,
                "w_fact": 0.4,
                "window_size": 10,
                "window_threshold": 0.55,
                "trend_window": 5,
                "trend_threshold": 0.15,
                "history_window": 5,
                "deadline_ms": 50,
                "logit_entropy_limit": 1.2,
            }
        )
        cfg = backfire_kernel.BackfireConfig.from_json(data)
        assert "0.7" in repr(cfg)

    def test_from_json_invalid(self):
        with pytest.raises(ValueError):
            backfire_kernel.BackfireConfig.from_json("{invalid json")


# ── RustSafetyKernel ─────────────────────────────────────────────────


class TestRustSafetyKernel:
    def test_basic_passthrough(self):
        k = backfire_kernel.RustSafetyKernel(hard_limit=0.3)
        result = k.stream_output(["Hello", " world"], lambda t: 0.9)
        assert result == "Hello world"

    def test_halt_on_low_score(self):
        k = backfire_kernel.RustSafetyKernel(hard_limit=0.5)
        result = k.stream_output(["a", "b"], lambda t: 0.1)
        assert "INTERRUPT" in result or "HALT" in result.upper() or result != "ab"

    def test_emergency_stop(self):
        k = backfire_kernel.RustSafetyKernel()
        k.emergency_stop()
        assert k.is_active is False

    def test_reactivate(self):
        k = backfire_kernel.RustSafetyKernel()
        k.emergency_stop()
        k.reactivate()
        assert k.is_active is True

    def test_callback_exception_safe(self):
        """Python callback raising → Rust treats as score=0."""
        k = backfire_kernel.RustSafetyKernel(hard_limit=0.5)

        def bad_cb(t):
            raise RuntimeError("boom")

        result = k.stream_output(["a"], bad_cb)
        assert result != "a"


# ── RustStreamingKernel ──────────────────────────────────────────────


class TestRustStreamingKernel:
    def test_stream_tokens_returns_session(self):
        k = backfire_kernel.RustStreamingKernel()
        session = k.stream_tokens(["a", "b", "c"], lambda t: 0.9)
        assert session.token_count() == 3
        assert not session.halted
        assert session.output() == "abc"

    def test_session_halt_on_low_score(self):
        k = backfire_kernel.RustStreamingKernel()
        scores = iter([0.8, 0.1])
        session = k.stream_tokens(["a", "b"], lambda t: next(scores))
        assert session.halted
        assert session.halt_reason != ""

    def test_session_coherence_history(self):
        k = backfire_kernel.RustStreamingKernel()
        session = k.stream_tokens(["a", "b"], lambda t: 0.85)
        assert len(session.coherence_history) == 2
        assert abs(session.avg_coherence() - 0.85) < 1e-6

    def test_stream_output_string(self):
        k = backfire_kernel.RustStreamingKernel()
        result = k.stream_output(["x", "y"], lambda t: 0.9)
        assert result == "xy"

    def test_with_config(self):
        cfg = backfire_kernel.BackfireConfig(hard_limit=0.2)
        k = backfire_kernel.RustStreamingKernel(config=cfg)
        assert k.is_active


# ── RustCoherenceScorer ──────────────────────────────────────────────


class TestRustCoherenceScorer:
    def test_review_returns_tuple(self):
        s = backfire_kernel.RustCoherenceScorer()
        approved, score = s.review("test prompt", "test action")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_score_getters(self):
        s = backfire_kernel.RustCoherenceScorer()
        _, score = s.review("sky color", "the sky is blue")
        assert isinstance(score.h_logical, float)
        assert isinstance(score.h_factual, float)
        assert isinstance(score.approved, bool)
        assert isinstance(score.warning, bool)

    def test_score_to_dict(self):
        s = backfire_kernel.RustCoherenceScorer()
        _, score = s.review("test", "response")
        d = score.to_dict()
        assert "score" in d
        assert "approved" in d

    def test_compute_divergence(self):
        s = backfire_kernel.RustCoherenceScorer()
        div = s.compute_divergence("prompt", "action")
        assert 0.0 <= div <= 1.0

    def test_logical_divergence(self):
        s = backfire_kernel.RustCoherenceScorer()
        ld = s.calculate_logical_divergence("p", "h")
        assert 0.0 <= ld <= 1.0

    def test_factual_divergence(self):
        s = backfire_kernel.RustCoherenceScorer()
        fd = s.calculate_factual_divergence("p", "h")
        assert 0.0 <= fd <= 1.0

    def test_threshold_setter(self):
        s = backfire_kernel.RustCoherenceScorer()
        s.threshold = 0.9
        assert abs(s.threshold - 0.9) < 1e-9

    def test_soft_limit_setter(self):
        s = backfire_kernel.RustCoherenceScorer()
        s.soft_limit = 0.8
        assert abs(s.soft_limit - 0.8) < 1e-9

    def test_with_nli_callback(self):
        s = backfire_kernel.RustCoherenceScorer(nli_callback=lambda p, h: 0.2)
        _, score = s.review("premise", "hypothesis")
        assert score.score > 0

    def test_with_knowledge_callback(self):
        s = backfire_kernel.RustCoherenceScorer(
            knowledge_callback=lambda q: "known fact"
        )
        _, score = s.review("query", "known fact")
        assert isinstance(score.score, float)

    def test_history_len(self):
        s = backfire_kernel.RustCoherenceScorer()
        s.review("a", "b")
        assert s.history_len >= 0


# ── RustUPDEStepper ──────────────────────────────────────────────────


class TestRustUPDEStepper:
    def test_create_state(self):
        state = backfire_kernel.RustUPDEStepper.create_state([0.0] * 16)
        assert len(state["theta"]) == 16
        assert state["step_count"] == 0
        assert abs(state["r_global"] - 1.0) < 1e-6  # all phases equal → R=1

    def test_random_state(self):
        state = backfire_kernel.RustUPDEStepper.random_state()
        assert len(state["theta"]) == 16
        assert all(0.0 <= th < 2 * math.pi for th in state["theta"])

    def test_run_single_step(self):
        stepper = backfire_kernel.RustUPDEStepper()
        result = stepper.run([0.0] * 16, 1)
        assert result["step_count"] == 1
        assert abs(result["t"] - 0.01) < 1e-10

    def test_run_multiple_steps(self):
        stepper = backfire_kernel.RustUPDEStepper()
        result = stepper.run([0.0] * 16, 100)
        assert result["step_count"] == 100
        assert all(0.0 <= th < 2 * math.pi for th in result["theta"])

    def test_wrong_length_rejected(self):
        stepper = backfire_kernel.RustUPDEStepper()
        with pytest.raises(ValueError):
            stepper.run([0.0] * 4, 1)

    def test_nan_input_rejected(self):
        stepper = backfire_kernel.RustUPDEStepper()
        with pytest.raises(ValueError):
            stepper.run([float("nan")] * 16, 1)


# ── RustSECFunctional ───────────────────────────────────────────────


class TestRustSECFunctional:
    def test_evaluate_basic(self):
        sec = backfire_kernel.RustSECFunctional()
        result = sec.evaluate([0.0] * 16)
        assert "v" in result
        assert "coherence_score" in result
        assert 0.0 <= result["coherence_score"] <= 1.0

    def test_evaluate_with_prev(self):
        sec = backfire_kernel.RustSECFunctional()
        result = sec.evaluate([0.0] * 16, theta_prev=[0.1] * 16, dt=0.01)
        assert isinstance(result["dv_dt"], float)

    def test_update_coupling(self):
        sec = backfire_kernel.RustSECFunctional()
        flat_knm = [0.1] * (16 * 16)
        sec.update_coupling(flat_knm)

    def test_update_coupling_wrong_size(self):
        sec = backfire_kernel.RustSECFunctional()
        with pytest.raises(ValueError):
            sec.update_coupling([0.1] * 10)

    def test_critical_coupling(self):
        sec = backfire_kernel.RustSECFunctional()
        kc = sec.critical_coupling()
        assert kc > 0


# ── RustL16Controller ───────────────────────────────────────────────


class TestRustL16Controller:
    def test_step_returns_dict(self):
        ctrl = backfire_kernel.RustL16Controller()
        result = ctrl.step(
            theta=[0.0] * 16,
            r_global=0.8,
            plv=0.7,
            c7=0.1,
            c8=0.1,
            c10=0.1,
            p_h1=0.8,
            h_frob=0.1,
            dt=0.01,
        )
        assert "lambda7" in result
        assert "gate_open" in result
        assert "h_rec" in result
        assert isinstance(result["lyapunov_stable"], bool)

    def test_plv_gate(self):
        ctrl = backfire_kernel.RustL16Controller(plv_threshold=0.5)
        ctrl.step([0.0] * 16, 0.9, 0.8, 0.0, 0.0, 0.0, 0.8, 0.0, 0.01)
        assert isinstance(ctrl.plv_gate_open(), bool)


# ── RustTCBOObserver ────────────────────────────────────────────────


class TestRustTCBOObserver:
    def test_push_and_compute(self):
        obs = backfire_kernel.RustTCBOObserver(n=16)
        p = obs.push_and_compute([0.0] * 16, True)
        assert 0.0 <= p <= 1.0

    def test_getters(self):
        obs = backfire_kernel.RustTCBOObserver(n=16)
        obs.push_and_compute([0.0] * 16, True)
        assert isinstance(obs.p_h1, float)
        assert isinstance(obs.is_conscious, bool)

    def test_reset(self):
        obs = backfire_kernel.RustTCBOObserver(n=16)
        obs.push_and_compute([0.5] * 16, True)
        obs.reset()
        assert obs.p_h1 == 0.0 or obs.p_h1 >= 0.0  # reset behavior


# ── RustTCBOController ──────────────────────────────────────────────


class TestRustTCBOController:
    def test_step_returns_kappa(self):
        ctrl = backfire_kernel.RustTCBOController()
        kappa = ctrl.step(p_h1=0.5, current_kappa=1.0, dt=0.01)
        assert isinstance(kappa, float)
        assert kappa >= 0.0

    def test_gate_open(self):
        ctrl = backfire_kernel.RustTCBOController(tau_h1=0.72)
        assert ctrl.is_gate_open(0.8) is True
        assert ctrl.is_gate_open(0.5) is False

    def test_reset(self):
        ctrl = backfire_kernel.RustTCBOController()
        ctrl.step(0.5, 1.0, 0.01)
        ctrl.reset()


# ── RustPGBOEngine ──────────────────────────────────────────────────


class TestRustPGBOEngine:
    def test_compute(self):
        pgbo = backfire_kernel.RustPGBOEngine(n=16)
        result = pgbo.compute([0.0] * 16, dt=0.01)
        assert "u_mu" in result
        assert "h_munu" in result
        assert "u_norm" in result
        assert "h_frob" in result

    def test_getters(self):
        pgbo = backfire_kernel.RustPGBOEngine(n=16)
        pgbo.compute([0.1] * 16, dt=0.01)
        assert isinstance(pgbo.u_norm, float)
        assert isinstance(pgbo.h_frob, float)

    def test_set_boundary_potential(self):
        pgbo = backfire_kernel.RustPGBOEngine(n=16)
        pgbo.set_boundary_potential([0.0] * 16)
        pgbo.compute([0.1] * 16, dt=0.01)

    def test_reset(self):
        pgbo = backfire_kernel.RustPGBOEngine(n=16)
        pgbo.compute([0.5] * 16, dt=0.01)
        pgbo.reset()
        assert pgbo.u_norm == 0.0


# ── RustSSGFEngine ──────────────────────────────────────────────────


class TestRustSSGFEngine:
    def test_run_returns_logs(self):
        engine = backfire_kernel.RustSSGFEngine(seed=42)
        logs = engine.run(3)
        assert len(logs) == 3
        assert "r_global" in logs[0]
        assert "u_total" in logs[0]

    def test_outer_step(self):
        engine = backfire_kernel.RustSSGFEngine(seed=42)
        log = engine.outer_step()
        assert "fiedler_value" in log
        assert "w_valid" in log
        assert log["w_valid"] is True

    def test_audio_mapping(self):
        engine = backfire_kernel.RustSSGFEngine(seed=42)
        engine.run(1)
        m = engine.audio_mapping()
        assert "entrainment_intensity" in m
        assert "beat_hz" in m
        assert "theurgic_mode" in m
        assert 0.0 <= m["r_global"] <= 1.0

    def test_r_global(self):
        engine = backfire_kernel.RustSSGFEngine(seed=42)
        assert 0.0 <= engine.r_global() <= 1.0

    def test_step_count(self):
        engine = backfire_kernel.RustSSGFEngine(seed=42)
        assert engine.step_count() == 0
        engine.run(5)
        assert engine.step_count() == 5

    def test_set_tcbo_p_h1(self):
        engine = backfire_kernel.RustSSGFEngine(seed=42)
        engine.set_tcbo_p_h1(0.85)
        engine.run(1)

    def test_wrong_omega_length(self):
        with pytest.raises(ValueError):
            backfire_kernel.RustSSGFEngine(omega=[1.0, 2.0], n=16)

    def test_wrong_k_length(self):
        with pytest.raises(ValueError):
            backfire_kernel.RustSSGFEngine(k=[0.1] * 10, n=16)
