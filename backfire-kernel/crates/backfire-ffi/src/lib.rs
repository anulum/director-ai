// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel PyO3 FFI Bindings
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
// Note: #[deny(unsafe_code)] not applied — PyO3 proc macros generate
// unsafe blocks internally. All hand-written code in this crate is safe.
//! Python-callable wrappers around the Rust Backfire Kernel.
//!
//! Exposes `RustSafetyKernel`, `RustStreamingKernel`, `RustCoherenceScorer`,
//! and supporting types to Python via PyO3.
//!
//! # FFI Safety
//!
//! - GIL acquired via `Python::with_gil` before every Python callback.
//! - Python exceptions → safe Rust defaults (0.0 for scores, None for strings).
//! - No borrowed references escape the GIL lock scope.
//! - All config validated before storage (`BackfireConfig::validate()`).
//!
//! Install: `cd backfire-kernel && pip install -e crates/backfire-ffi`
//! (requires maturin).
//!
//! Usage from Python:
//! ```python
//! from backfire_kernel import RustSafetyKernel, RustStreamingKernel
//!
//! kernel = RustSafetyKernel(hard_limit=0.5)
//! result = kernel.stream_output(["Hello ", "world"], lambda t: 0.8)
//! ```

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use backfire_core::knowledge::ExternalKnowledge;
use backfire_core::nli::ExternalNli;
use backfire_core::{CoherenceScorer, InMemoryKnowledge, SafetyKernel, StreamingKernel};
use backfire_types::score::CoherenceScore;
use backfire_types::{BackfireConfig, StreamSession};

use backfire_observers::{
    PGBOConfig, PGBOEngine, TCBOConfig, TCBOController, TCBOControllerConfig, TCBOObserver,
};
use backfire_physics::{
    l16_closure::L16CostInputs, params::N_LAYERS, L16Controller, SECFunctional, UPDEState,
    UPDEStepper,
};
use backfire_ssgf::{SSGFConfig, SSGFEngine};

// ─── PyBackfireConfig ───────────────────────────────────────────────

/// Python-visible configuration for the Backfire Kernel.
#[pyclass(name = "BackfireConfig")]
#[derive(Clone)]
struct PyBackfireConfig {
    inner: BackfireConfig,
}

#[pymethods]
impl PyBackfireConfig {
    #[new]
    #[pyo3(signature = (
        coherence_threshold = 0.6,
        hard_limit = 0.5,
        soft_limit = 0.7,
        w_logic = 0.6,
        w_fact = 0.4,
        window_size = 10,
        window_threshold = 0.55,
        trend_window = 5,
        trend_threshold = 0.15,
        history_window = 5,
        deadline_ms = 50,
        logit_entropy_limit = 1.2,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        coherence_threshold: f64,
        hard_limit: f64,
        soft_limit: f64,
        w_logic: f64,
        w_fact: f64,
        window_size: usize,
        window_threshold: f64,
        trend_window: usize,
        trend_threshold: f64,
        history_window: usize,
        deadline_ms: u64,
        logit_entropy_limit: f64,
    ) -> PyResult<Self> {
        let config = BackfireConfig {
            coherence_threshold,
            hard_limit,
            soft_limit,
            w_logic,
            w_fact,
            window_size,
            window_threshold,
            trend_window,
            trend_threshold,
            history_window,
            deadline_ms,
            logit_entropy_limit,
        };
        config
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: config })
    }

    /// Construct from JSON string.
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let config =
            BackfireConfig::from_json(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        config
            .validate()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: config })
    }

    fn __repr__(&self) -> String {
        format!(
            "BackfireConfig(threshold={}, hard_limit={}, deadline_ms={})",
            self.inner.coherence_threshold, self.inner.hard_limit, self.inner.deadline_ms
        )
    }
}

// ─── PyCoherenceScore ───────────────────────────────────────────────

/// Python-visible coherence score result.
#[pyclass(name = "CoherenceScore")]
#[derive(Clone)]
struct PyCoherenceScore {
    inner: CoherenceScore,
}

#[pymethods]
impl PyCoherenceScore {
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    #[getter]
    fn approved(&self) -> bool {
        self.inner.approved
    }

    #[getter]
    fn h_logical(&self) -> f64 {
        self.inner.h_logical
    }

    #[getter]
    fn h_factual(&self) -> f64 {
        self.inner.h_factual
    }

    #[getter]
    fn warning(&self) -> bool {
        self.inner.warning
    }

    /// Evidence is not yet computed on the Rust side; returns None for
    /// API compatibility with the Python CoherenceScore dataclass.
    #[getter]
    fn evidence(&self) -> Option<()> {
        None
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("score", self.inner.score)?;
        dict.set_item("approved", self.inner.approved)?;
        dict.set_item("h_logical", self.inner.h_logical)?;
        dict.set_item("h_factual", self.inner.h_factual)?;
        dict.set_item("warning", self.inner.warning)?;
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "CoherenceScore(score={:.4}, approved={}, h_logical={:.4}, h_factual={:.4}, warning={})",
            self.inner.score, self.inner.approved, self.inner.h_logical, self.inner.h_factual, self.inner.warning
        )
    }
}

// ─── PyStreamSession ────────────────────────────────────────────────

/// Python-visible streaming session trace.
#[pyclass(name = "StreamSession")]
#[derive(Clone)]
struct PyStreamSession {
    inner: StreamSession,
}

#[pymethods]
impl PyStreamSession {
    #[getter]
    fn halted(&self) -> bool {
        self.inner.halted
    }

    #[getter]
    fn halt_index(&self) -> i32 {
        self.inner.halt_index
    }

    #[getter]
    fn halt_reason(&self) -> &str {
        &self.inner.halt_reason
    }

    #[getter]
    fn tokens(&self) -> Vec<String> {
        self.inner.tokens.clone()
    }

    #[getter]
    fn coherence_history(&self) -> Vec<f64> {
        self.inner.coherence_history.clone()
    }

    fn output(&self) -> String {
        self.inner.output()
    }

    fn token_count(&self) -> usize {
        self.inner.token_count()
    }

    fn avg_coherence(&self) -> f64 {
        self.inner.avg_coherence()
    }

    fn min_coherence(&self) -> f64 {
        self.inner.min_coherence()
    }

    fn duration_ms(&self) -> f64 {
        self.inner.duration_ms()
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamSession(tokens={}, halted={}, avg_coh={:.4})",
            self.inner.token_count(),
            self.inner.halted,
            self.inner.avg_coherence(),
        )
    }
}

// ─── RustSafetyKernel ───────────────────────────────────────────────

/// Basic per-token safety kernel exposed to Python.
///
/// Drop-in replacement for `SafetyKernel` from `kernel.py`.
#[pyclass(name = "RustSafetyKernel")]
struct PySafetyKernel {
    inner: SafetyKernel,
}

#[pymethods]
impl PySafetyKernel {
    #[new]
    #[pyo3(signature = (hard_limit = 0.5))]
    fn new(hard_limit: f64) -> Self {
        Self {
            inner: SafetyKernel::new(hard_limit),
        }
    }

    /// Process tokens with a coherence callback.
    ///
    /// Args:
    ///     tokens: List of token strings.
    ///     coherence_callback: Callable[[str], float] — returns score per token.
    ///
    /// Returns:
    ///     Assembled output string, or halt message.
    fn stream_output(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        coherence_callback: PyObject,
    ) -> PyResult<String> {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let cb = |token: &str| -> f64 {
            Python::with_gil(|py| match coherence_callback.call1(py, (token,)) {
                Ok(result) => result.extract::<f64>(py).unwrap_or(0.0),
                Err(_) => 0.0,
            })
        };
        let _ = py;
        Ok(self.inner.stream_output(&token_refs, &cb))
    }

    fn emergency_stop(&self) {
        self.inner.emergency_stop();
    }

    fn reactivate(&self) {
        self.inner.reactivate();
    }

    #[getter]
    fn is_active(&self) -> bool {
        self.inner.is_active()
    }
}

// ─── RustStreamingKernel ────────────────────────────────────────────

/// Streaming safety kernel with sliding window + trend detection.
///
/// Drop-in replacement for `StreamingKernel` from `streaming.py`.
#[pyclass(name = "RustStreamingKernel")]
struct PyStreamingKernel {
    inner: StreamingKernel,
}

#[pymethods]
impl PyStreamingKernel {
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(config: Option<PyBackfireConfig>) -> Self {
        let cfg = config.map(|c| c.inner).unwrap_or_default();
        Self {
            inner: StreamingKernel::new(cfg),
        }
    }

    /// Process tokens with full streaming oversight.
    ///
    /// Returns a `StreamSession` with complete oversight trace.
    fn stream_tokens(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        coherence_callback: PyObject,
    ) -> PyResult<PyStreamSession> {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let cb = |token: &str| -> f64 {
            Python::with_gil(|py| match coherence_callback.call1(py, (token,)) {
                Ok(result) => result.extract::<f64>(py).unwrap_or(0.0),
                Err(_) => 0.0,
            })
        };
        let _ = py;
        let session = self.inner.stream_tokens(&token_refs, &cb);
        Ok(PyStreamSession { inner: session })
    }

    /// Backward-compatible string output.
    fn stream_output(
        &self,
        py: Python<'_>,
        tokens: Vec<String>,
        coherence_callback: PyObject,
    ) -> PyResult<String> {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let cb = |token: &str| -> f64 {
            Python::with_gil(|py| match coherence_callback.call1(py, (token,)) {
                Ok(result) => result.extract::<f64>(py).unwrap_or(0.0),
                Err(_) => 0.0,
            })
        };
        let _ = py;
        Ok(self.inner.stream_output(&token_refs, &cb))
    }

    fn reactivate(&self) {
        self.inner.reactivate();
    }

    #[getter]
    fn is_active(&self) -> bool {
        self.inner.is_active()
    }
}

// ─── RustCoherenceScorer ────────────────────────────────────────────

/// Dual-entropy coherence scorer exposed to Python.
///
/// Drop-in replacement for `CoherenceScorer` from `scorer.py`.
///
/// NLI and knowledge callbacks are Python callables that cross the FFI
/// boundary per invocation. The rest of the hot path runs in Rust.
#[pyclass(name = "RustCoherenceScorer")]
struct PyCoherenceScorer {
    inner: CoherenceScorer,
}

#[pymethods]
impl PyCoherenceScorer {
    /// Create a new scorer.
    ///
    /// Args:
    ///     config: Optional BackfireConfig (uses defaults if None).
    ///     nli_callback: Optional Callable[[str, str], float] for NLI scoring.
    ///                   If None, uses the heuristic NLI fallback.
    ///     knowledge_callback: Optional Callable[[str], Optional[str]] for RAG.
    ///                        If None, uses in-memory default facts.
    #[new]
    #[pyo3(signature = (config = None, nli_callback = None, knowledge_callback = None))]
    fn new(
        config: Option<PyBackfireConfig>,
        nli_callback: Option<PyObject>,
        knowledge_callback: Option<PyObject>,
    ) -> PyResult<Self> {
        let cfg = config.map(|c| c.inner).unwrap_or_default();

        let nli: Arc<dyn backfire_core::nli::NliBackend> = match nli_callback {
            Some(cb) => Arc::new(ExternalNli::new(move |premise: &str, hypothesis: &str| {
                Python::with_gil(|py| match cb.call1(py, (premise, hypothesis)) {
                    Ok(result) => result.extract::<f64>(py).unwrap_or(0.5),
                    Err(_) => 0.5,
                })
            })),
            None => Arc::new(backfire_core::HeuristicNli),
        };

        let knowledge: Arc<dyn backfire_core::knowledge::GroundTruthStore> =
            match knowledge_callback {
                Some(cb) => Arc::new(ExternalKnowledge::new(move |query: &str| {
                    Python::with_gil(|py| match cb.call1(py, (query,)) {
                        Ok(result) => result.extract::<Option<String>>(py).unwrap_or(None),
                        Err(_) => None,
                    })
                })),
                None => Arc::new(InMemoryKnowledge::new()),
            };

        Ok(Self {
            inner: CoherenceScorer::new(cfg, nli, knowledge),
        })
    }

    /// Score an action and decide whether to approve it.
    ///
    /// Returns: tuple(approved: bool, score: CoherenceScore)
    fn review(&self, prompt: &str, action: &str) -> PyResult<(bool, PyCoherenceScore)> {
        let (approved, score) = self.inner.review(prompt, action);
        Ok((approved, PyCoherenceScore { inner: score }))
    }

    /// Compute composite divergence (lower is better).
    fn compute_divergence(&self, prompt: &str, action: &str) -> f64 {
        self.inner.compute_divergence(prompt, action)
    }

    /// Logical divergence via NLI.
    fn calculate_logical_divergence(&self, prompt: &str, text_output: &str) -> f64 {
        self.inner.calculate_logical_divergence(prompt, text_output)
    }

    /// Factual divergence via ground truth store.
    fn calculate_factual_divergence(&self, prompt: &str, text_output: &str) -> f64 {
        self.inner.calculate_factual_divergence(prompt, text_output)
    }

    #[getter]
    fn history_len(&self) -> usize {
        self.inner.history_len()
    }

    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.config().coherence_threshold
    }

    #[setter]
    fn set_threshold(&mut self, value: f64) {
        self.inner.set_threshold(value);
    }

    #[getter]
    fn soft_limit(&self) -> f64 {
        self.inner.config().soft_limit
    }

    #[setter]
    fn set_soft_limit(&mut self, value: f64) {
        self.inner.set_soft_limit(value);
    }

    #[getter]
    fn use_nli(&self) -> bool {
        true
    }
}

// ─── RustUPDEStepper ──────────────────────────────────────────────

/// UPDE Kuramoto integrator for 16-layer SCPN phase dynamics.
#[pyclass(name = "RustUPDEStepper")]
struct PyUPDEStepper {
    inner: UPDEStepper,
}

#[pymethods]
impl PyUPDEStepper {
    #[new]
    #[pyo3(signature = (dt = 0.01, field_pressure = 0.1, noise_amplitude = 0.05))]
    fn new(dt: f64, field_pressure: f64, noise_amplitude: f64) -> Self {
        Self {
            inner: UPDEStepper::new(dt, field_pressure, noise_amplitude),
        }
    }

    /// Create initial state with given phases.
    #[staticmethod]
    fn create_state(theta: Vec<f64>) -> PyResult<PyObject> {
        let state = UPDEState::new(theta);
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("theta", state.theta.clone())?;
            dict.set_item("dtheta_dt", state.dtheta_dt.clone())?;
            dict.set_item("t", state.t)?;
            dict.set_item("r_global", state.r_global)?;
            dict.set_item("step_count", state.step_count)?;
            Ok(dict.into())
        })
    }

    /// Create random initial state.
    #[staticmethod]
    fn random_state() -> PyResult<PyObject> {
        let state = UPDEState::random(N_LAYERS);
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("theta", state.theta.clone())?;
            dict.set_item("dtheta_dt", state.dtheta_dt.clone())?;
            dict.set_item("t", state.t)?;
            dict.set_item("r_global", state.r_global)?;
            dict.set_item("step_count", state.step_count)?;
            Ok(dict.into())
        })
    }

    /// Advance by n_steps. Returns dict with theta, dtheta_dt, t, r_global, step_count.
    fn run(&mut self, theta: Vec<f64>, n_steps: u64) -> PyResult<PyObject> {
        let state = UPDEState::new(theta);
        let result = self
            .inner
            .run(&state, n_steps)
            .map_err(PyValueError::new_err)?;
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("theta", result.theta.clone())?;
            dict.set_item("dtheta_dt", result.dtheta_dt.clone())?;
            dict.set_item("t", result.t)?;
            dict.set_item("r_global", result.r_global)?;
            dict.set_item("step_count", result.step_count)?;
            Ok(dict.into())
        })
    }
}

// ─── RustSECFunctional ───────────────────────────────────────────

/// SEC Lyapunov functional for coherence scoring.
#[pyclass(name = "RustSECFunctional")]
struct PySECFunctional {
    inner: SECFunctional,
}

#[pymethods]
impl PySECFunctional {
    #[new]
    #[pyo3(signature = (lambda_omega = 0.1, lambda_entropy = 0.01))]
    fn new(lambda_omega: f64, lambda_entropy: f64) -> Self {
        Self {
            inner: SECFunctional::new(lambda_omega, lambda_entropy),
        }
    }

    /// Evaluate the full SEC functional.
    #[pyo3(signature = (theta, theta_prev = None, dt = 0.01))]
    fn evaluate(
        &mut self,
        theta: Vec<f64>,
        theta_prev: Option<Vec<f64>>,
        dt: f64,
    ) -> PyResult<PyObject> {
        let prev_ref = theta_prev.as_deref();
        let result = self
            .inner
            .evaluate(&theta, prev_ref, dt)
            .map_err(PyValueError::new_err)?;
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("v", result.v)?;
            dict.set_item("v_normalised", result.v_normalised)?;
            dict.set_item("r_global", result.r_global)?;
            dict.set_item("dv_dt", result.dv_dt)?;
            dict.set_item("coherence_score", result.coherence_score)?;
            dict.set_item("v_coupling", result.v_coupling)?;
            dict.set_item("v_frequency", result.v_frequency)?;
            dict.set_item("v_entropy", result.v_entropy)?;
            Ok(dict.into())
        })
    }

    /// Update coupling matrix (e.g. when SSGF geometry changes W).
    ///
    /// Accepts a flat 16×16 row-major array (256 elements).
    fn update_coupling(&mut self, knm_flat: Vec<f64>) -> PyResult<()> {
        if knm_flat.len() != N_LAYERS * N_LAYERS {
            return Err(PyValueError::new_err(format!(
                "expected {} elements, got {}",
                N_LAYERS * N_LAYERS,
                knm_flat.len()
            )));
        }
        let mut knm = [[0.0f64; N_LAYERS]; N_LAYERS];
        for i in 0..N_LAYERS {
            for j in 0..N_LAYERS {
                knm[i][j] = knm_flat[i * N_LAYERS + j];
            }
        }
        self.inner.update_coupling(knm);
        Ok(())
    }

    /// Critical coupling K_c estimate.
    fn critical_coupling(&self) -> f64 {
        self.inner.critical_coupling()
    }
}

// ─── RustL16Controller ───────────────────────────────────────────

/// L16 Director cybernetic closure controller.
#[pyclass(name = "RustL16Controller")]
struct PyL16Controller {
    inner: L16Controller,
}

#[pymethods]
impl PyL16Controller {
    #[new]
    #[pyo3(signature = (
        n = 16,
        plv_threshold = 0.6,
        plv_window = 10,
        h_rec_window = 5,
        refusal_lr_factor = 0.5,
        refusal_d_factor = 0.5,
        refusal_tau_factor = 1.5,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n: usize,
        plv_threshold: f64,
        plv_window: usize,
        h_rec_window: usize,
        refusal_lr_factor: f64,
        refusal_d_factor: f64,
        refusal_tau_factor: f64,
    ) -> Self {
        Self {
            inner: L16Controller::new(
                n,
                plv_threshold,
                plv_window,
                h_rec_window,
                refusal_lr_factor,
                refusal_d_factor,
                refusal_tau_factor,
            ),
        }
    }

    /// Execute one L16 controller step.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        theta: Vec<f64>,
        r_global: f64,
        plv: f64,
        c7: f64,
        c8: f64,
        c10: f64,
        p_h1: f64,
        h_frob: f64,
        dt: f64,
    ) -> PyResult<PyObject> {
        let costs = L16CostInputs {
            c7_symbolic: c7,
            c8_phase: c8,
            c10_boundary: c10,
            p_h1,
            h_frob,
        };
        let result = self
            .inner
            .step(&theta, &[], 0, &[], 0, r_global, plv, &costs, dt);
        Python::with_gil(|py| {
            let dict = PyDict::new(py);
            dict.set_item("lambda7", result.lambda7)?;
            dict.set_item("lambda8", result.lambda8)?;
            dict.set_item("lambda10", result.lambda10)?;
            dict.set_item("nu_star", result.nu_star)?;
            dict.set_item("gate_open", result.gate_open)?;
            dict.set_item("refusal", result.refusal)?;
            dict.set_item("h_rec", result.h_rec)?;
            dict.set_item("dh_rec_dt", result.dh_rec_dt)?;
            dict.set_item("lyapunov_stable", result.lyapunov_stable)?;
            dict.set_item("avg_plv", result.avg_plv)?;
            dict.set_item("lr_z_scale", result.lr_z_scale)?;
            dict.set_item("d_theta_scale", result.d_theta_scale)?;
            dict.set_item("tau_d_scale", result.tau_d_scale)?;
            Ok(dict.into())
        })
    }

    /// Check if PLV precedence gate is open.
    fn plv_gate_open(&self) -> bool {
        self.inner.plv_gate_open()
    }
}

// ─── RustTCBOObserver ─────────────────────────────────────────────

/// TCBO boundary observer.
#[pyclass(name = "RustTCBOObserver")]
struct PyTCBOObserver {
    inner: TCBOObserver,
}

#[pymethods]
impl PyTCBOObserver {
    #[new]
    #[pyo3(signature = (n = 16, tau_h1 = 0.72, beta = 8.0, window_size = 50))]
    fn new(n: usize, tau_h1: f64, beta: f64, window_size: usize) -> Self {
        let mut cfg = TCBOConfig::default();
        cfg.tau_h1 = tau_h1;
        cfg.beta = beta;
        cfg.window_size = window_size;
        Self {
            inner: TCBOObserver::new(n, cfg),
        }
    }

    /// Push a phase vector and compute p_h1.
    fn push_and_compute(&mut self, theta: Vec<f64>, force: bool) -> f64 {
        self.inner.push_and_compute(&theta, force)
    }

    /// Push a phase vector without computing.
    fn push(&mut self, theta: Vec<f64>) {
        self.inner.push(&theta);
    }

    /// Force computation of p_h1.
    fn compute(&mut self) -> f64 {
        self.inner.compute(true)
    }

    #[getter]
    fn p_h1(&self) -> f64 {
        self.inner.p_h1
    }

    #[getter]
    fn is_conscious(&self) -> bool {
        self.inner.is_conscious
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ─── RustTCBOController ──────────────────────────────────────────

/// TCBO PI controller for gap-junction coupling.
#[pyclass(name = "RustTCBOController")]
struct PyTCBOController {
    inner: TCBOController,
}

#[pymethods]
impl PyTCBOController {
    #[new]
    #[pyo3(signature = (tau_h1 = 0.72, kp = 0.8, ki = 0.2, kappa_max = 5.0))]
    fn new(tau_h1: f64, kp: f64, ki: f64, kappa_max: f64) -> Self {
        let mut cfg = TCBOControllerConfig::default();
        cfg.tau_h1 = tau_h1;
        cfg.kp = kp;
        cfg.ki = ki;
        cfg.kappa_max = kappa_max;
        Self {
            inner: TCBOController::new(cfg),
        }
    }

    /// Execute one PI step. Returns new kappa.
    fn step(&mut self, p_h1: f64, current_kappa: f64, dt: f64) -> f64 {
        self.inner.step(p_h1, current_kappa, dt)
    }

    /// Check if boundary gate is open.
    #[pyo3(signature = (p_h1 = None))]
    fn is_gate_open(&self, p_h1: Option<f64>) -> bool {
        self.inner.is_gate_open(p_h1)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ─── RustPGBOEngine ──────────────────────────────────────────────

/// PGBO Phase→Geometry Bridge Operator engine.
#[pyclass(name = "RustPGBOEngine")]
struct PyPGBOEngine {
    inner: PGBOEngine,
}

#[pymethods]
impl PyPGBOEngine {
    #[new]
    #[pyo3(signature = (n = 16, alpha = 0.5, kappa = 0.3, u_cap = 10.0, traceless = false))]
    fn new(n: usize, alpha: f64, kappa: f64, u_cap: f64, traceless: bool) -> Self {
        let mut cfg = PGBOConfig::default();
        cfg.alpha = alpha;
        cfg.kappa = kappa;
        cfg.u_cap = u_cap;
        cfg.traceless = traceless;
        Self {
            inner: PGBOEngine::new(n, cfg),
        }
    }

    /// Compute PGBO from current phases. Returns dict with u_mu, h_munu, u_norm, h_frob.
    fn compute(&mut self, py: Python<'_>, theta: Vec<f64>, dt: f64) -> PyResult<PyObject> {
        self.inner.compute(&theta, dt);
        let dict = PyDict::new(py);
        dict.set_item("u_mu", self.inner.u_mu.clone())?;
        dict.set_item("h_munu", self.inner.h_munu.clone())?;
        dict.set_item("u_norm", self.inner.u_norm)?;
        dict.set_item("h_trace", self.inner.h_trace)?;
        dict.set_item("h_frob", self.inner.h_frob)?;
        Ok(dict.into())
    }

    /// Set boundary injection potential (L10 handle).
    fn set_boundary_potential(&mut self, a_mu: Vec<f64>) {
        self.inner.set_boundary_potential(&a_mu);
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn u_norm(&self) -> f64 {
        self.inner.u_norm
    }

    #[getter]
    fn h_frob(&self) -> f64 {
        self.inner.h_frob
    }
}

// ─── RustSSGFEngine ──────────────────────────────────────────────

/// SSGF Geometry Engine — outer-cycle orchestrator.
#[pyclass(name = "RustSSGFEngine")]
struct PySSGFEngine {
    inner: SSGFEngine,
}

#[pymethods]
impl PySSGFEngine {
    #[new]
    #[pyo3(signature = (
        omega = None,
        k = None,
        n = 16,
        lr_z = 0.01,
        n_micro = 10,
        noise_amp = 0.02,
        pgbo_enabled = true,
        seed = 42,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        omega: Option<Vec<f64>>,
        k: Option<Vec<f64>>,
        n: usize,
        lr_z: f64,
        n_micro: usize,
        noise_amp: f64,
        pgbo_enabled: bool,
        seed: u64,
    ) -> PyResult<Self> {
        use backfire_physics::params::{build_knm_matrix, OMEGA_N};

        let omega_vec = omega.unwrap_or_else(|| OMEGA_N.to_vec());
        let k_vec = k.unwrap_or_else(|| {
            let knm = build_knm_matrix();
            knm.iter().flat_map(|row| row.iter().copied()).collect()
        });

        if omega_vec.len() != n {
            return Err(PyValueError::new_err(format!(
                "omega length {} != n={}",
                omega_vec.len(),
                n
            )));
        }
        if k_vec.len() != n * n {
            return Err(PyValueError::new_err(format!(
                "k length {} != n*n={}",
                k_vec.len(),
                n * n
            )));
        }

        let config = SSGFConfig {
            n,
            lr_z,
            n_micro,
            noise_amp,
            pgbo_enabled,
            seed,
            ..SSGFConfig::default()
        };

        Ok(Self {
            inner: SSGFEngine::new(&omega_vec, &k_vec, config),
        })
    }

    /// Run n outer-cycle steps. Returns list of step log dicts.
    fn run(&mut self, py: Python<'_>, n_outer: usize) -> PyResult<PyObject> {
        let logs = self.inner.run(n_outer);
        let list = pyo3::types::PyList::empty(py);
        for log in &logs {
            let dict = PyDict::new(py);
            dict.set_item("step", log.step)?;
            dict.set_item("r_global", log.r_global)?;
            dict.set_item("fiedler_value", log.fiedler_value)?;
            dict.set_item("spectral_gap", log.spectral_gap)?;
            dict.set_item("h_rec", log.h_rec)?;
            dict.set_item("gate_open", log.gate_open)?;
            dict.set_item("refusal", log.refusal)?;
            dict.set_item("w_valid", log.w_valid)?;
            dict.set_item("eigval_ordered", log.eigval_ordered)?;
            dict.set_item("u_total", log.costs.u_total)?;
            dict.set_item("c_micro", log.costs.c_micro)?;
            dict.set_item("pgbo_u_norm", log.pgbo_u_norm)?;
            dict.set_item("pgbo_h_frob", log.pgbo_h_frob)?;
            dict.set_item("tcbo_p_h1", log.tcbo_p_h1)?;
            dict.set_item("tcbo_gate_open", log.tcbo_gate_open)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    /// Execute one outer-cycle step. Returns step log dict.
    fn outer_step(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        let log = self.inner.outer_step();
        let dict = PyDict::new(py);
        dict.set_item("step", log.step)?;
        dict.set_item("r_global", log.r_global)?;
        dict.set_item("fiedler_value", log.fiedler_value)?;
        dict.set_item("spectral_gap", log.spectral_gap)?;
        dict.set_item("h_rec", log.h_rec)?;
        dict.set_item("gate_open", log.gate_open)?;
        dict.set_item("refusal", log.refusal)?;
        dict.set_item("w_valid", log.w_valid)?;
        dict.set_item("eigval_ordered", log.eigval_ordered)?;
        dict.set_item("u_total", log.costs.u_total)?;
        dict.set_item("c_micro", log.costs.c_micro)?;
        dict.set_item("pgbo_u_norm", log.pgbo_u_norm)?;
        dict.set_item("pgbo_h_frob", log.pgbo_h_frob)?;
        dict.set_item("tcbo_p_h1", log.tcbo_p_h1)?;
        dict.set_item("tcbo_gate_open", log.tcbo_gate_open)?;
        Ok(dict.into())
    }

    /// Get audio mapping from current state.
    fn audio_mapping(&self, py: Python<'_>) -> PyResult<PyObject> {
        let m = self.inner.audio_mapping();
        let dict = PyDict::new(py);
        dict.set_item("r_global", m.r_global)?;
        dict.set_item("entrainment_intensity", m.entrainment_intensity)?;
        dict.set_item("beat_hz", m.beat_hz)?;
        dict.set_item("pulse_hz", m.pulse_hz)?;
        dict.set_item("spatial_angle_deg", m.spatial_angle_deg)?;
        dict.set_item("brainwave_band", m.brainwave_band)?;
        dict.set_item("fiedler_stability", m.fiedler_stability)?;
        dict.set_item("spectral_gap", m.spectral_gap)?;
        dict.set_item("geometry_density", m.geometry_density)?;
        dict.set_item("l16_gate_open", m.l16_gate_open)?;
        dict.set_item("l16_refusal", m.l16_refusal)?;
        dict.set_item("tcbo_p_h1", m.tcbo_p_h1)?;
        dict.set_item("tcbo_gate_open", m.tcbo_gate_open)?;
        dict.set_item("pgbo_u_norm", m.pgbo_u_norm)?;
        dict.set_item("pgbo_h_frob", m.pgbo_h_frob)?;
        dict.set_item("theurgic_mode", m.theurgic_mode)?;
        dict.set_item("healing_acceleration", m.healing_acceleration)?;
        Ok(dict.into())
    }

    /// Inject TCBO p_h1 from external observer.
    fn set_tcbo_p_h1(&mut self, p_h1: f64) {
        self.inner.set_tcbo_p_h1(p_h1);
    }

    /// Current R_global.
    fn r_global(&self) -> f64 {
        self.inner.r_global()
    }

    /// Current step count.
    fn step_count(&self) -> usize {
        self.inner.step_count()
    }
}

// ─── Module Registration ────────────────────────────────────────────

/// Backfire Kernel — Rust-accelerated safety gate for Director-Class AI.
///
/// This module exposes the entire hot-path safety gate to Python:
/// - `BackfireConfig` — configuration
/// - `RustSafetyKernel` — basic per-token hard limit
/// - `RustStreamingKernel` — sliding window + trend detection
/// - `RustCoherenceScorer` — dual-entropy coherence scoring
/// - `CoherenceScore` — score result
/// - `StreamSession` — streaming session trace
#[pymodule]
fn backfire_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core safety gate
    m.add_class::<PyBackfireConfig>()?;
    m.add_class::<PyCoherenceScore>()?;
    m.add_class::<PyStreamSession>()?;
    m.add_class::<PySafetyKernel>()?;
    m.add_class::<PyStreamingKernel>()?;
    m.add_class::<PyCoherenceScorer>()?;
    // Physics engine
    m.add_class::<PyUPDEStepper>()?;
    m.add_class::<PySECFunctional>()?;
    m.add_class::<PyL16Controller>()?;
    // Boundary observers
    m.add_class::<PyTCBOObserver>()?;
    m.add_class::<PyTCBOController>()?;
    m.add_class::<PyPGBOEngine>()?;
    // SSGF geometry engine
    m.add_class::<PySSGFEngine>()?;
    Ok(())
}
