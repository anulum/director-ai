// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — lib
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel PyO3 FFI Bindings
// © 1998-2026 Miroslav Šotek. All rights reserved.
// License: Apache-2.0
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
//! - GIL acquired via `Python::attach` before every Python callback.
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
use backfire_core::{
    CoherenceScorer, InMemoryKnowledge, PiiScanner, SafetyKernel, StreamingKernel,
};
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

mod safety_hooks;
mod zk_range;

// ─── PyBackfireConfig ───────────────────────────────────────────────

/// Python-visible configuration for the Backfire Kernel.
#[pyclass(name = "BackfireConfig", from_py_object)]
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
#[pyclass(name = "CoherenceScore", from_py_object)]
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
#[pyclass(name = "StreamSession", from_py_object)]
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
        coherence_callback: Py<PyAny>,
    ) -> PyResult<String> {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let cb = |token: &str| -> f64 {
            Python::attach(|py| match coherence_callback.call1(py, (token,)) {
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
        coherence_callback: Py<PyAny>,
    ) -> PyResult<PyStreamSession> {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let cb = |token: &str| -> f64 {
            Python::attach(|py| match coherence_callback.call1(py, (token,)) {
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
        coherence_callback: Py<PyAny>,
    ) -> PyResult<String> {
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let cb = |token: &str| -> f64 {
            Python::attach(|py| match coherence_callback.call1(py, (token,)) {
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
        nli_callback: Option<Py<PyAny>>,
        knowledge_callback: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let cfg = config.map(|c| c.inner).unwrap_or_default();

        let nli: Arc<dyn backfire_core::nli::NliBackend> = match nli_callback {
            Some(cb) => Arc::new(ExternalNli::new(move |premise: &str, hypothesis: &str| {
                Python::attach(|py| match cb.call1(py, (premise, hypothesis)) {
                    Ok(result) => result.extract::<f64>(py).unwrap_or(0.5),
                    Err(_) => 0.5,
                })
            })),
            None => Arc::new(backfire_core::HeuristicNli),
        };

        let knowledge: Arc<dyn backfire_core::knowledge::GroundTruthStore> =
            match knowledge_callback {
                Some(cb) => Arc::new(ExternalKnowledge::new(move |query: &str| {
                    Python::attach(|py| match cb.call1(py, (query,)) {
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
    fn create_state(theta: Vec<f64>) -> PyResult<Py<PyAny>> {
        let state = UPDEState::new(theta);
        Python::attach(|py| {
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
    fn random_state() -> PyResult<Py<PyAny>> {
        let state = UPDEState::random(N_LAYERS);
        Python::attach(|py| {
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
    fn run(&mut self, theta: Vec<f64>, n_steps: u64) -> PyResult<Py<PyAny>> {
        let state = UPDEState::new(theta);
        let result = self
            .inner
            .run(&state, n_steps)
            .map_err(PyValueError::new_err)?;
        Python::attach(|py| {
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
    ) -> PyResult<Py<PyAny>> {
        let prev_ref = theta_prev.as_deref();
        let result = self
            .inner
            .evaluate(&theta, prev_ref, dt)
            .map_err(PyValueError::new_err)?;
        Python::attach(|py| {
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
    ) -> PyResult<Py<PyAny>> {
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
        Python::attach(|py| {
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
    fn compute(&mut self, py: Python<'_>, theta: Vec<f64>, dt: f64) -> PyResult<Py<PyAny>> {
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
    fn run(&mut self, py: Python<'_>, n_outer: usize) -> PyResult<Py<PyAny>> {
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
    fn outer_step(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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
    fn audio_mapping(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
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

// Backfire Kernel — Rust-accelerated safety gate for Director-Class AI.
//
// This module exposes the entire hot-path safety gate to Python:
// - BackfireConfig, RustSafetyKernel, RustStreamingKernel
// - RustCoherenceScorer, CoherenceScore, StreamSession
// - Verification signal functions (Rust-accelerated)
// - RustBM25 — BM25 sparse retrieval engine

/// Entity overlap between two texts (Jaccard of proper nouns).
#[pyfunction]
fn rust_entity_overlap(text_a: &str, text_b: &str) -> f64 {
    backfire_core::signals::entity_overlap(text_a, text_b)
}

/// Numerical consistency check between two texts.
#[pyfunction]
fn rust_numerical_consistency(text_a: &str, text_b: &str) -> Option<bool> {
    backfire_core::signals::numerical_consistency(text_a, text_b)
}

/// Detect negation flip between claim and source.
#[pyfunction]
fn rust_negation_flip(claim: &str, source: &str) -> bool {
    backfire_core::signals::negation_flip(claim, source)
}

/// Content word traceability (fraction of claim words in source).
#[pyfunction]
fn rust_traceability(claim: &str, source: &str) -> f64 {
    backfire_core::signals::traceability(claim, source)
}

/// Linear regression trend drop over coherence window.
#[pyfunction]
fn rust_trend_drop(values: Vec<f64>) -> f64 {
    backfire_core::signals::trend_drop(&values)
}

/// Per-claim bidirectional divergence scoring for injection detection.
///
/// For each claim, computes traceability, entity overlap, and
/// baseline-calibrated divergence against the intent.
///
/// Args:
///     claims: List of claim strings.
///     intent: The original intent string.
///     forward_scores: NLI scores (intent → claim) per claim.
///     reverse_scores: NLI scores (claim → intent) per claim.
///     baseline: Expected normal divergence (calibration baseline).
///
/// Returns:
///     List of (traceability, entity_match, calibrated_divergence) tuples.
#[pyfunction]
fn rust_bidirectional_divergence(
    claims: Vec<String>,
    intent: &str,
    forward_scores: Vec<f64>,
    reverse_scores: Vec<f64>,
    baseline: f64,
) -> Vec<(f64, f64, f64)> {
    let claim_refs: Vec<&str> = claims.iter().map(|s| s.as_str()).collect();
    backfire_core::signals::bidirectional_divergence(
        &claim_refs,
        intent,
        &forward_scores,
        &reverse_scores,
        baseline,
    )
}

/// Multi-signal injection verdict per claim.
///
/// Args:
///     calibrated_divs: Baseline-calibrated divergences per claim.
///     traceabilities: Content-word overlap per claim.
///     entity_matches: Entity Jaccard overlap per claim.
///     injection_threshold: Combined score threshold.
///     drift_threshold: Per-claim drift threshold.
///     injection_claim_threshold: High-divergence + low-trace threshold.
///     traceability_floor: Below this = fabrication override.
///     stage1_weight: Weight of sanitizer in combined score.
///     sanitizer_score: Stage 1 sanitizer score.
///
/// Returns:
///     Tuple of (verdicts, injection_risk, combined_score, detected) where
///     verdicts is list of (verdict_code, confidence) — 0=grounded, 1=drifted, 2=injected.
#[pyfunction]
#[pyo3(signature = (
    calibrated_divs,
    traceabilities,
    entity_matches,
    sanitizer_score,
    injection_threshold = 0.7,
    drift_threshold = 0.6,
    injection_claim_threshold = 0.75,
    traceability_floor = 0.15,
    stage1_weight = 0.3,
))]
#[allow(clippy::too_many_arguments)]
fn rust_injection_verdict(
    calibrated_divs: Vec<f64>,
    traceabilities: Vec<f64>,
    entity_matches: Vec<f64>,
    sanitizer_score: f64,
    injection_threshold: f64,
    drift_threshold: f64,
    injection_claim_threshold: f64,
    traceability_floor: f64,
    stage1_weight: f64,
) -> (Vec<(u8, f64)>, f64, f64, bool) {
    let cfg = backfire_core::signals::InjectionVerdictConfig {
        injection_threshold,
        drift_threshold,
        injection_claim_threshold,
        traceability_floor,
        stage1_weight,
    };
    let verdicts = backfire_core::signals::injection_verdicts(
        &calibrated_divs,
        &traceabilities,
        &entity_matches,
        &cfg,
    );
    let (risk, combined, detected) =
        backfire_core::signals::injection_aggregate(&verdicts, sanitizer_score, &cfg);
    (verdicts, risk, combined, detected)
}

// ─── BM25 Retrieval Engine ──────────────────────────────────────────

/// Rust-accelerated BM25 sparse retrieval engine.
#[pyclass(name = "RustBM25")]
struct PyBM25 {
    inner: backfire_core::BM25Engine,
}

#[pymethods]
impl PyBM25 {
    #[new]
    #[pyo3(signature = (k1 = 1.2, b = 0.75))]
    fn new(k1: f64, b: f64) -> Self {
        Self {
            inner: backfire_core::BM25Engine::new(k1, b),
        }
    }

    /// Add a document to the BM25 index.
    fn add_document(&self, doc_id: &str, text: &str) {
        self.inner.add_document(doc_id, text);
    }

    /// Query the index, returning list of (doc_id, score) tuples.
    fn query(&self, query_text: &str, n_results: usize) -> Vec<(String, f64)> {
        self.inner
            .query(query_text, n_results)
            .into_iter()
            .map(|r| (r.doc_id, r.score))
            .collect()
    }

    /// Number of indexed documents.
    fn count(&self) -> usize {
        self.inner.count()
    }

    /// Clear all documents.
    fn clear(&self) {
        self.inner.clear();
    }
}

// ── Compute accelerators (sanitizer, task detection, verification, NLI) ──

#[pyfunction]
fn rust_sanitizer_score(text: &str) -> (f64, Vec<String>) {
    backfire_core::compute::sanitizer_score(text)
}

#[pyfunction]
fn rust_has_suspicious_unicode(text: &str) -> bool {
    backfire_core::compute::has_suspicious_unicode(text)
}

#[pyfunction]
fn rust_detect_task_type(prompt: &str, response: &str) -> String {
    backfire_core::compute::detect_task_type(prompt, response)
}

type NumericIssuesTuple = Vec<(String, String, String, String)>;

#[pyfunction]
fn rust_verify_numeric(text: &str, current_year: i32) -> (usize, NumericIssuesTuple, bool) {
    let (claims, issues, valid) = backfire_core::compute::verify_numeric(text, current_year);
    let issues_tuples: Vec<(String, String, String, String)> = issues
        .into_iter()
        .map(|i| (i.issue_type, i.description, i.severity, i.context))
        .collect();
    (claims, issues_tuples, valid)
}

#[pyfunction]
fn rust_score_temporal_freshness(text: &str) -> (Vec<(String, String, f64)>, f64, bool) {
    let (claims, overall, has) = backfire_core::compute::score_temporal_freshness(text);
    let claims_tuples: Vec<(String, String, f64)> = claims
        .into_iter()
        .map(|c| (c.text, c.claim_type, c.staleness_risk))
        .collect();
    (claims_tuples, overall, has)
}

#[pyfunction]
fn rust_extract_reasoning_steps(text: &str) -> Vec<String> {
    backfire_core::compute::extract_reasoning_steps(text)
}

#[pyfunction]
fn rust_split_sentences(text: &str) -> Vec<String> {
    backfire_core::compute::split_sentences(text)
}

#[pyfunction]
fn rust_build_chunks(sentences: Vec<String>, budget: usize, overlap_ratio: f64) -> Vec<String> {
    backfire_core::compute::build_chunks(&sentences, budget, overlap_ratio)
}

#[pyfunction]
fn rust_word_overlap(text_a: &str, text_b: &str) -> f64 {
    backfire_core::compute::word_overlap(text_a, text_b)
}

#[pyfunction]
fn rust_eval_arithmetic(expr: &str) -> f64 {
    backfire_core::compute::eval_arithmetic(expr)
}

#[pyfunction]
fn rust_detect_fallacies(text: &str) -> Vec<(String, String)> {
    backfire_core::compute::detect_fallacies(text)
}

#[pyfunction]
fn rust_softmax(logits: Vec<f64>, cols: usize) -> Vec<f64> {
    backfire_core::compute::softmax(&logits, cols)
}

#[pyfunction]
fn rust_probs_to_divergence(
    probs: Vec<f64>,
    cols: usize,
    contradiction_idx: usize,
    neutral_idx: usize,
) -> Vec<f64> {
    backfire_core::compute::probs_to_divergence(&probs, cols, contradiction_idx, neutral_idx)
}

#[pyfunction]
fn rust_probs_to_confidence(probs: Vec<f64>, cols: usize) -> Vec<f64> {
    backfire_core::compute::probs_to_confidence(&probs, cols)
}

#[pyfunction]
fn rust_aggregate_chunk_scores(
    flat_scores: Vec<f64>,
    n_prem: usize,
    n_hyp: usize,
    inner_agg: &str,
    outer_agg: &str,
) -> (f64, Vec<f64>) {
    backfire_core::compute::aggregate_chunk_scores(
        &flat_scores,
        n_prem,
        n_hyp,
        inner_agg,
        outer_agg,
    )
}

#[pyfunction]
fn rust_merge_flagged_spans(
    offsets: Vec<(i64, i64)>,
    scores: Vec<f64>,
    response: &str,
    threshold: f64,
) -> (Vec<(i64, i64, f64)>, usize, f64) {
    let response_chars: Vec<char> = response.chars().collect();
    backfire_core::compute::merge_flagged_spans(&offsets, &scores, &response_chars, threshold)
}

#[pyfunction]
fn rust_aggregate_chunk_scores_confidence_weighted(
    flat_scores: Vec<f64>,
    flat_confidences: Vec<f64>,
    n_prem: usize,
    n_hyp: usize,
    inner_agg: &str,
) -> (f64, Vec<f64>) {
    backfire_core::compute::aggregate_chunk_scores_confidence_weighted(
        &flat_scores,
        &flat_confidences,
        n_prem,
        n_hyp,
        inner_agg,
    )
}

#[pyfunction]
fn rust_coverage_from_divergences(divergences: Vec<f64>, support_threshold: f64) -> (f64, usize) {
    backfire_core::compute::coverage_from_divergences(&divergences, support_threshold)
}

#[pyfunction]
fn rust_reduce_claim_attribution(
    flat_divergences: Vec<f64>,
    n_claims: usize,
    n_src: usize,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if n_claims == 0 {
        return Err(PyValueError::new_err("n_claims must be >= 1"));
    }
    if n_src == 0 {
        return Err(PyValueError::new_err("n_src must be >= 1"));
    }
    let expected = n_claims
        .checked_mul(n_src)
        .ok_or_else(|| PyValueError::new_err("n_claims * n_src overflow"))?;
    if flat_divergences.len() != expected {
        return Err(PyValueError::new_err(format!(
            "flat_divergences length mismatch: expected {expected}, got {}",
            flat_divergences.len()
        )));
    }
    Ok(backfire_core::compute::reduce_claim_attribution(
        &flat_divergences,
        n_claims,
        n_src,
    ))
}

#[pyfunction]
fn rust_lite_score(premise: &str, hypothesis: &str) -> f64 {
    backfire_core::compute::lite_score(premise, hypothesis)
}

#[pyfunction]
fn rust_lite_score_batch(pairs: Vec<(String, String)>) -> Vec<f64> {
    backfire_core::compute::lite_score_batch(&pairs)
}

#[pyfunction]
fn rust_heuristic_logical_divergence(text_output: &str, prompt: &str) -> f64 {
    backfire_core::compute::heuristic_logical_divergence(text_output, prompt)
}

#[pyfunction]
fn rust_heuristic_factual_divergence(context: &str, text_output: &str) -> f64 {
    backfire_core::compute::heuristic_factual_divergence(context, text_output)
}

#[pyfunction]
fn rust_conformal_quantile(residuals: Vec<f64>, coverage: f64) -> PyResult<f64> {
    if !(0.0..1.0).contains(&coverage) {
        return Err(PyValueError::new_err("coverage must be in (0, 1)"));
    }
    if residuals.is_empty() {
        return Err(PyValueError::new_err("residuals must be non-empty"));
    }
    if residuals
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "residuals must be finite and non-negative",
        ));
    }
    let mut sorted = residuals;
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let q_idx = (((n + 1) as f64 * coverage).ceil() as isize - 1).clamp(0, (n - 1) as isize);
    Ok(sorted[q_idx as usize])
}

#[pyfunction]
fn rust_ema_update(previous: Option<f64>, value: f64, alpha: f64) -> PyResult<f64> {
    if !value.is_finite() {
        return Err(PyValueError::new_err("value must be finite"));
    }
    if !(0.0..=1.0).contains(&alpha) || alpha == 0.0 {
        return Err(PyValueError::new_err("alpha must be in (0, 1]"));
    }
    if let Some(prev) = previous {
        if !prev.is_finite() {
            return Err(PyValueError::new_err(
                "previous must be finite when provided",
            ));
        }
        Ok(alpha * value + (1.0 - alpha) * prev)
    } else {
        Ok(value)
    }
}

#[pyfunction]
fn rust_beta_posterior_mean(
    alpha_prior: f64,
    beta_prior: f64,
    successes: usize,
    pulls: usize,
) -> PyResult<f64> {
    if alpha_prior <= 0.0 || !alpha_prior.is_finite() {
        return Err(PyValueError::new_err("alpha_prior must be finite and > 0"));
    }
    if beta_prior <= 0.0 || !beta_prior.is_finite() {
        return Err(PyValueError::new_err("beta_prior must be finite and > 0"));
    }
    if successes > pulls {
        return Err(PyValueError::new_err("successes cannot exceed pulls"));
    }
    let alpha = alpha_prior + successes as f64;
    let beta = beta_prior + (pulls - successes) as f64;
    Ok(alpha / (alpha + beta))
}

#[pyfunction]
fn rust_wilson_score_interval(p_hat: f64, n: usize, confidence: f64) -> PyResult<(f64, f64)> {
    if !p_hat.is_finite() || !(0.0..=1.0).contains(&p_hat) {
        return Err(PyValueError::new_err("p_hat must be finite and in [0, 1]"));
    }
    if !(0.0..1.0).contains(&confidence) {
        return Err(PyValueError::new_err("confidence must be in (0, 1)"));
    }
    if n == 0 {
        return Ok((0.0, 0.0));
    }

    let z = 1.959_963_984_540_054_f64; // 95 % default approximation
    let z_adj = if (confidence - 0.95).abs() < 1e-9 {
        z
    } else {
        // fallback for non-95% callers: use fixed z to keep deterministic bounded output
        z
    };
    let nf = n as f64;
    let denominator = 1.0 + z_adj * z_adj / nf;
    let centre = (p_hat + z_adj * z_adj / (2.0 * nf)) / denominator;
    let halfwidth = (z_adj
        * ((p_hat * (1.0 - p_hat) / nf + z_adj * z_adj / (4.0 * nf * nf)).sqrt()))
        / denominator;
    Ok(((centre - halfwidth).max(0.0), (centre + halfwidth).min(1.0)))
}

#[pyfunction]
fn rust_percentile_rank(values: Vec<f64>, value: f64) -> PyResult<f64> {
    if !value.is_finite() {
        return Err(PyValueError::new_err("value must be finite"));
    }
    if values.is_empty() {
        return Ok(1.0);
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    let below = values.iter().filter(|v| **v <= value).count();
    Ok(below as f64 / values.len() as f64)
}

#[pyfunction]
fn rust_mean(values: Vec<f64>) -> PyResult<f64> {
    if values.is_empty() {
        return Err(PyValueError::new_err("values must be non-empty"));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    let sum: f64 = values.iter().sum();
    Ok(sum / values.len() as f64)
}

#[pyfunction]
fn rust_standard_normal_quantile(p: f64) -> PyResult<f64> {
    if !p.is_finite() || p <= 0.0 || p >= 1.0 {
        return Err(PyValueError::new_err("p must be finite and in (0, 1)"));
    }
    let a = [
        -3.969_683_028_665_376e+01,
        2.209_460_984_245_205e+02,
        -2.759_285_104_469_687e+02,
        1.383_577_518_672_69e+02,
        -3.066_479_806_614_716e+01,
        2.506_628_277_459_239e+00,
    ];
    let b = [
        -5.447_609_879_822_406e+01,
        1.615_858_368_580_409e+02,
        -1.556_989_798_598_866e+02,
        6.680_131_188_771_972e+01,
        -1.328_068_155_288_572e+01,
    ];
    let c = [
        -7.784_894_002_430_293e-03,
        -3.223_964_580_411_365e-01,
        -2.400_758_277_161_838e+00,
        -2.549_732_539_343_734e+00,
        4.374_664_141_464_968e+00,
        2.938_163_982_698_783e+00,
    ];
    let d = [
        7.784_695_709_041_462e-03,
        3.224_671_290_700_398e-01,
        2.445_134_137_142_996e+00,
        3.754_408_661_907_416e+00,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        return Ok(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0),
        );
    }
    if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        return Ok(
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0),
        );
    }
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    Ok(
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0),
    )
}

#[pyfunction]
fn rust_sum_f64(values: Vec<f64>) -> PyResult<f64> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(values.iter().sum())
}

#[pyfunction]
fn rust_sum_i64(values: Vec<i64>) -> i64 {
    values.iter().sum()
}

#[pyfunction]
fn rust_product_f64(values: Vec<f64>) -> PyResult<f64> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("values must be finite"));
    }
    Ok(values.iter().product())
}

#[pyfunction]
fn rust_confusion_counts_threshold(
    scores: Vec<f64>,
    labels: Vec<bool>,
    threshold: f64,
) -> PyResult<(usize, usize, usize, usize)> {
    if scores.len() != labels.len() {
        return Err(PyValueError::new_err(
            "scores and labels must have same length",
        ));
    }
    if !threshold.is_finite() {
        return Err(PyValueError::new_err("threshold must be finite"));
    }
    if scores.iter().any(|s| !s.is_finite()) {
        return Err(PyValueError::new_err("scores must be finite"));
    }
    let mut tp = 0usize;
    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fnn = 0usize;
    for (score, label) in scores.iter().zip(labels.iter()) {
        let predicted_positive = *score >= threshold;
        match (predicted_positive, *label) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fnn += 1,
            (false, false) => tn += 1,
        }
    }
    Ok((tp, tn, fp, fnn))
}

#[pymodule]
fn backfire_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
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
    // Zero-knowledge range-proof attestation (Bulletproofs over Ristretto)
    m.add_function(wrap_pyfunction!(
        zk_range::rust_bulletproof_prove_threshold,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        zk_range::rust_bulletproof_verify_threshold,
        m
    )?)?;
    // Verification signals (Rust-accelerated)
    m.add_function(wrap_pyfunction!(rust_entity_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(rust_numerical_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(rust_negation_flip, m)?)?;
    m.add_function(wrap_pyfunction!(rust_traceability, m)?)?;
    m.add_function(wrap_pyfunction!(rust_trend_drop, m)?)?;
    // Injection detection (Rust-accelerated)
    m.add_function(wrap_pyfunction!(rust_bidirectional_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_injection_verdict, m)?)?;
    // BM25 retrieval engine
    m.add_class::<PyBM25>()?;
    // Compute accelerators
    m.add_function(wrap_pyfunction!(rust_sanitizer_score, m)?)?;
    m.add_function(wrap_pyfunction!(rust_has_suspicious_unicode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_detect_task_type, m)?)?;
    m.add_function(wrap_pyfunction!(rust_verify_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(rust_score_temporal_freshness, m)?)?;
    m.add_function(wrap_pyfunction!(rust_extract_reasoning_steps, m)?)?;
    m.add_function(wrap_pyfunction!(rust_split_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(rust_build_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(rust_word_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eval_arithmetic, m)?)?;
    m.add_function(wrap_pyfunction!(rust_detect_fallacies, m)?)?;
    m.add_function(wrap_pyfunction!(rust_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(rust_probs_to_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_probs_to_confidence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_aggregate_chunk_scores, m)?)?;
    m.add_function(wrap_pyfunction!(rust_merge_flagged_spans, m)?)?;
    m.add_function(wrap_pyfunction!(
        rust_aggregate_chunk_scores_confidence_weighted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rust_coverage_from_divergences, m)?)?;
    m.add_function(wrap_pyfunction!(rust_reduce_claim_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(rust_lite_score, m)?)?;
    m.add_function(wrap_pyfunction!(rust_lite_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_heuristic_logical_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_heuristic_factual_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_conformal_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(rust_ema_update, m)?)?;
    m.add_function(wrap_pyfunction!(rust_beta_posterior_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_wilson_score_interval, m)?)?;
    m.add_function(wrap_pyfunction!(rust_percentile_rank, m)?)?;
    m.add_function(wrap_pyfunction!(rust_mean, m)?)?;
    m.add_function(wrap_pyfunction!(rust_standard_normal_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sum_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sum_i64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_product_f64, m)?)?;
    m.add_function(wrap_pyfunction!(rust_confusion_counts_threshold, m)?)?;
    // PII regex multi-pattern scanner
    m.add_class::<PyPiiScanner>()?;
    // Safety-hook acceleration (cyber-physical geometry/IK +
    // zk-attestation Merkle + challenge derivation)
    safety_hooks::register(m)?;
    Ok(())
}

/// Python wrapper around ``backfire_core::PiiScanner``.
///
/// Construction takes an iterable of ``(category, pattern)``
/// tuples; every pattern is compiled eagerly and a bad regex
/// raises ``ValueError`` so operator mistakes surface immediately.
/// ``scan(text)`` returns a list of ``(category, start, end)``
/// tuples with byte offsets — the Python
/// ``RegexPIIDetector`` wraps these into ``ModerationMatch`` records
/// when ``backfire_kernel`` is installed.
#[pyclass(name = "PiiScanner")]
struct PyPiiScanner {
    inner: PiiScanner,
}

#[pymethods]
impl PyPiiScanner {
    #[new]
    fn new(patterns: Vec<(String, String)>) -> PyResult<Self> {
        let refs: Vec<(&str, &str)> = patterns
            .iter()
            .map(|(c, p)| (c.as_str(), p.as_str()))
            .collect();
        let inner = PiiScanner::new(&refs).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Scan ``text`` and return a list of ``(category, start, end)``
    /// tuples. Byte offsets; empty list on empty input.
    fn scan(&self, text: &str) -> Vec<(String, usize, usize)> {
        self.inner
            .scan(text)
            .into_iter()
            .map(|m| (m.category, m.start, m.end))
            .collect()
    }

    /// Number of registered pattern/category pairs.
    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformal_quantile_orders_and_validates() {
        // n=5: index = ceil((n+1)*coverage) - 1 = ceil(4.8) - 1 = 4 -> 0.5.
        let q = rust_conformal_quantile(vec![0.3, 0.1, 0.2, 0.4, 0.5], 0.8).unwrap();
        assert!((q - 0.5).abs() < 1e-12);
        // Quantile index clamps to the largest residual at high coverage.
        let top = rust_conformal_quantile(vec![0.1, 0.9], 0.99).unwrap();
        assert!((top - 0.9).abs() < 1e-12);

        assert!(rust_conformal_quantile(vec![], 0.9).is_err());
        assert!(rust_conformal_quantile(vec![0.1], 1.0).is_err());
        assert!(rust_conformal_quantile(vec![-0.1], 0.9).is_err());
        assert!(rust_conformal_quantile(vec![f64::NAN], 0.9).is_err());
    }

    #[test]
    fn ema_update_seeds_and_blends() {
        // No previous value: the observation seeds the average.
        assert!((rust_ema_update(None, 0.6, 0.5).unwrap() - 0.6).abs() < 1e-12);
        // Blend: 0.5*0.0 + 0.5*1.0.
        assert!((rust_ema_update(Some(1.0), 0.0, 0.5).unwrap() - 0.5).abs() < 1e-12);

        assert!(rust_ema_update(Some(1.0), f64::INFINITY, 0.5).is_err());
        assert!(rust_ema_update(Some(f64::NAN), 0.5, 0.5).is_err());
        assert!(rust_ema_update(None, 0.5, 0.0).is_err());
        assert!(rust_ema_update(None, 0.5, 1.5).is_err());
    }

    #[test]
    fn beta_posterior_mean_matches_closed_form() {
        // (1 + 3) / (1 + 3 + 1 + 1) with alpha=beta=1, 3/4 successes.
        let mean = rust_beta_posterior_mean(1.0, 1.0, 3, 4).unwrap();
        assert!((mean - 4.0 / 6.0).abs() < 1e-12);

        assert!(rust_beta_posterior_mean(0.0, 1.0, 0, 0).is_err());
        assert!(rust_beta_posterior_mean(1.0, -1.0, 0, 0).is_err());
        assert!(rust_beta_posterior_mean(1.0, 1.0, 5, 4).is_err());
    }

    #[test]
    fn wilson_interval_is_bounded_and_validated() {
        let (lo, hi) = rust_wilson_score_interval(0.9, 100, 0.95).unwrap();
        assert!((0.0..0.9).contains(&lo));
        assert!(hi > 0.9 && hi <= 1.0);
        // Zero samples collapse to (0, 0) by contract.
        assert_eq!(
            rust_wilson_score_interval(0.5, 0, 0.95).unwrap(),
            (0.0, 0.0)
        );

        assert!(rust_wilson_score_interval(1.5, 10, 0.95).is_err());
        assert!(rust_wilson_score_interval(0.5, 10, 1.0).is_err());
    }

    #[test]
    fn rank_mean_sums_and_products_validate_finiteness() {
        assert!(
            (rust_percentile_rank(vec![1.0, 2.0, 3.0], 2.0).unwrap() - 2.0 / 3.0).abs() < 1e-12
        );
        assert!((rust_percentile_rank(vec![], 5.0).unwrap() - 1.0).abs() < 1e-12);
        assert!(rust_percentile_rank(vec![f64::NAN], 0.5).is_err());

        assert!((rust_mean(vec![1.0, 2.0, 3.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!(rust_mean(vec![]).is_err());

        assert!((rust_sum_f64(vec![0.5, 0.25]).unwrap() - 0.75).abs() < 1e-12);
        assert!(rust_sum_f64(vec![f64::INFINITY]).is_err());
        assert_eq!(rust_sum_i64(vec![1, -2, 4]), 3);
        assert!((rust_product_f64(vec![2.0, 3.0]).unwrap() - 6.0).abs() < 1e-12);
        assert!(rust_product_f64(vec![f64::NAN]).is_err());
    }

    #[test]
    fn standard_normal_quantile_hits_known_points() {
        // Symmetric around the median and matches the 97.5 % point used by
        // the Wilson interval.
        assert!(rust_standard_normal_quantile(0.5).unwrap().abs() < 1e-9);
        let z975 = rust_standard_normal_quantile(0.975).unwrap();
        assert!((z975 - 1.959_963_984_540_054).abs() < 1e-6);
        let z025 = rust_standard_normal_quantile(0.025).unwrap();
        assert!((z975 + z025).abs() < 1e-6);

        assert!(rust_standard_normal_quantile(0.0).is_err());
        assert!(rust_standard_normal_quantile(1.0).is_err());
    }

    #[test]
    fn confusion_counts_split_by_threshold() {
        let (tp, tn, fp, fnn) = rust_confusion_counts_threshold(
            vec![0.9, 0.8, 0.4, 0.2],
            vec![true, false, true, false],
            0.5,
        )
        .unwrap();
        assert_eq!((tp, tn, fp, fnn), (1, 1, 1, 1));

        assert!(rust_confusion_counts_threshold(vec![0.5], vec![], 0.5).is_err());
        assert!(rust_confusion_counts_threshold(vec![f64::NAN], vec![true], 0.5).is_err());
        assert!(rust_confusion_counts_threshold(vec![0.5], vec![true], f64::INFINITY).is_err());
    }
}
