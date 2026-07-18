// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::core_gate
//! Core safety-gate bindings: config, score/session types, kernels,
//! and the coherence scorer.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use backfire_core::knowledge::ExternalKnowledge;
use backfire_core::nli::ExternalNli;
use backfire_core::{CoherenceScorer, InMemoryKnowledge, SafetyKernel, StreamingKernel};
use backfire_types::score::CoherenceScore;
use backfire_types::{BackfireConfig, StreamSession};

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

/// Register the core safety-gate classes on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBackfireConfig>()?;
    m.add_class::<PyCoherenceScore>()?;
    m.add_class::<PyStreamSession>()?;
    m.add_class::<PySafetyKernel>()?;
    m.add_class::<PyStreamingKernel>()?;
    m.add_class::<PyCoherenceScorer>()?;
    Ok(())
}
