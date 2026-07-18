// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::observers
//! Boundary-observer bindings: TCBO observer/controller and PGBO engine.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use backfire_observers::{
    PGBOConfig, PGBOEngine, TCBOConfig, TCBOController, TCBOControllerConfig, TCBOObserver,
};

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

/// Register the boundary-observer classes on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTCBOObserver>()?;
    m.add_class::<PyTCBOController>()?;
    m.add_class::<PyPGBOEngine>()?;
    Ok(())
}
