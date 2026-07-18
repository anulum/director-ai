// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::physics
//! Physics-engine bindings: UPDE stepper, SEC functional, L16 controller.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use backfire_physics::{
    l16_closure::L16CostInputs, params::N_LAYERS, L16Controller, SECFunctional, UPDEState,
    UPDEStepper,
};

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

/// Register the physics-engine classes on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUPDEStepper>()?;
    m.add_class::<PySECFunctional>()?;
    m.add_class::<PyL16Controller>()?;
    Ok(())
}
