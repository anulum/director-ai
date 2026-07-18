// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::ssgf
//! SSGF geometry-engine binding: outer-cycle orchestrator.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use backfire_ssgf::{SSGFConfig, SSGFEngine};

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

/// Register the SSGF geometry engine on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySSGFEngine>()?;
    Ok(())
}
