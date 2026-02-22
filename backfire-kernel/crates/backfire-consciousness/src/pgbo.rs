// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — PGBO (Phase→Geometry Bridge Operator)
// Mirrors: src/director_ai/research/consciousness/pgbo.py
// ─────────────────────────────────────────────────────────────────────
//! Phase→Geometry Bridge Operator (PGBO).
//!
//! Converts coherent phase dynamics into a symmetric rank-2 tensor field
//! h_munu that modulates propagation and coupling.
//!
//! Key equations:
//!   u_mu   = dphi_mu - alpha * A_mu       (covariant phase-flow drive)
//!   h_munu = kappa * u_mu outer u_mu      (induced geometry proxy)
//!
//! Properties:
//!   - h_munu is symmetric and positive semi-definite
//!   - grows with coherence intensity (proportional to |u|^2)
//!   - optional traceless projection isolates shear effects

use serde::{Deserialize, Serialize};

/// PGBO configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PGBOConfig {
    pub alpha: f64,
    pub kappa: f64,
    pub alpha_min: f64,
    pub alpha_max: f64,
    pub kappa_min: f64,
    pub kappa_max: f64,
    pub u_cap: f64,
    pub traceless: bool,
}

impl Default for PGBOConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            kappa: 0.3,
            alpha_min: 0.0,
            alpha_max: 5.0,
            kappa_min: 0.0,
            kappa_max: 5.0,
            u_cap: 10.0,
            traceless: false,
        }
    }
}

/// PGBO engine state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PGBOState {
    pub alpha: f64,
    pub kappa: f64,
    pub u_norm: f64,
    pub h_trace: f64,
    pub h_frob: f64,
    pub traceless: bool,
    pub d: usize,
}

/// Phase→Geometry Bridge Operator engine.
///
/// Mirrors `PGBOEngine` from `pgbo.py`.
pub struct PGBOEngine {
    d: usize,
    pub cfg: PGBOConfig,
    // Boundary potential (zero by default)
    a_mu: Vec<f64>,
    // Pre-allocated outputs
    pub u_mu: Vec<f64>,
    pub h_munu: Vec<f64>, // flattened D×D, row-major
    // Cached scalars
    pub u_norm: f64,
    pub h_trace: f64,
    pub h_frob: f64,
    // Previous theta for finite-difference gradient
    prev_theta: Option<Vec<f64>>,
}

impl PGBOEngine {
    pub fn new(n: usize, config: PGBOConfig) -> Self {
        Self {
            d: n,
            cfg: config,
            a_mu: vec![0.0; n],
            u_mu: vec![0.0; n],
            h_munu: vec![0.0; n * n],
            u_norm: 0.0,
            h_trace: 0.0,
            h_frob: 0.0,
            prev_theta: None,
        }
    }

    pub fn default_params(n: usize) -> Self {
        Self::new(n, PGBOConfig::default())
    }

    /// Set the boundary injection potential (L10 handle).
    pub fn set_boundary_potential(&mut self, a_mu: &[f64]) {
        let len = a_mu.len().min(self.d);
        self.a_mu[..len].copy_from_slice(&a_mu[..len]);
    }

    /// Compute the PGBO from current phases.
    ///
    /// Returns (u_mu, h_munu_flat) where h_munu is D×D row-major.
    pub fn compute(&mut self, theta: &[f64], dt: f64) -> (&[f64], &[f64]) {
        let d = self.d;

        // Finite-difference phase gradient with wrapping
        if let Some(ref prev) = self.prev_theta {
            let tau = std::f64::consts::TAU;
            let pi = std::f64::consts::PI;
            for i in 0..d.min(theta.len()).min(prev.len()) {
                let raw_diff = theta[i] - prev[i];
                let wrapped = ((raw_diff + pi) % tau + tau) % tau - pi;
                self.u_mu[i] = wrapped / dt.max(1e-12);
            }
        } else {
            for v in self.u_mu.iter_mut() {
                *v = 0.0;
            }
        }

        self.prev_theta = Some(theta.to_vec());

        // Covariant drive: u_mu -= alpha * A_mu
        for i in 0..d {
            self.u_mu[i] -= self.cfg.alpha * self.a_mu[i];
        }

        // Saturation cap
        self.u_norm = self.u_mu.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if self.u_norm > 1e-12 {
            let scale = 1.0 / (1.0 + self.u_norm / self.cfg.u_cap);
            for v in self.u_mu.iter_mut() {
                *v *= scale;
            }
            self.u_norm *= scale;
        }

        // Induced metric perturbation: h_munu = kappa * u_mu outer u_mu
        for i in 0..d {
            for j in 0..d {
                self.h_munu[i * d + j] = self.cfg.kappa * self.u_mu[i] * self.u_mu[j];
            }
        }

        // Optional traceless projection: h -= (Tr(h) / D) * I
        if self.cfg.traceless {
            let trace: f64 = (0..d).map(|i| self.h_munu[i * d + i]).sum();
            let corr = trace / d as f64;
            for i in 0..d {
                self.h_munu[i * d + i] -= corr;
            }
            self.h_trace = 0.0; // traceless by construction
        } else {
            self.h_trace = (0..d).map(|i| self.h_munu[i * d + i]).sum();
        }

        self.h_frob = self.h_munu.iter().map(|&v| v * v).sum::<f64>().sqrt();

        (&self.u_mu, &self.h_munu)
    }

    /// Return serialisable engine state.
    pub fn get_state(&self) -> PGBOState {
        PGBOState {
            alpha: self.cfg.alpha,
            kappa: self.cfg.kappa,
            u_norm: self.u_norm,
            h_trace: self.h_trace,
            h_frob: self.h_frob,
            traceless: self.cfg.traceless,
            d: self.d,
        }
    }

    /// Reset engine state.
    pub fn reset(&mut self) {
        for v in self.u_mu.iter_mut() { *v = 0.0; }
        for v in self.h_munu.iter_mut() { *v = 0.0; }
        for v in self.a_mu.iter_mut() { *v = 0.0; }
        self.u_norm = 0.0;
        self.h_trace = 0.0;
        self.h_frob = 0.0;
        self.prev_theta = None;
    }
}

/// Standalone phase→geometry bridge (single-shot, no state).
///
/// `dphi_mu` and `a_mu` are 1-D vectors of length D.
pub fn phase_geometry_bridge(
    dphi_mu: &[f64],
    a_mu: &[f64],
    alpha: f64,
    kappa: f64,
    u_cap: f64,
    traceless: bool,
) -> (Vec<f64>, Vec<f64>) {
    let d = dphi_mu.len();

    // u_mu = dphi_mu - alpha * A_mu
    let mut u_mu: Vec<f64> = dphi_mu.iter()
        .zip(a_mu.iter())
        .map(|(&dp, &a)| dp - alpha * a)
        .collect();

    // Saturation
    let u_norm = u_mu.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if u_norm > 1e-12 {
        let scale = 1.0 / (1.0 + u_norm / u_cap);
        for v in u_mu.iter_mut() {
            *v *= scale;
        }
    }

    // Outer product: h_munu = kappa * u outer u
    let mut h_munu = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..d {
            h_munu[i * d + j] = kappa * u_mu[i] * u_mu[j];
        }
    }

    // Traceless projection
    if traceless {
        let trace: f64 = (0..d).map(|i| h_munu[i * d + i]).sum();
        let corr = trace / d as f64;
        for i in 0..d {
            h_munu[i * d + i] -= corr;
        }
    }

    (u_mu, h_munu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pgbo_engine_initial_state() {
        let engine = PGBOEngine::default_params(4);
        let state = engine.get_state();
        assert!((state.u_norm).abs() < 1e-9);
        assert!((state.h_frob).abs() < 1e-9);
        assert_eq!(state.d, 4);
    }

    #[test]
    fn test_pgbo_engine_first_compute_zero() {
        let mut engine = PGBOEngine::default_params(4);
        let theta = vec![0.1, 0.2, 0.3, 0.4];
        engine.compute(&theta, 0.01);
        // First call: no prev_theta, so u_mu = -alpha * A_mu = 0
        assert!(engine.u_norm < 1e-9, "First compute should give u_norm~0, got {}", engine.u_norm);
    }

    #[test]
    fn test_pgbo_engine_second_compute_nonzero() {
        let mut engine = PGBOEngine::default_params(4);
        let theta1 = vec![0.1, 0.2, 0.3, 0.4];
        engine.compute(&theta1, 0.01);
        let theta2 = vec![0.2, 0.3, 0.4, 0.5];
        engine.compute(&theta2, 0.01);
        assert!(engine.u_norm > 0.0, "Second compute should give u_norm>0");
        assert!(engine.h_frob > 0.0, "h_frob should be positive");
    }

    #[test]
    fn test_pgbo_h_munu_symmetric() {
        let mut engine = PGBOEngine::default_params(4);
        let theta1 = vec![0.1, 0.2, 0.3, 0.4];
        engine.compute(&theta1, 0.01);
        let theta2 = vec![0.5, 0.1, 0.7, 0.2];
        engine.compute(&theta2, 0.01);

        let d = 4;
        for i in 0..d {
            for j in 0..d {
                let hij = engine.h_munu[i * d + j];
                let hji = engine.h_munu[j * d + i];
                assert!(
                    (hij - hji).abs() < 1e-12,
                    "h[{i},{j}]={hij} != h[{j},{i}]={hji}"
                );
            }
        }
    }

    #[test]
    fn test_pgbo_h_munu_psd() {
        let mut engine = PGBOEngine::default_params(4);
        let theta1 = vec![0.1, 0.2, 0.3, 0.4];
        engine.compute(&theta1, 0.01);
        let theta2 = vec![0.5, 0.1, 0.7, 0.2];
        engine.compute(&theta2, 0.01);

        // h = kappa * u outer u is rank-1 PSD. Check x^T h x >= 0 for random x.
        let d = 4;
        let x = vec![1.0, -0.5, 0.3, 0.7];
        let mut xhx = 0.0;
        for i in 0..d {
            for j in 0..d {
                xhx += x[i] * engine.h_munu[i * d + j] * x[j];
            }
        }
        assert!(xhx >= -1e-12, "h should be PSD, got x^T h x = {xhx}");
    }

    #[test]
    fn test_pgbo_traceless_mode() {
        let mut config = PGBOConfig::default();
        config.traceless = true;
        let mut engine = PGBOEngine::new(4, config);
        let theta1 = vec![0.1, 0.2, 0.3, 0.4];
        engine.compute(&theta1, 0.01);
        let theta2 = vec![0.5, 0.1, 0.7, 0.2];
        engine.compute(&theta2, 0.01);

        let d = 4;
        let trace: f64 = (0..d).map(|i| engine.h_munu[i * d + i]).sum();
        assert!(trace.abs() < 1e-10, "Traceless h should have |Tr|<1e-10, got {trace}");
    }

    #[test]
    fn test_pgbo_saturation() {
        let mut config = PGBOConfig::default();
        config.u_cap = 1.0;
        let mut engine = PGBOEngine::new(4, config);
        let theta1 = vec![0.0, 0.0, 0.0, 0.0];
        engine.compute(&theta1, 0.01);
        // Large phase jump
        let theta2 = vec![3.0, 3.0, 3.0, 3.0];
        engine.compute(&theta2, 0.01);
        // u_norm should be bounded by the saturation mechanism
        assert!(engine.u_norm < 100.0, "u_norm should be saturated, got {}", engine.u_norm);
    }

    #[test]
    fn test_pgbo_reset() {
        let mut engine = PGBOEngine::default_params(4);
        let theta1 = vec![0.1, 0.2, 0.3, 0.4];
        engine.compute(&theta1, 0.01);
        let theta2 = vec![0.5, 0.1, 0.7, 0.2];
        engine.compute(&theta2, 0.01);
        engine.reset();
        assert!(engine.u_norm < 1e-9);
        assert!(engine.h_frob < 1e-9);
    }

    #[test]
    fn test_phase_geometry_bridge_standalone() {
        let dphi = vec![1.0, 0.5, -0.3, 0.2];
        let a_mu = vec![0.1, 0.0, 0.2, 0.0];
        let (u, h) = phase_geometry_bridge(&dphi, &a_mu, 0.5, 0.3, 10.0, false);
        assert_eq!(u.len(), 4);
        assert_eq!(h.len(), 16);
        // h should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert!((h[i * 4 + j] - h[j * 4 + i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_phase_geometry_bridge_traceless() {
        let dphi = vec![1.0, 0.5, -0.3, 0.2];
        let a_mu = vec![0.0; 4];
        let (_, h) = phase_geometry_bridge(&dphi, &a_mu, 0.0, 0.3, 10.0, true);
        let trace: f64 = (0..4).map(|i| h[i * 4 + i]).sum();
        assert!(trace.abs() < 1e-10, "Traceless bridge should have |Tr|<1e-10, got {trace}");
    }

    #[test]
    fn test_boundary_potential() {
        let mut engine = PGBOEngine::default_params(4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        engine.set_boundary_potential(&a);
        let theta1 = vec![0.0; 4];
        engine.compute(&theta1, 0.01);
        // First compute: u_mu = 0 - alpha*A_mu = -0.5*[1,2,3,4]
        // After saturation, u_norm should be > 0
        assert!(engine.u_norm > 0.0, "Boundary potential should create non-zero u");
    }
}
