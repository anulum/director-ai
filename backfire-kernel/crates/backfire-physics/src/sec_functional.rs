// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SEC Functional (Lyapunov Stability)
// Mirrors: src/director_ai/research/physics/sec_functional.py
// ─────────────────────────────────────────────────────────────────────
//! Sustainable Ethical Coherence (SEC) as a Lyapunov functional.
//!
//! V(θ) = V_coupling + V_frequency + V_entropy
//!
//! - V_coupling = Σ_{n,m} K_nm [1 - cos(θ_n - θ_m)]
//! - V_frequency = λ_ω Σ_n (dθ_n/dt - ω_n)²
//! - V_entropy = -λ_S Σ_n log(p_n + ε) (phase distribution)
//!
//! Maps to CoherenceScore: score = 1 - V_normalised.

use serde::{Deserialize, Serialize};

use crate::params::{build_knm_matrix, N_LAYERS, OMEGA_N};

/// Result of a SEC functional evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SECResult {
    /// Raw Lyapunov value (≥ 0, lower = more coherent).
    pub v: f64,
    /// Normalised to [0, 1].
    pub v_normalised: f64,
    /// Kuramoto order parameter.
    pub r_global: f64,
    /// Time derivative estimate (should be ≤ 0 for stability).
    pub dv_dt: f64,
    /// Consumer-facing score: 1 - V_normalised.
    pub coherence_score: f64,
    /// Breakdown of functional terms.
    pub v_coupling: f64,
    pub v_frequency: f64,
    pub v_entropy: f64,
}

/// SEC Lyapunov functional for the SCPN Kuramoto system.
///
/// Mirrors `SECFunctional` from `sec_functional.py`.
pub struct SECFunctional {
    knm: [[f64; N_LAYERS]; N_LAYERS],
    omega: [f64; N_LAYERS],
    lambda_omega: f64,
    lambda_entropy: f64,
    v_max: f64,
    prev_v: Option<f64>,
}

impl SECFunctional {
    pub fn new(lambda_omega: f64, lambda_entropy: f64) -> Self {
        Self::with_params(build_knm_matrix(), OMEGA_N, lambda_omega, lambda_entropy)
    }

    pub fn with_params(
        knm: [[f64; N_LAYERS]; N_LAYERS],
        omega: [f64; N_LAYERS],
        lambda_omega: f64,
        lambda_entropy: f64,
    ) -> Self {
        // V_max: worst case (all pairwise phases at π)
        let knm_sum: f64 = knm.iter().flat_map(|row| row.iter()).sum();
        let v_max = (knm_sum * 2.0).max(1e-12);
        Self {
            knm,
            omega,
            lambda_omega,
            lambda_entropy,
            v_max,
            prev_v: None,
        }
    }

    /// Default: λ_ω = 0.1, λ_S = 0.01.
    pub fn default_params() -> Self {
        Self::new(0.1, 0.01)
    }

    /// V_coupling = Σ_{n,m} K_nm [1 - cos(θ_n - θ_m)].
    pub fn coupling_potential(&self, theta: &[f64]) -> f64 {
        let n = theta.len().min(N_LAYERS);
        let mut v = 0.0;
        for i in 0..n {
            for j in 0..n {
                v += self.knm[i][j] * (1.0 - (theta[i] - theta[j]).cos());
            }
        }
        v
    }

    /// V_frequency = λ_ω Σ_n (dθ_n/dt - ω_n)².
    pub fn frequency_penalty(&self, theta: &[f64], theta_prev: Option<&[f64]>, dt: f64) -> f64 {
        let prev = match theta_prev {
            Some(p) => p,
            None => return 0.0,
        };
        let n = theta.len().min(N_LAYERS).min(prev.len());
        let safe_dt = dt.max(1e-12);
        let mut v = 0.0;
        for i in 0..n {
            let dtheta_dt = (theta[i] - prev[i]) / safe_dt;
            let dev = dtheta_dt - self.omega[i];
            v += dev * dev;
        }
        self.lambda_omega * v
    }

    /// V_entropy = -λ_S Σ_n log(p_n + ε) via phase histogram.
    pub fn entropy_term(&self, theta: &[f64]) -> f64 {
        let n = theta.len().max(1);
        let n_bins = n.max(8);
        let mut counts = vec![0u32; n_bins];
        let tau = std::f64::consts::TAU;

        for &th in theta {
            let phase = th.rem_euclid(tau);
            let bin = ((phase / tau) * n_bins as f64) as usize;
            let bin = bin.min(n_bins - 1);
            counts[bin] += 1;
        }

        let total = theta.len().max(1) as f64;
        let eps = 1e-12;
        let mut entropy = 0.0;
        for &c in &counts {
            let p = c as f64 / total + eps;
            entropy -= p * p.ln();
        }

        let max_entropy = (n_bins as f64).ln();
        self.lambda_entropy * (max_entropy - entropy)
    }

    /// Evaluate the full SEC functional.
    pub fn evaluate(
        &mut self,
        theta: &[f64],
        theta_prev: Option<&[f64]>,
        dt: f64,
    ) -> Result<SECResult, &'static str> {
        // Validate
        for &th in theta {
            if !th.is_finite() {
                return Err("theta contains NaN or Inf");
            }
        }
        if let Some(prev) = theta_prev {
            for &th in prev {
                if !th.is_finite() {
                    return Err("theta_prev contains NaN or Inf");
                }
            }
        }

        let v_coupling = self.coupling_potential(theta);
        let v_freq = self.frequency_penalty(theta, theta_prev, dt);
        let v_entropy = self.entropy_term(theta);

        let mut v = v_coupling + v_freq + v_entropy;
        if !v.is_finite() {
            v = self.v_max;
        }

        let v_norm = (v / self.v_max).clamp(0.0, 1.0);

        // Kuramoto order parameter
        let n = theta.len() as f64;
        let (sum_sin, sum_cos) = theta
            .iter()
            .fold((0.0, 0.0), |(s, c), &th| (s + th.sin(), c + th.cos()));
        let r = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2))
            .sqrt()
            .clamp(0.0, 1.0);

        // dV/dt estimation
        let dv_dt = if let Some(prev_v) = self.prev_v {
            let d = (v - prev_v) / dt.max(1e-12);
            if d.is_finite() { d } else { 0.0 }
        } else {
            0.0
        };
        self.prev_v = Some(v);

        let coherence_score = (1.0 - v_norm).clamp(0.0, 1.0);

        Ok(SECResult {
            v,
            v_normalised: v_norm,
            r_global: r,
            dv_dt,
            coherence_score,
            v_coupling,
            v_frequency: v_freq,
            v_entropy,
        })
    }

    /// Critical coupling K_c estimate.
    pub fn critical_coupling(&self) -> f64 {
        let mean: f64 = self.omega.iter().sum::<f64>() / N_LAYERS as f64;
        let var: f64 = self.omega.iter().map(|&w| (w - mean).powi(2)).sum::<f64>() / N_LAYERS as f64;
        let std_omega = var.sqrt();
        if std_omega < 1e-12 {
            return 0.0;
        }
        let g0 = 1.0 / ((std::f64::consts::TAU).sqrt() * std_omega);
        2.0 / (std::f64::consts::PI * g0)
    }

    /// Check Lyapunov stability: dV/dt ≤ tolerance.
    pub fn is_stable(result: &SECResult, tolerance: f64) -> bool {
        result.dv_dt <= tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_potential_aligned() {
        let sec = SECFunctional::default_params();
        // All phases equal → V_coupling = 0
        let theta = vec![1.0; N_LAYERS];
        let v = sec.coupling_potential(&theta);
        assert!(v.abs() < 1e-9, "V_coupling should be ~0 for aligned phases, got {v}");
    }

    #[test]
    fn test_coupling_potential_non_negative() {
        let sec = SECFunctional::default_params();
        let theta: Vec<f64> = (0..N_LAYERS)
            .map(|i| i as f64 * 0.4)
            .collect();
        let v = sec.coupling_potential(&theta);
        assert!(v >= 0.0, "V_coupling should be non-negative, got {v}");
    }

    #[test]
    fn test_frequency_penalty_zero_without_prev() {
        let sec = SECFunctional::default_params();
        let theta = vec![0.0; N_LAYERS];
        let v = sec.frequency_penalty(&theta, None, 0.01);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_entropy_term_non_negative() {
        let sec = SECFunctional::default_params();
        let theta: Vec<f64> = (0..N_LAYERS).map(|i| i as f64 * 0.3).collect();
        let v = sec.entropy_term(&theta);
        assert!(v >= 0.0, "V_entropy should be non-negative, got {v}");
    }

    #[test]
    fn test_evaluate_coherence_in_range() {
        let mut sec = SECFunctional::default_params();
        let theta = vec![0.5; N_LAYERS];
        let result = sec.evaluate(&theta, None, 0.01).unwrap();
        assert!(
            (0.0..=1.0).contains(&result.coherence_score),
            "coherence_score={} out of [0,1]",
            result.coherence_score
        );
    }

    #[test]
    fn test_evaluate_nan_rejected() {
        let mut sec = SECFunctional::default_params();
        let theta = vec![f64::NAN; N_LAYERS];
        assert!(sec.evaluate(&theta, None, 0.01).is_err());
    }

    #[test]
    fn test_aligned_phases_high_coherence() {
        let mut sec = SECFunctional::default_params();
        let theta = vec![1.0; N_LAYERS];
        let result = sec.evaluate(&theta, None, 0.01).unwrap();
        assert!(
            result.coherence_score > 0.9,
            "Aligned phases should give high coherence, got {}",
            result.coherence_score
        );
    }

    #[test]
    fn test_critical_coupling_positive() {
        let sec = SECFunctional::default_params();
        let kc = sec.critical_coupling();
        assert!(kc > 0.0, "K_c should be positive, got {kc}");
    }
}
