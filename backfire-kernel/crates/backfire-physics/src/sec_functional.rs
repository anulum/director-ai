// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SEC Functional (Lyapunov Stability)
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

/// Modified Bessel function I_0(x) — polynomial approximation.
/// Abramowitz & Stegun 9.8.1, accurate to ~1e-7 for |x| ≤ 3.75,
/// asymptotic expansion for larger x.
fn bessel_i0_approx(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (ax / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.39894228
                + t * (0.01328592
                    + t * (0.00225319
                        + t * (-0.00157565
                            + t * (0.00916281
                                + t * (-0.02057706
                                    + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))))
    }
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

    /// Recompute V_max from a new coupling matrix.
    ///
    /// Call this when SSGF geometry changes W, which effectively changes
    /// the coupling topology. Without this, V_normalised drifts because
    /// the denominator is stale.
    pub fn update_coupling(&mut self, knm: [[f64; N_LAYERS]; N_LAYERS]) {
        let knm_sum: f64 = knm.iter().flat_map(|row| row.iter()).sum();
        self.v_max = (knm_sum * 2.0).max(1e-12);
        self.knm = knm;
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

    /// V_entropy via von Mises kernel density on the circle.
    ///
    /// Evaluates KDE at n_eval points, computes Shannon entropy from
    /// the smoothed density. Bandwidth κ = 2.0 (moderate smoothing).
    pub fn entropy_term(&self, theta: &[f64]) -> f64 {
        let n = theta.len();
        if n == 0 {
            return 0.0;
        }
        let n_eval = n.max(16);
        let tau = std::f64::consts::TAU;
        let kappa = 2.0; // von Mises concentration (lower = smoother)
        let nf = n as f64;
        let eps = 1e-12;

        // Evaluate KDE at n_eval equispaced points on [0, 2π)
        let mut density = vec![0.0; n_eval];
        for (k, d) in density.iter_mut().enumerate() {
            let x = (k as f64 / n_eval as f64) * tau;
            let mut sum = 0.0;
            for &th in theta {
                sum += (kappa * (x - th).cos()).exp();
            }
            *d = sum / (nf * tau * bessel_i0_approx(kappa));
        }

        // Normalise density to a proper distribution
        let total: f64 = density.iter().sum();
        if total < eps {
            return 0.0;
        }
        for d in density.iter_mut() {
            *d /= total;
        }

        // Shannon entropy
        let mut entropy = 0.0;
        for &p in &density {
            if p > eps {
                entropy -= p * p.ln();
            }
        }

        let max_entropy = (n_eval as f64).ln();
        self.lambda_entropy * (max_entropy - entropy).max(0.0)
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
            if d.is_finite() {
                d
            } else {
                0.0
            }
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
        let var: f64 =
            self.omega.iter().map(|&w| (w - mean).powi(2)).sum::<f64>() / N_LAYERS as f64;
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
        assert!(
            v.abs() < 1e-9,
            "V_coupling should be ~0 for aligned phases, got {v}"
        );
    }

    #[test]
    fn test_coupling_potential_non_negative() {
        let sec = SECFunctional::default_params();
        let theta: Vec<f64> = (0..N_LAYERS).map(|i| i as f64 * 0.4).collect();
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
    fn test_bessel_i0_at_zero() {
        assert!((bessel_i0_approx(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_i0_at_two() {
        // I_0(2) ≈ 2.2796
        assert!((bessel_i0_approx(2.0) - 2.2796).abs() < 0.001);
    }

    #[test]
    fn test_entropy_uniform_high() {
        let sec = SECFunctional::default_params();
        // Phases spread evenly → high entropy → low V_entropy
        let theta: Vec<f64> = (0..N_LAYERS)
            .map(|i| i as f64 * std::f64::consts::TAU / N_LAYERS as f64)
            .collect();
        let v = sec.entropy_term(&theta);
        assert!(v >= 0.0);
        assert!(
            v < 0.005,
            "Uniform phases should give near-zero V_entropy, got {v}"
        );
    }

    #[test]
    fn test_entropy_clustered_higher() {
        let sec = SECFunctional::default_params();
        // All phases at same value → low entropy → higher V_entropy
        let theta_clustered = vec![1.0; N_LAYERS];
        let theta_spread: Vec<f64> = (0..N_LAYERS)
            .map(|i| i as f64 * std::f64::consts::TAU / N_LAYERS as f64)
            .collect();
        let v_c = sec.entropy_term(&theta_clustered);
        let v_s = sec.entropy_term(&theta_spread);
        assert!(
            v_c > v_s,
            "Clustered phases should have higher V_entropy: {v_c} vs {v_s}"
        );
    }

    #[test]
    fn test_critical_coupling_positive() {
        let sec = SECFunctional::default_params();
        let kc = sec.critical_coupling();
        assert!(kc > 0.0, "K_c should be positive, got {kc}");
    }

    #[test]
    fn test_update_coupling_changes_v_max() {
        let mut sec = SECFunctional::default_params();
        let v_max_before = sec.v_max;

        // Double the coupling matrix
        let mut knm = build_knm_matrix();
        for row in knm.iter_mut() {
            for v in row.iter_mut() {
                *v *= 2.0;
            }
        }
        sec.update_coupling(knm);
        assert!(
            (sec.v_max - v_max_before * 2.0).abs() < 1e-9,
            "V_max should double when coupling doubles"
        );
    }

    #[test]
    fn test_update_coupling_affects_score() {
        let mut sec = SECFunctional::default_params();
        let theta: Vec<f64> = (0..N_LAYERS).map(|i| i as f64 * 0.4).collect();

        let r1 = sec.evaluate(&theta, None, 0.01).unwrap();
        let v_coupling_before = r1.v_coupling;

        // Zero out coupling → V_coupling = 0
        let knm = [[0.0; N_LAYERS]; N_LAYERS];
        sec.update_coupling(knm);
        let r2 = sec.evaluate(&theta, None, 0.01).unwrap();
        assert!(
            r2.v_coupling < 1e-12,
            "Zero coupling should give v_coupling≈0, got {}",
            r2.v_coupling
        );
        assert!(
            r2.v_coupling < v_coupling_before,
            "Zero coupling should reduce v_coupling"
        );
    }

    #[test]
    fn test_is_stable_negative_dv_dt() {
        let result = SECResult {
            v: 0.5,
            v_normalised: 0.1,
            r_global: 0.8,
            dv_dt: -0.01,
            coherence_score: 0.9,
            v_coupling: 0.3,
            v_frequency: 0.1,
            v_entropy: 0.1,
        };
        assert!(SECFunctional::is_stable(&result, 0.0));
    }

    #[test]
    fn test_is_stable_positive_dv_dt_below_tolerance() {
        let result = SECResult {
            v: 0.5,
            v_normalised: 0.1,
            r_global: 0.8,
            dv_dt: 0.005,
            coherence_score: 0.9,
            v_coupling: 0.3,
            v_frequency: 0.1,
            v_entropy: 0.1,
        };
        assert!(SECFunctional::is_stable(&result, 0.01));
        assert!(!SECFunctional::is_stable(&result, 0.001));
    }

    #[test]
    fn test_critical_coupling_finite() {
        let sec = SECFunctional::default_params();
        let kc = sec.critical_coupling();
        assert!(kc.is_finite(), "K_c should be finite, got {kc}");
        assert!(kc > 0.0 && kc < 100.0, "K_c should be reasonable, got {kc}");
    }
}
