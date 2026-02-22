// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SSGF Decoders (z → W)
// Mirrors: SCPN-CODEBASE/optimizations/ssgf/decoder.py
// ─────────────────────────────────────────────────────────────────────
//! Latent vector z → weight matrix W decoders.
//!
//! Two decoders:
//!   - `gram_softplus`: z → symmetric A → softplus(A) → zero-diag
//!   - `rbf`: z as N coordinates → exp(-||z_i - z_j||^2 / (2σ²))
//!
//! Both guarantee: W ≥ 0, W = W^T, diag(W) = 0.

use serde::{Deserialize, Serialize};

use crate::costs::CostWeights;

/// Gradient computation method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GradientMethod {
    /// Central-difference finite-difference (240 forward passes, ~31 ms).
    FiniteDifference,
    /// Analytic Jacobian through W-dependent costs only (~50 µs).
    Analytic,
}

/// Stable softplus: log(1 + exp(x)) with overflow protection.
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Derivative of softplus: sigmoid(x) = 1/(1+exp(-x)).
#[inline]
pub fn softplus_deriv(x: f64) -> f64 {
    if x > 20.0 {
        1.0
    } else if x < -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Fill a symmetric n×n matrix `a_out` from upper-triangle vector z.
///
/// z must have length n*(n-1)/2.
/// a_out must have length n*n (row-major).
fn z_to_symmetric(z: &[f64], n: usize, a_out: &mut [f64]) {
    debug_assert_eq!(z.len(), n * (n - 1) / 2);
    debug_assert_eq!(a_out.len(), n * n);

    // Zero everything first
    for v in a_out.iter_mut() {
        *v = 0.0;
    }

    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            a_out[i * n + j] = z[idx];
            a_out[j * n + i] = z[idx];
            idx += 1;
        }
    }
}

/// Decode z → W via Gram-softplus.
///
/// W = softplus(A) where A is symmetric from z, then diag(W) = 0.
/// `w_out` must have length n*n.
pub fn decode_gram_softplus(z: &[f64], n: usize, w_out: &mut [f64]) {
    z_to_symmetric(z, n, w_out);
    for i in 0..n {
        for j in 0..n {
            w_out[i * n + j] = softplus(w_out[i * n + j]);
        }
        w_out[i * n + i] = 0.0;
    }
}

/// Decode z → W via RBF kernel.
///
/// z is interpreted as N points of dimension D, where z.len() = N*D.
/// W[i,j] = exp(-||z_i - z_j||^2 / (2σ²)), diag = 0.
pub fn decode_rbf(z: &[f64], n: usize, sigma: f64, w_out: &mut [f64]) {
    let d = z.len() / n;
    if d == 0 {
        for v in w_out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let two_sigma2 = 2.0 * sigma * sigma;

    for i in 0..n {
        w_out[i * n + i] = 0.0;
        for j in (i + 1)..n {
            let mut dist2 = 0.0;
            for k in 0..d {
                let diff = z[i * d + k] - z[j * d + k];
                dist2 += diff * diff;
            }
            let w = (-dist2 / two_sigma2).exp();
            w_out[i * n + j] = w;
            w_out[j * n + i] = w;
        }
    }
}

/// Compute finite-difference gradient of a loss function w.r.t. z.
///
/// `decode_fn` decodes z into W. `loss_fn` evaluates the loss from W.
/// Returns gradient vector of same length as z.
pub fn gradient_fd<D, L>(
    z: &[f64],
    n: usize,
    w_scratch: &mut [f64],
    decode_fn: &D,
    loss_fn: &mut L,
    eps: f64,
) -> Vec<f64>
where
    D: Fn(&[f64], usize, &mut [f64]),
    L: FnMut(&[f64]) -> f64,
{
    let dim = z.len();
    let mut grad = vec![0.0; dim];
    let mut z_perturb = z.to_vec();

    for i in 0..dim {
        // Forward
        z_perturb[i] = z[i] + eps;
        decode_fn(&z_perturb, n, w_scratch);
        let loss_plus = loss_fn(w_scratch);

        // Backward
        z_perturb[i] = z[i] - eps;
        decode_fn(&z_perturb, n, w_scratch);
        let loss_minus = loss_fn(w_scratch);

        grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
        z_perturb[i] = z[i]; // restore
    }

    grad
}

/// Compute ∂U/∂W[i,j] from W-dependent cost terms (C9, C10, R_graph Frobenius).
///
/// The eigensolver-dependent terms (C7, C8, gap penalty in R_graph) are
/// stop-gradient: eigenpairs are frozen from step 5.
///
/// `du_dw` must have length n*n (row-major). Filled in-place.
fn compute_du_dw(
    w: &[f64],
    n: usize,
    protected_mask: &[bool],
    weights: &CostWeights,
    du_dw: &mut [f64],
) {
    debug_assert_eq!(du_dw.len(), n * n);
    for v in du_dw.iter_mut() {
        *v = 0.0;
    }

    // ── C9_memory: default mode = (row_sum[8]/n - 1)^2 ──────────────
    // ∂C9/∂W[8,j] = 2 * (row_sum[8]/n - 1) * (1/n)   for all j ≠ 8
    let l9_idx = 8;
    if l9_idx < n && w.len() >= n * n {
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += w[l9_idx * n + j];
        }
        let norm_conn = row_sum / n.max(1) as f64;
        let dc9_d_row = 2.0 * (norm_conn - 1.0) / n.max(1) as f64;
        let w_c9 = weights.c9;
        for j in 0..n {
            if j != l9_idx {
                du_dw[l9_idx * n + j] += w_c9 * dc9_d_row;
                // W is symmetric, so also add the transpose contribution
                du_dw[j * n + l9_idx] += w_c9 * dc9_d_row;
            }
        }
    }

    // ── C10_boundary: Σ_{i,j: mask[i]≠mask[j]} W[i,j]² / count ────
    // ∂C10/∂W[i,j] = 2 * W[i,j] / count   when mask[i] ≠ mask[j]
    if !protected_mask.is_empty() {
        let prot_count: usize = protected_mask.iter().filter(|&&v| v).count();
        if prot_count > 0 && prot_count < n {
            let mut cross_count = 0usize;
            for i in 0..n {
                for j in 0..n {
                    if protected_mask[i] != protected_mask[j] {
                        cross_count += 1;
                    }
                }
            }
            let denom = cross_count.max(1) as f64;
            let w_c10 = weights.c10;
            for i in 0..n {
                for j in 0..n {
                    if protected_mask[i] != protected_mask[j] {
                        du_dw[i * n + j] += w_c10 * 2.0 * w[i * n + j] / denom;
                    }
                }
            }
        }
    }

    // ── R_graph (Frobenius part only): α_frob * ||W||_F² ────────────
    // ∂(α_frob * Σ W[i,j]²)/∂W[i,j] = 2 * α_frob * W[i,j]
    // The gap penalty term depends on eigvals → stop-gradient.
    let alpha_frob = 0.01;
    let w_reg = weights.reg;
    for idx in 0..n * n {
        du_dw[idx] += w_reg * 2.0 * alpha_frob * w[idx];
    }
}

/// Analytic gradient for gram_softplus decoder.
///
/// Chain rule: ∂U/∂z[k] = (∂U/∂W[i,j] + ∂U/∂W[j,i]) · sigmoid(z[k])
/// where k indexes the upper-triangle pair (i,j) with i<j,
/// and sigmoid is the derivative of softplus.
///
/// `du_dw` is an n×n scratch buffer (will be overwritten).
/// Returns gradient vector of length n*(n-1)/2.
pub fn gradient_analytic_gram(
    z: &[f64],
    n: usize,
    w: &[f64],
    protected_mask: &[bool],
    weights: &CostWeights,
    du_dw: &mut [f64],
) -> Vec<f64> {
    let dim_z = n * (n - 1) / 2;
    debug_assert_eq!(z.len(), dim_z);
    debug_assert_eq!(du_dw.len(), n * n);

    // Step 1: Compute ∂U/∂W
    compute_du_dw(w, n, protected_mask, weights, du_dw);

    // Step 2: Chain rule through z → A → softplus(A) → W
    // z[k] maps to pair (i,j) with i<j.
    // W[i,j] = softplus(z[k]), W[j,i] = softplus(z[k]), W[i,i]=0.
    // ∂W[i,j]/∂z[k] = sigmoid(z[k])
    // ∂U/∂z[k] = (∂U/∂W[i,j] + ∂U/∂W[j,i]) · sigmoid(z[k])
    let mut grad = vec![0.0; dim_z];
    let mut k = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let sig = softplus_deriv(z[k]);
            grad[k] = (du_dw[i * n + j] + du_dw[j * n + i]) * sig;
            k += 1;
        }
    }
    grad
}

/// Analytic gradient for RBF decoder.
///
/// W[i,j] = exp(-||z_i - z_j||² / (2σ²))
/// ∂W[i,j]/∂z[i·D+d] = W[i,j] · (z[j·D+d] - z[i·D+d]) / σ²
/// ∂W[i,j]/∂z[j·D+d] = W[i,j] · (z[i·D+d] - z[j·D+d]) / σ²
///
/// `du_dw` is an n×n scratch buffer (will be overwritten).
/// Returns gradient vector of length z.len() (= n*D).
pub fn gradient_analytic_rbf(
    z: &[f64],
    n: usize,
    sigma: f64,
    w: &[f64],
    protected_mask: &[bool],
    weights: &CostWeights,
    du_dw: &mut [f64],
) -> Vec<f64> {
    let dim_z = z.len();
    let d = dim_z / n;
    debug_assert_eq!(du_dw.len(), n * n);

    // Step 1: Compute ∂U/∂W
    compute_du_dw(w, n, protected_mask, weights, du_dw);

    // Step 2: Chain rule through z coords → RBF → W
    let sigma2 = sigma * sigma;
    let mut grad = vec![0.0; dim_z];

    for i in 0..n {
        for j in (i + 1)..n {
            let du_ij = du_dw[i * n + j] + du_dw[j * n + i];
            let w_ij = w[i * n + j];
            for dd in 0..d {
                let diff = z[j * d + dd] - z[i * d + dd];
                let dw_dz_id = w_ij * diff / sigma2;
                // ∂W[i,j]/∂z[i·D+d] = W[i,j] · (z[j·D+d] - z[i·D+d]) / σ²
                grad[i * d + dd] += du_ij * dw_dz_id;
                // ∂W[i,j]/∂z[j·D+d] = W[i,j] · (z[i·D+d] - z[j·D+d]) / σ²  = -dw_dz_id
                grad[j * d + dd] -= du_ij * dw_dz_id;
            }
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softplus_positive() {
        assert!(softplus(0.0) > 0.0);
        assert!(softplus(-5.0) > 0.0);
        assert!((softplus(0.0) - 0.6931471805599453).abs() < 1e-10);
    }

    #[test]
    fn test_softplus_large() {
        // For large x, softplus(x) ≈ x
        assert!((softplus(100.0) - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_gram_softplus_symmetric_nonneg_zerodiag() {
        let n = 4;
        let z = vec![0.1, -0.3, 0.5, 0.2, -0.1, 0.4]; // 4*3/2 = 6
        let mut w = vec![0.0; n * n];
        decode_gram_softplus(&z, n, &mut w);

        // Symmetric
        for i in 0..n {
            for j in 0..n {
                assert!((w[i * n + j] - w[j * n + i]).abs() < 1e-12);
            }
        }
        // Non-negative
        assert!(w.iter().all(|&v| v >= 0.0));
        // Zero diagonal
        for i in 0..n {
            assert!((w[i * n + i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rbf_symmetric_nonneg_zerodiag() {
        let n = 4;
        let d = 3;
        let z: Vec<f64> = (0..n * d).map(|i| i as f64 * 0.1).collect();
        let mut w = vec![0.0; n * n];
        decode_rbf(&z, n, 1.0, &mut w);

        for i in 0..n {
            for j in 0..n {
                assert!((w[i * n + j] - w[j * n + i]).abs() < 1e-12);
            }
        }
        assert!(w.iter().all(|&v| v >= 0.0));
        for i in 0..n {
            assert!((w[i * n + i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rbf_identical_points() {
        let n = 3;
        // All points at origin → W[i,j] = 1.0 for i≠j
        let z = vec![0.0; n * 2];
        let mut w = vec![0.0; n * n];
        decode_rbf(&z, n, 1.0, &mut w);

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    assert!((w[i * n + j]).abs() < 1e-12);
                } else {
                    assert!((w[i * n + j] - 1.0).abs() < 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_gradient_fd_direction() {
        let n = 3;
        let z = vec![0.0; n * (n - 1) / 2]; // 3 elements
        let mut w_scratch = vec![0.0; n * n];

        let decode_fn = |z: &[f64], n: usize, w: &mut [f64]| {
            decode_gram_softplus(z, n, w);
        };
        // Loss = sum of W
        let loss_fn = |w: &[f64]| -> f64 { w.iter().sum() };

        let mut loss_fn = loss_fn;
        let grad = gradient_fd(&z, n, &mut w_scratch, &decode_fn, &mut loss_fn, 1e-5);
        // Gradient should be non-zero (softplus is monotone)
        assert!(grad.iter().any(|&v| v.abs() > 1e-8));
    }

    #[test]
    fn test_softplus_deriv_is_sigmoid() {
        // sigmoid(0) = 0.5
        assert!((softplus_deriv(0.0) - 0.5).abs() < 1e-10);
        // sigmoid(large) → 1
        assert!((softplus_deriv(25.0) - 1.0).abs() < 1e-10);
        // sigmoid(very negative) → 0
        assert!((softplus_deriv(-25.0)).abs() < 1e-10);
        // Check d/dx softplus(x) ≈ sigmoid(x) numerically
        for &x in &[-3.0, -1.0, 0.0, 1.0, 3.0] {
            let fd = (softplus(x + 1e-7) - softplus(x - 1e-7)) / 2e-7;
            assert!(
                (fd - softplus_deriv(x)).abs() < 1e-5,
                "softplus_deriv({x}) = {} but FD = {fd}",
                softplus_deriv(x)
            );
        }
    }

    #[test]
    fn test_analytic_vs_fd_gram_agreement() {
        // Compare analytic gradient against FD on W-dependent costs only
        // (C9, C10, R_graph Frobenius).
        use crate::costs::compute_costs;
        use crate::spectral::{GaugeMethod, SpectralBridge};

        let n = 6;
        let z: Vec<f64> = (0..n * (n - 1) / 2)
            .map(|i| (i as f64 * 0.13).sin() * 0.8)
            .collect();
        let mut w = vec![0.0; n * n];
        decode_gram_softplus(&z, n, &mut w);

        let mask = vec![true, true, true, false, false, false];
        // Weights: only W-dependent terms (C9, C10, R_graph), others zeroed
        let weights = CostWeights {
            micro: 0.0,
            c7: 0.0,
            c8: 0.0,
            c9: 0.2,
            c10: 0.4,
            reg: 0.1,
            c4_tcbo: 0.0,
            pgbo: 0.0,
        };

        // Analytic gradient
        let mut du_dw = vec![0.0; n * n];
        let grad_analytic = gradient_analytic_gram(&z, n, &w, &mask, &weights, &mut du_dw);

        // FD gradient using the same cost (only W-dependent terms)
        let theta = vec![0.0; n];
        let phi_target = vec![0.0; n * n];
        let p_mask = mask.clone();
        let w_clone = weights.clone();

        let mut w_scratch = vec![0.0; n * n];
        let mut grad_spectral = SpectralBridge::new(n, GaugeMethod::None);
        let mut grad_eigvals = vec![0.0; n];
        let mut grad_eigvecs = vec![0.0; n * n];

        let decode_fn = |z: &[f64], n: usize, w_out: &mut [f64]| {
            decode_gram_softplus(z, n, w_out);
        };
        let mut loss_fn = |w: &[f64]| -> f64 {
            grad_spectral.compute_eigenpairs(w, &mut grad_eigvals, &mut grad_eigvecs);
            compute_costs(
                &theta, w, n, &grad_eigvals, &grad_eigvecs,
                &phi_target, &p_mask, &w_clone, 0.5, None,
            )
            .u_total
        };

        let grad_fd_result = gradient_fd(&z, n, &mut w_scratch, &decode_fn, &mut loss_fn, 1e-5);

        // Cosine similarity
        let dot: f64 = grad_analytic
            .iter()
            .zip(grad_fd_result.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f64 = grad_analytic.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_b: f64 = grad_fd_result.iter().map(|v| v * v).sum::<f64>().sqrt();

        if norm_a > 1e-10 && norm_b > 1e-10 {
            let cosine = dot / (norm_a * norm_b);
            assert!(
                cosine > 0.95,
                "Cosine similarity between analytic and FD gradients = {cosine:.4} (expected > 0.95)"
            );
        }

        // Component-wise relative error on non-negligible components.
        // Tolerance is generous (50%) because the analytic gradient uses
        // stop-gradient on eigensolver-dependent terms (gap penalty in R_graph),
        // while FD captures the full derivative through eigenpairs.
        for (i, (a, f)) in grad_analytic.iter().zip(grad_fd_result.iter()).enumerate() {
            if f.abs() > 1e-6 {
                let rel_err = (a - f).abs() / f.abs();
                assert!(
                    rel_err < 0.50,
                    "Gradient component {i}: analytic={a:.6}, fd={f:.6}, rel_err={rel_err:.4}"
                );
            }
        }
    }

    #[test]
    fn test_analytic_gram_zero_w_zero_grad() {
        // When z is very negative, W ≈ 0 everywhere, so cost contributions from
        // C10 and R_graph Frobenius should be ~0, and gradient ~0.
        let n = 4;
        let z = vec![-30.0; n * (n - 1) / 2];
        let mut w = vec![0.0; n * n];
        decode_gram_softplus(&z, n, &mut w);

        let mask = vec![false; n];
        let weights = CostWeights::default();
        let mut du_dw = vec![0.0; n * n];

        let grad = gradient_analytic_gram(&z, n, &w, &mask, &weights, &mut du_dw);
        let max_grad = grad.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        assert!(
            max_grad < 1e-8,
            "Gradient should be ~0 when W ≈ 0, max component = {max_grad}"
        );
    }

    #[test]
    fn test_analytic_rbf_gradient_direction() {
        // Basic sanity: gradient should be non-zero for non-trivial z
        let n = 4;
        let d = 3;
        let z: Vec<f64> = (0..n * d).map(|i| (i as f64 * 0.2).sin()).collect();
        let mut w = vec![0.0; n * n];
        decode_rbf(&z, n, 1.0, &mut w);

        let mask = vec![true, true, false, false];
        let weights = CostWeights {
            micro: 0.0,
            c7: 0.0,
            c8: 0.0,
            c9: 0.2,
            c10: 0.4,
            reg: 0.1,
            c4_tcbo: 0.0,
            pgbo: 0.0,
        };
        let mut du_dw = vec![0.0; n * n];

        let grad = gradient_analytic_rbf(&z, n, 1.0, &w, &mask, &weights, &mut du_dw);
        assert!(
            grad.iter().any(|&v| v.abs() > 1e-8),
            "RBF analytic gradient should be non-zero for non-trivial z"
        );
    }

    #[test]
    fn test_gradient_method_enum() {
        assert_ne!(GradientMethod::FiniteDifference, GradientMethod::Analytic);
        let m = GradientMethod::Analytic;
        let _m2 = m; // Copy
        let _m3 = m.clone(); // Clone
    }
}
