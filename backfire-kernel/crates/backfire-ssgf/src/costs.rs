// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SSGF Layer Cost Terms
// Mirrors: SCPN-CODEBASE/optimizations/ssgf/costs.py
// ─────────────────────────────────────────────────────────────────────
//! Eight cost terms for the SSGF outer-cycle gradient descent:
//!   C_micro, C7_symbolic, C8_phase, C9_memory, C10_boundary,
//!   R_graph, C4_tcbo, C_pgbo → U_total (weighted sum).

use serde::{Deserialize, Serialize};

/// Weights for each cost term in U_total.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostWeights {
    pub micro: f64,
    pub c7: f64,
    pub c8: f64,
    pub c9: f64,
    pub c10: f64,
    pub reg: f64,
    pub c4_tcbo: f64,
    pub pgbo: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            micro: 1.0,
            c7: 0.5,
            c8: 0.3,
            c9: 0.2,
            c10: 0.4,
            reg: 0.1,
            c4_tcbo: 0.3,
            pgbo: 0.05,
        }
    }
}

/// Breakdown of all SSGF cost terms.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub c_micro: f64,
    pub c7_symbolic: f64,
    pub c8_phase: f64,
    pub c9_memory: f64,
    pub c10_boundary: f64,
    pub r_graph: f64,
    pub c4_tcbo: f64,
    pub c_pgbo: f64,
    pub u_total: f64,
}

/// Micro-cycle coherence cost: C_micro = 1 - R.
pub fn cost_micro(theta: &[f64]) -> f64 {
    let n = theta.len() as f64;
    if n < 1.0 {
        return 1.0;
    }
    let (sum_sin, sum_cos) = theta
        .iter()
        .fold((0.0, 0.0), |(s, c), &th| (s + th.sin(), c + th.cos()));
    let r = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt().clamp(0.0, 1.0);
    1.0 - r
}

/// L7 symbolic alignment: ||φ_actual - φ_target||² at Fiedler vector.
///
/// `eigvecs` and `phi_target` are n×n row-major (columns = eigvectors).
pub fn cost_c7_symbolic(eigvecs: &[f64], phi_target: &[f64], n: usize) -> f64 {
    if n < 2 || eigvecs.len() < n * 2 || phi_target.len() < n * 2 {
        return 0.0;
    }
    // Compare Fiedler vector (column 1) between actual and target
    let mut sum_sq = 0.0;
    for row in 0..n {
        let actual = eigvecs[row * n + 1];
        let target = phi_target[row * n + 1];
        let diff = actual - target;
        sum_sq += diff * diff;
    }
    sum_sq
}

/// L8 phase-field misalignment.
///
/// Penalises phase deviation at layer 8 (index 7) relative to Fiedler value.
pub fn cost_c8_phase(theta: &[f64], eigvals: &[f64]) -> f64 {
    if eigvals.len() < 2 {
        return 0.0;
    }
    let fiedler_val = eigvals[1].max(1e-12);
    let layer_idx = 7; // L8 = index 7
    if layer_idx >= theta.len() {
        return 0.0;
    }

    // Phase deviation from circular mean
    let n = theta.len() as f64;
    let (sum_sin, sum_cos) = theta
        .iter()
        .fold((0.0, 0.0), |(s, c), &th| (s + th.sin(), c + th.cos()));
    let mean_phase = (sum_sin / n).atan2(sum_cos / n);
    let delta = theta[layer_idx] - mean_phase;
    let phase_err = delta.sin().powi(2);
    phase_err / fiedler_val
}

/// L9 memory probe error.
///
/// Default: penalises low connectivity at L9 row (index 8).
/// If `memory_probes` is provided, computes ||W_probes - targets||².
pub fn cost_c9_memory(w: &[f64], n: usize, memory_probes: Option<&[f64]>) -> f64 {
    if let Some(probes) = memory_probes {
        let mut sum_sq = 0.0;
        let len = w.len().min(probes.len());
        for i in 0..len {
            let diff = w[i] - probes[i];
            sum_sq += diff * diff;
        }
        return sum_sq / len.max(1) as f64;
    }

    // Default: connectivity at L9 (index 8)
    let l9_idx = 8;
    if l9_idx >= n || w.len() < n * n {
        return 0.0;
    }
    let mut row_sum = 0.0;
    for j in 0..n {
        row_sum += w[l9_idx * n + j];
    }
    let norm_conn = row_sum / n.max(1) as f64;
    let target = 1.0;
    (norm_conn - target).powi(2)
}

/// L10 boundary mask violation.
///
/// Penalises cross-boundary W entries between protected and unprotected nodes.
pub fn cost_c10_boundary(w: &[f64], n: usize, protected_mask: &[bool]) -> f64 {
    if protected_mask.is_empty() {
        return 0.0;
    }
    let prot_count: usize = protected_mask.iter().filter(|&&v| v).count();
    if prot_count == 0 || prot_count == n {
        return 0.0;
    }

    let mut sum_sq = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in 0..n {
            if protected_mask[i] != protected_mask[j] {
                let v = w[i * n + j];
                sum_sq += v * v;
                count += 1;
            }
        }
    }
    sum_sq / count.max(1) as f64
}

/// Graph regulariser: α_frob ||W||_F² + α_gap / (λ_1 + ε).
pub fn regularise_graph(
    w: &[f64],
    eigvals: &[f64],
    alpha_frob: f64,
    alpha_gap: f64,
) -> f64 {
    let frob: f64 = w.iter().map(|&v| v * v).sum();
    let gap_penalty = if eigvals.len() >= 2 {
        1.0 / (eigvals[1] + 1e-8)
    } else {
        1.0
    };
    alpha_frob * frob + alpha_gap * gap_penalty
}

/// TCBO deficit cost: max(0, τ_h1 - p_h1)².
pub fn cost_c4_tcbo(p_h1: f64, tau_h1: f64) -> f64 {
    let deficit = (tau_h1 - p_h1).max(0.0);
    deficit * deficit
}

/// PGBO anisotropy regulariser: α_frob ||h_munu||_F².
pub fn cost_pgbo(h_munu: Option<&[f64]>, alpha_frob: f64) -> f64 {
    match h_munu {
        Some(h) => alpha_frob * h.iter().map(|&v| v * v).sum::<f64>(),
        None => 0.0,
    }
}

/// Compute all cost terms and weighted total.
pub fn compute_costs(
    theta: &[f64],
    w: &[f64],
    n: usize,
    eigvals: &[f64],
    eigvecs: &[f64],
    phi_target: &[f64],
    protected_mask: &[bool],
    weights: &CostWeights,
    p_h1: f64,
    h_munu: Option<&[f64]>,
) -> CostBreakdown {
    let cm = cost_micro(theta);
    let c7 = cost_c7_symbolic(eigvecs, phi_target, n);
    let c8 = cost_c8_phase(theta, eigvals);
    let c9 = cost_c9_memory(w, n, None);
    let c10 = cost_c10_boundary(w, n, protected_mask);
    let rg = regularise_graph(w, eigvals, 0.01, 0.1);
    let c4t = cost_c4_tcbo(p_h1, 0.72);
    let cpg = cost_pgbo(h_munu, 0.01);

    let u_total = weights.micro * cm
        + weights.c7 * c7
        + weights.c8 * c8
        + weights.c9 * c9
        + weights.c10 * c10
        + weights.reg * rg
        + weights.c4_tcbo * c4t
        + weights.pgbo * cpg;

    CostBreakdown {
        c_micro: cm,
        c7_symbolic: c7,
        c8_phase: c8,
        c9_memory: c9,
        c10_boundary: c10,
        r_graph: rg,
        c4_tcbo: c4t,
        c_pgbo: cpg,
        u_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_micro_synced() {
        let theta = vec![0.5; 16]; // all equal
        let c = cost_micro(&theta);
        assert!(c < 0.01, "Synced phases should give low cost, got {c}");
    }

    #[test]
    fn test_cost_micro_desynced() {
        let tau = std::f64::consts::TAU;
        let theta: Vec<f64> = (0..4).map(|i| i as f64 * tau / 4.0).collect();
        let c = cost_micro(&theta);
        assert!(c > 0.9, "Uniform phases should give cost near 1, got {c}");
    }

    #[test]
    fn test_c7_zero_when_aligned() {
        let n = 4;
        let eigvecs = vec![1.0; n * n]; // same as target
        let phi_target = vec![1.0; n * n];
        let c = cost_c7_symbolic(&eigvecs, &phi_target, n);
        assert!(c < 1e-12, "Aligned should give 0, got {c}");
    }

    #[test]
    fn test_c10_zero_when_no_boundary() {
        let n = 4;
        let w = vec![1.0; n * n];
        let mask = vec![false; n]; // no protection
        let c = cost_c10_boundary(&w, n, &mask);
        assert!(c < 1e-12, "No boundary should give 0, got {c}");
    }

    #[test]
    fn test_c10_nonzero_with_boundary() {
        let n = 4;
        let w = vec![1.0; n * n];
        let mask = vec![true, true, false, false]; // half protected
        let c = cost_c10_boundary(&w, n, &mask);
        assert!(c > 0.0, "Cross-boundary weights should give nonzero cost");
    }

    #[test]
    fn test_c4_tcbo_above_threshold() {
        assert!(cost_c4_tcbo(0.9, 0.72) < 1e-12);
    }

    #[test]
    fn test_c4_tcbo_below_threshold() {
        let c = cost_c4_tcbo(0.5, 0.72);
        let expected = (0.72 - 0.5_f64).powi(2);
        assert!((c - expected).abs() < 1e-12);
    }

    #[test]
    fn test_regularise_graph() {
        let w = vec![0.0, 1.0, 0.5, 1.0, 0.0, 0.8, 0.5, 0.8, 0.0];
        let eigvals = vec![0.0, 0.5, 1.5];
        let rg = regularise_graph(&w, &eigvals, 0.01, 0.1);
        let frob: f64 = w.iter().map(|&v| v * v).sum();
        let expected = 0.01 * frob + 0.1 / (0.5 + 1e-8);
        assert!((rg - expected).abs() < 1e-8);
    }

    #[test]
    fn test_compute_costs_total() {
        let n = 4;
        let theta = vec![0.1, 0.2, 0.3, 0.4];
        let w = vec![0.0; n * n];
        let eigvals = vec![0.0, 0.5, 1.0, 1.5];
        let eigvecs = vec![0.0; n * n];
        let phi_target = vec![0.0; n * n];
        let mask = vec![false; n];
        let weights = CostWeights::default();

        let cb = compute_costs(
            &theta, &w, n, &eigvals, &eigvecs, &phi_target, &mask, &weights, 0.5, None,
        );
        assert!(cb.u_total > 0.0, "Total cost should be positive");
        assert!(cb.c_micro > 0.0, "Micro cost should be positive");
    }
}
