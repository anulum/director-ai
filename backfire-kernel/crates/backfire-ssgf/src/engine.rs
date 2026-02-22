// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SSGF Engine (Outer-Cycle Orchestrator)
// Mirrors: SCPN-CODEBASE/optimizations/ssgf/outer_cycle.py
// ─────────────────────────────────────────────────────────────────────
//! Ten-step outer cycle:
//!   1. Decode z → W
//!   2. Verify W (symmetric, non-negative, zero diagonal)
//!   3. PGBO: compute h_munu from phase gradients
//!   4. Run micro-cycle (Kuramoto + geometry feedback + PGBO)
//!   5. Spectral bridge: W → Laplacian → eigenpairs
//!   6. Verify eigenvalue ordering
//!   7. Compute cost terms
//!   8. Gradient update: z -= lr_z · ∇_z U_total
//!   9. L16 closure: PI controllers + PLV gate + refusal
//!   10. Log step

use serde::{Deserialize, Serialize};

use backfire_consciousness::{PGBOConfig, PGBOEngine};
use backfire_physics::l16_closure::L16CostInputs;
use backfire_physics::L16Controller;

use crate::costs::{compute_costs, CostBreakdown, CostWeights};
use crate::decoder::{
    decode_gram_softplus, decode_rbf, gradient_analytic_gram, gradient_analytic_rbf, gradient_fd,
    GradientMethod,
};
use crate::micro::MicroCycleEngine;
use crate::node_space::NodeSpace;
use crate::spectral::{GaugeMethod, SpectralBridge};

/// SSGF engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSGFConfig {
    pub n: usize,
    pub decoder: String,
    pub rbf_sigma: f64,
    pub n_micro: usize,
    pub dt_micro: f64,
    pub dt_outer: f64,
    pub lr_z: f64,
    pub sigma_g: f64,
    pub noise_amp: f64,
    pub field_pressure: f64,
    pub cost_weights: CostWeights,
    pub plv_threshold: f64,
    pub plv_window: usize,
    pub h_rec_window: usize,
    pub fd_eps: f64,
    pub gradient_method: GradientMethod,
    pub max_outer_steps: usize,
    pub seed: u64,
    // PGBO
    pub pgbo_enabled: bool,
    pub pgbo_alpha: f64,
    pub pgbo_kappa: f64,
    pub pgbo_weight: f64,
    pub pgbo_u_cap: f64,
    pub pgbo_traceless: bool,
    // TCBO (observer runs externally; engine receives p_h1)
    pub tcbo_enabled: bool,
}

impl Default for SSGFConfig {
    fn default() -> Self {
        Self {
            n: 16,
            decoder: "gram_softplus".to_string(),
            rbf_sigma: 1.0,
            n_micro: 10,
            dt_micro: 0.001,
            dt_outer: 0.01,
            lr_z: 0.01,
            sigma_g: 0.1,
            noise_amp: 0.02,
            field_pressure: 0.05,
            cost_weights: CostWeights::default(),
            plv_threshold: 0.6,
            plv_window: 10,
            h_rec_window: 5,
            fd_eps: 1e-5,
            gradient_method: GradientMethod::Analytic,
            max_outer_steps: 200,
            seed: 42,
            pgbo_enabled: true,
            pgbo_alpha: 0.5,
            pgbo_kappa: 0.3,
            pgbo_weight: 0.1,
            pgbo_u_cap: 10.0,
            pgbo_traceless: false,
            tcbo_enabled: true,
        }
    }
}

/// Log entry for one outer-cycle step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SSGFStepLog {
    pub step: usize,
    pub costs: CostBreakdown,
    pub r_global: f64,
    pub fiedler_value: f64,
    pub spectral_gap: f64,
    pub h_rec: f64,
    pub gate_open: bool,
    pub refusal: bool,
    pub w_valid: bool,
    pub eigval_ordered: bool,
    pub pgbo_u_norm: f64,
    pub pgbo_h_frob: f64,
    pub tcbo_p_h1: f64,
    pub tcbo_gate_open: bool,
}

/// SSGF outer-cycle orchestrator.
pub struct SSGFEngine {
    pub cfg: SSGFConfig,
    pub ns: NodeSpace,
    micro: MicroCycleEngine,
    spectral: SpectralBridge,
    l16: L16Controller,
    pgbo: Option<PGBOEngine>,
    tcbo_p_h1: f64,
    step_count: usize,
    pub log: Vec<SSGFStepLog>,
    // Scratch for gradient computation
    w_scratch: Vec<f64>,
    du_dw: Vec<f64>,
}

impl SSGFEngine {
    /// Create a new SSGF engine.
    ///
    /// `omega`: intrinsic frequencies (length n).
    /// `k`: baseline coupling matrix (n×n row-major).
    pub fn new(omega: &[f64], k: &[f64], config: SSGFConfig) -> Self {
        let n = config.n;

        let mut ns = NodeSpace::new(n);
        ns.randomise(config.seed);

        let micro = MicroCycleEngine::new(
            n,
            omega,
            k,
            config.dt_micro,
            config.sigma_g,
            config.noise_amp,
            config.field_pressure,
            if config.pgbo_enabled {
                config.pgbo_weight
            } else {
                0.0
            },
            config.seed.wrapping_add(1),
        );

        let spectral = SpectralBridge::new(n, GaugeMethod::Sign);

        let l16 = L16Controller::new(
            n,
            config.plv_threshold,
            config.plv_window,
            config.h_rec_window,
            0.5,
            0.5,
            1.5,
        );

        let pgbo = if config.pgbo_enabled {
            Some(PGBOEngine::new(
                n,
                PGBOConfig {
                    alpha: config.pgbo_alpha,
                    kappa: config.pgbo_kappa,
                    u_cap: config.pgbo_u_cap,
                    traceless: config.pgbo_traceless,
                    ..PGBOConfig::default()
                },
            ))
        } else {
            None
        };

        let mut engine = Self {
            ns,
            micro,
            spectral,
            l16,
            pgbo,
            tcbo_p_h1: 0.0,
            step_count: 0,
            log: Vec::new(),
            w_scratch: vec![0.0; n * n],
            du_dw: vec![0.0; n * n],
            cfg: config,
        };

        // Decode initial W and compute initial eigenpairs
        engine.decode();
        engine.spectral.compute_eigenpairs(
            &engine.ns.w,
            &mut engine.ns.eigvals,
            &mut engine.ns.eigvecs,
        );

        engine
    }

    pub fn default_params(omega: &[f64], k: &[f64]) -> Self {
        Self::new(omega, k, SSGFConfig::default())
    }

    // ------------------------------------------------------------------
    // Decode
    // ------------------------------------------------------------------

    fn decode(&mut self) {
        let n = self.cfg.n;
        if self.cfg.decoder == "rbf" {
            decode_rbf(&self.ns.z, n, self.cfg.rbf_sigma, &mut self.ns.w);
        } else {
            decode_gram_softplus(&self.ns.z, n, &mut self.ns.w);
        }
    }

    // ------------------------------------------------------------------
    // Verification
    // ------------------------------------------------------------------

    fn verify_w(&self) -> bool {
        let n = self.cfg.n;
        let w = &self.ns.w;
        // Symmetric
        for i in 0..n {
            for j in (i + 1)..n {
                if (w[i * n + j] - w[j * n + i]).abs() > 1e-12 {
                    return false;
                }
            }
        }
        // Non-negative
        if w.iter().any(|&v| v < -1e-12) {
            return false;
        }
        // Zero diagonal
        for i in 0..n {
            if w[i * n + i].abs() > 1e-12 {
                return false;
            }
        }
        true
    }

    fn verify_eigvals(&self) -> bool {
        self.ns
            .eigvals
            .windows(2)
            .all(|w| w[0] <= w[1] + 1e-10)
    }

    // ------------------------------------------------------------------
    // Gradient
    // ------------------------------------------------------------------

    fn compute_gradient(&mut self) -> Vec<f64> {
        let n = self.cfg.n;

        match self.cfg.gradient_method {
            GradientMethod::Analytic => {
                if self.cfg.decoder == "rbf" {
                    gradient_analytic_rbf(
                        &self.ns.z,
                        n,
                        self.cfg.rbf_sigma,
                        &self.ns.w,
                        &self.ns.protected_mask,
                        &self.cfg.cost_weights,
                        &mut self.du_dw,
                    )
                } else {
                    gradient_analytic_gram(
                        &self.ns.z,
                        n,
                        &self.ns.w,
                        &self.ns.protected_mask,
                        &self.cfg.cost_weights,
                        &mut self.du_dw,
                    )
                }
            }
            GradientMethod::FiniteDifference => {
                let eps = self.cfg.fd_eps;
                let decoder_name = self.cfg.decoder.clone();
                let rbf_sigma = self.cfg.rbf_sigma;

                let theta = self.ns.theta.clone();
                let phi_target = self.ns.phi_target.clone();
                let protected_mask = self.ns.protected_mask.clone();
                let weights = self.cfg.cost_weights.clone();
                let p_h1 = self.tcbo_p_h1;

                let mut grad_spectral = SpectralBridge::new(n, GaugeMethod::None);
                let mut grad_eigvals = vec![0.0; n];
                let mut grad_eigvecs = vec![0.0; n * n];

                let decode_fn = move |z: &[f64], n: usize, w_out: &mut [f64]| {
                    if decoder_name == "rbf" {
                        decode_rbf(z, n, rbf_sigma, w_out);
                    } else {
                        decode_gram_softplus(z, n, w_out);
                    }
                };

                let loss_fn = |w: &[f64]| -> f64 {
                    grad_spectral.compute_eigenpairs(w, &mut grad_eigvals, &mut grad_eigvecs);
                    compute_costs(
                        &theta,
                        w,
                        n,
                        &grad_eigvals,
                        &grad_eigvecs,
                        &phi_target,
                        &protected_mask,
                        &weights,
                        p_h1,
                        None,
                    )
                    .u_total
                };

                let mut loss_fn = loss_fn;
                gradient_fd(
                    &self.ns.z,
                    n,
                    &mut self.w_scratch,
                    &decode_fn,
                    &mut loss_fn,
                    eps,
                )
            }
        }
    }

    // ------------------------------------------------------------------
    // External TCBO injection
    // ------------------------------------------------------------------

    /// Inject TCBO p_h1 from external observer.
    pub fn set_tcbo_p_h1(&mut self, p_h1: f64) {
        self.tcbo_p_h1 = p_h1;
    }

    // ------------------------------------------------------------------
    // Outer cycle
    // ------------------------------------------------------------------

    /// Execute one complete outer-cycle step (10 stages).
    pub fn outer_step(&mut self) -> SSGFStepLog {
        let mut log = SSGFStepLog {
            step: self.step_count,
            ..Default::default()
        };

        let n = self.cfg.n;

        // 1. Decode z → W
        self.decode();

        // 2. Verify W
        log.w_valid = self.verify_w();

        // 3. PGBO: compute h_munu from phase gradients
        if let Some(ref mut pgbo) = self.pgbo {
            pgbo.compute(&self.ns.theta, self.cfg.dt_outer);
            log.pgbo_u_norm = pgbo.u_norm;
            log.pgbo_h_frob = pgbo.h_frob;
        }

        // 4. Micro-cycle
        let h_munu_slice = self.pgbo.as_ref().map(|p| p.h_munu.as_slice());
        self.micro.run_microcycle(
            &mut self.ns.theta,
            &self.ns.w,
            self.cfg.n_micro,
            h_munu_slice,
        );
        let r = self.micro.compute_order_parameter(&self.ns.theta);
        log.r_global = r;

        // 5. Spectral bridge
        self.spectral.compute_eigenpairs(
            &self.ns.w,
            &mut self.ns.eigvals,
            &mut self.ns.eigvecs,
        );

        // 6. Verify eigenvalues
        log.eigval_ordered = self.verify_eigvals();
        log.fiedler_value = SpectralBridge::fiedler_value(&self.ns.eigvals);
        log.spectral_gap = SpectralBridge::spectral_gap(&self.ns.eigvals);

        // TCBO observables
        log.tcbo_p_h1 = self.tcbo_p_h1;
        log.tcbo_gate_open = self.tcbo_p_h1 > 0.72;

        // 7. Compute costs
        let h_slice = self.pgbo.as_ref().map(|p| p.h_munu.as_slice());
        let costs = compute_costs(
            &self.ns.theta,
            &self.ns.w,
            n,
            &self.ns.eigvals,
            &self.ns.eigvecs,
            &self.ns.phi_target,
            &self.ns.protected_mask,
            &self.cfg.cost_weights,
            self.tcbo_p_h1,
            h_slice,
        );
        log.costs = costs;

        // 8. Gradient update on z
        let grad = self.compute_gradient();
        let effective_lr = self.cfg.lr_z * self.l16.state.lr_z_scale;
        for (z_i, g_i) in self.ns.z.iter_mut().zip(grad.iter()) {
            *z_i -= effective_lr * g_i;
        }

        // 9. L16 closure
        let plv_i = 6.min(n.saturating_sub(2));
        let plv_j = 8.min(n.saturating_sub(1));
        let plv = self.micro.compute_plv(&self.ns.theta, plv_i, plv_j);

        let l16_costs = L16CostInputs {
            c7_symbolic: log.costs.c7_symbolic,
            c8_phase: log.costs.c8_phase,
            c10_boundary: log.costs.c10_boundary,
            p_h1: self.tcbo_p_h1,
            h_frob: self.pgbo.as_ref().map_or(0.0, |p| p.h_frob),
        };

        let l16_result = self.l16.step(
            &self.ns.theta,
            &self.ns.eigvecs,
            n, // eigvecs_cols
            &self.ns.phi_target,
            n, // phi_cols
            r,
            plv,
            &l16_costs,
            self.cfg.dt_outer,
        );

        log.h_rec = l16_result.h_rec;
        log.gate_open = l16_result.gate_open;
        log.refusal = l16_result.refusal;

        // 10. Adaptive PGBO kappa scaling
        if let Some(ref mut pgbo) = self.pgbo {
            if !l16_result.refusal {
                let new_kappa = pgbo.cfg.kappa * (0.9 + 0.2 * r);
                pgbo.cfg.kappa = new_kappa.clamp(pgbo.cfg.kappa_min, pgbo.cfg.kappa_max);
            }
        }

        self.log.push(log.clone());
        self.step_count += 1;
        log
    }

    /// Run multiple outer-cycle steps.
    pub fn run(&mut self, n_outer: usize) -> Vec<SSGFStepLog> {
        let mut logs = Vec::with_capacity(n_outer);
        for _ in 0..n_outer {
            logs.push(self.outer_step());
        }
        logs
    }

    /// Current step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Current global order parameter.
    pub fn r_global(&self) -> f64 {
        self.micro.compute_order_parameter(&self.ns.theta)
    }

    /// Get audio mapping from current state.
    pub fn audio_mapping(&self) -> AudioMapping {
        let r = self.micro.compute_order_parameter(&self.ns.theta);
        let fiedler = SpectralBridge::fiedler_value(&self.ns.eigvals);
        let gap = SpectralBridge::spectral_gap(&self.ns.eigvals);
        let n = self.cfg.n;

        // W density
        let w_density = self.ns.w.iter().filter(|&&v| v > 0.01).count() as f64
            / (n * (n - 1)).max(1) as f64;

        // Layer reads
        let beat_hz = if n > 1 {
            0.5 + 39.5 * self.ns.theta[1].cos().abs()
        } else {
            10.0
        };
        let pulse_hz = if n > 3 {
            1.0 + 19.0 * self.ns.theta[3].cos().abs()
        } else {
            6.0
        };
        let spatial_angle = if n > 6 {
            (self.ns.theta[6].rem_euclid(std::f64::consts::TAU)).to_degrees()
        } else {
            0.0
        };

        let band = if beat_hz < 4.0 {
            "delta"
        } else if beat_hz < 8.0 {
            "theta"
        } else if beat_hz < 12.0 {
            "alpha"
        } else if beat_hz < 30.0 {
            "beta"
        } else {
            "gamma"
        };

        let tcbo_gate = self.tcbo_p_h1 > 0.72;
        let theurgic = tcbo_gate && r > 0.95;

        AudioMapping {
            r_global: r,
            entrainment_intensity: r,
            beat_hz,
            pulse_hz,
            spatial_angle_deg: spatial_angle,
            brainwave_band: band.to_string(),
            fiedler_stability: fiedler,
            spectral_gap: gap,
            geometry_density: w_density,
            l16_gate_open: self.l16.plv_gate_open(),
            l16_refusal: self.l16.state.refusal_active,
            tcbo_p_h1: self.tcbo_p_h1,
            tcbo_gate_open: tcbo_gate,
            pgbo_u_norm: self.pgbo.as_ref().map_or(0.0, |p| p.u_norm),
            pgbo_h_frob: self.pgbo.as_ref().map_or(0.0, |p| p.h_frob),
            theurgic_mode: theurgic,
            healing_acceleration: if theurgic { 1.75 } else { 1.0 },
        }
    }
}

/// Audio mapping from SSGF state to CCW parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioMapping {
    pub r_global: f64,
    pub entrainment_intensity: f64,
    pub beat_hz: f64,
    pub pulse_hz: f64,
    pub spatial_angle_deg: f64,
    pub brainwave_band: String,
    pub fiedler_stability: f64,
    pub spectral_gap: f64,
    pub geometry_density: f64,
    pub l16_gate_open: bool,
    pub l16_refusal: bool,
    pub tcbo_p_h1: f64,
    pub tcbo_gate_open: bool,
    pub pgbo_u_norm: f64,
    pub pgbo_h_frob: f64,
    pub theurgic_mode: bool,
    pub healing_acceleration: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use backfire_physics::params::{build_knm_matrix, N_LAYERS, OMEGA_N};

    fn make_engine() -> SSGFEngine {
        let omega: Vec<f64> = OMEGA_N.to_vec();
        let knm = build_knm_matrix();
        let k: Vec<f64> = knm.iter().flat_map(|row| row.iter().copied()).collect();
        SSGFEngine::new(&omega, &k, SSGFConfig::default())
    }

    #[test]
    fn test_engine_init() {
        let engine = make_engine();
        assert_eq!(engine.cfg.n, N_LAYERS);
        assert_eq!(engine.ns.z.len(), N_LAYERS * (N_LAYERS - 1) / 2);
        assert_eq!(engine.step_count(), 0);
    }

    #[test]
    fn test_w_valid_after_init() {
        let engine = make_engine();
        assert!(engine.verify_w(), "W should be valid after init");
    }

    #[test]
    fn test_eigvals_ordered_after_init() {
        let engine = make_engine();
        assert!(engine.verify_eigvals(), "Eigenvalues should be ordered after init");
    }

    #[test]
    fn test_outer_step() {
        let mut engine = make_engine();
        let log = engine.outer_step();
        assert_eq!(log.step, 0);
        assert!(log.w_valid, "W should be valid");
        assert!(log.eigval_ordered, "Eigenvalues should be ordered");
        assert!(log.r_global >= 0.0 && log.r_global <= 1.0);
        assert_eq!(engine.step_count(), 1);
    }

    #[test]
    fn test_multiple_steps() {
        let mut engine = make_engine();
        let logs = engine.run(5);
        assert_eq!(logs.len(), 5);
        assert_eq!(engine.step_count(), 5);
        // All steps should have valid W
        assert!(logs.iter().all(|l| l.w_valid));
        // All steps should have ordered eigenvalues
        assert!(logs.iter().all(|l| l.eigval_ordered));
    }

    #[test]
    fn test_cost_decreasing_tendency() {
        let mut engine = SSGFEngine::new(
            &OMEGA_N.to_vec(),
            &build_knm_matrix()
                .iter()
                .flat_map(|r| r.iter().copied())
                .collect::<Vec<_>>(),
            SSGFConfig {
                lr_z: 0.05,
                noise_amp: 0.0,
                pgbo_enabled: false,
                n_micro: 20,
                ..SSGFConfig::default()
            },
        );
        let logs = engine.run(20);
        // U_total should show some downward pressure (not strict due to stochasticity)
        let first_5_avg: f64 = logs[..5].iter().map(|l| l.costs.u_total).sum::<f64>() / 5.0;
        let last_5_avg: f64 = logs[15..].iter().map(|l| l.costs.u_total).sum::<f64>() / 5.0;
        // Allow tolerance — gradient descent should reduce cost on average
        assert!(
            last_5_avg < first_5_avg + 2.0,
            "Cost should not increase dramatically: first_5={first_5_avg:.4}, last_5={last_5_avg:.4}"
        );
    }

    #[test]
    fn test_audio_mapping() {
        let mut engine = make_engine();
        engine.outer_step();
        let mapping = engine.audio_mapping();
        assert!(mapping.r_global >= 0.0 && mapping.r_global <= 1.0);
        assert!(mapping.beat_hz >= 0.5 && mapping.beat_hz <= 40.0);
        assert!(mapping.pulse_hz >= 1.0 && mapping.pulse_hz <= 20.0);
        assert!(["delta", "theta", "alpha", "beta", "gamma"]
            .contains(&mapping.brainwave_band.as_str()));
    }

    #[test]
    fn test_tcbo_injection() {
        let mut engine = make_engine();
        engine.set_tcbo_p_h1(0.85);
        let log = engine.outer_step();
        assert!((log.tcbo_p_h1 - 0.85).abs() < 1e-12);
        assert!(log.tcbo_gate_open);
    }

    #[test]
    fn test_theurgic_mode_requires_both() {
        let mut engine = make_engine();
        // Low p_h1, any R → not theurgic
        engine.set_tcbo_p_h1(0.5);
        engine.outer_step();
        let mapping = engine.audio_mapping();
        assert!(!mapping.theurgic_mode);
    }
}
