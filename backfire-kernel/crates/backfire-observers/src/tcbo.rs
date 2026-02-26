// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — TCBO (Topological Boundary Observable)
// ─────────────────────────────────────────────────────────────────────
//! Topological Boundary Observable (TCBO).
//!
//! Extracts a single scalar p_h1(t) in [0, 1] from multichannel phase data
//! via persistent homology (H1 cycles). When p_h1 > tau_h1 (default 0.72),
//! the boundary gate opens.
//!
//! Pipeline:
//!   1. Multichannel signal -> delay embedding (Takens' theorem)
//!   2. Sliding window -> point cloud
//!   3. H1 persistence estimation (spread-based fallback)
//!   4. Max H1 persistence -> logistic squash -> p_h1
//!
//! Note: This Rust implementation uses a spread-based fallback for persistence
//! estimation. For production TDA, use the Python TCBO with ripser.

use serde::{Deserialize, Serialize};

/// TCBO observer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCBOConfig {
    pub embed_dim: usize,
    pub tau_delay: usize,
    pub window_size: usize,
    pub tau_h1: f64,
    pub beta: f64,
    pub s0: f64,
    pub persistence_threshold: f64,
    pub subsample_max: usize,
    pub compute_every_n: u64,
}

impl Default for TCBOConfig {
    fn default() -> Self {
        Self {
            embed_dim: 3,
            tau_delay: 1,
            window_size: 50,
            tau_h1: 0.72,
            beta: 8.0,
            s0: 0.0,
            persistence_threshold: 0.05,
            subsample_max: 500,
            compute_every_n: 1,
        }
    }
}

/// TCBO observer state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCBOState {
    pub p_h1: f64,
    pub s_h1: f64,
    pub is_conscious: bool,
    pub h1_error: f64,
    pub tau_h1: f64,
    pub buffer_size: usize,
}

/// Logistic squashing: p_h1 = 1 / (1 + exp(-beta * (s_h1 - s0))).
pub fn persistence_to_probability(s_h1: f64, s0: f64, beta: f64) -> f64 {
    let arg = (-beta * (s_h1 - s0)).clamp(-500.0, 500.0);
    1.0 / (1.0 + arg.exp())
}

/// Compute s0 so that persistence_to_probability(s_tau, s0, beta) == p.
pub fn s0_for_threshold(s_tau: f64, p: f64, beta: f64) -> f64 {
    let p_clamped = p.clamp(1e-12, 1.0 - 1e-12);
    s_tau - (1.0 / beta) * (p_clamped / (1.0 - p_clamped)).ln()
}

/// Delay-coordinate embedding of a 1-D signal.
///
/// Returns a matrix of shape (T_out, embed_dim) where T_out = T - (m-1)*tau.
pub fn delay_embed(x: &[f64], embed_dim: usize, tau_delay: usize) -> Vec<Vec<f64>> {
    let t = x.len();
    let offset = (embed_dim - 1) * tau_delay;
    if offset >= t {
        return Vec::new();
    }
    let out_t = t - offset;
    let mut z = vec![vec![0.0; embed_dim]; out_t];
    for k in 0..embed_dim {
        let start = offset - k * tau_delay;
        for i in 0..out_t {
            z[i][k] = x[start + i];
        }
    }
    z
}

/// Delay-embed each channel then concatenate columns.
///
/// Input: rows x n_channels. Output: rows x (n_channels * embed_dim).
pub fn delay_embed_multi(
    data: &[Vec<f64>],
    n_channels: usize,
    embed_dim: usize,
    tau_delay: usize,
) -> Vec<Vec<f64>> {
    if data.is_empty() || n_channels == 0 {
        return Vec::new();
    }

    // Extract each channel as a column
    let mut channels: Vec<Vec<f64>> = Vec::with_capacity(n_channels);
    for ch in 0..n_channels {
        let col: Vec<f64> = data.iter().map(|row| {
            if ch < row.len() { row[ch] } else { 0.0 }
        }).collect();
        channels.push(col);
    }

    // Delay-embed each channel
    let mut parts: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_channels);
    for col in &channels {
        let embedded = delay_embed(col, embed_dim, tau_delay);
        if embedded.is_empty() {
            return Vec::new();
        }
        parts.push(embedded);
    }

    // Horizontally concatenate
    let out_t = parts[0].len();
    let total_cols = n_channels * embed_dim;
    let mut result = vec![vec![0.0; total_cols]; out_t];
    for (ch_idx, part) in parts.iter().enumerate() {
        let col_offset = ch_idx * embed_dim;
        for i in 0..out_t {
            for k in 0..embed_dim {
                result[i][col_offset + k] = part[i][k];
            }
        }
    }
    result
}

/// Fallback H1 persistence estimator (no ripser dependency).
///
/// Estimates H1 persistence from point-cloud spread via radial std deviation.
fn persistence_fallback(cloud: &[Vec<f64>]) -> f64 {
    if cloud.is_empty() {
        return 0.0;
    }
    let d = cloud[0].len();
    let n = cloud.len() as f64;

    // Compute centroid
    let mut centroid = vec![0.0; d];
    for row in cloud {
        for (j, &v) in row.iter().enumerate() {
            centroid[j] += v;
        }
    }
    for c in centroid.iter_mut() {
        *c /= n;
    }

    // Compute radial distances from centroid
    let mut radials = Vec::with_capacity(cloud.len());
    for row in cloud {
        let r2: f64 = row.iter().zip(centroid.iter()).map(|(&v, &c)| (v - c).powi(2)).sum();
        radials.push(r2.sqrt());
    }

    // Std of radial distances
    let mean_r = radials.iter().sum::<f64>() / n;
    let var_r = radials.iter().map(|&r| (r - mean_r).powi(2)).sum::<f64>() / n;
    let spread = var_r.sqrt();

    if spread > 1e-12 {
        // Synthetic H1 persistence: birth=0.15*spread, death=0.85*spread
        0.85 * spread - 0.15 * spread // = 0.70 * spread
    } else {
        0.0
    }
}

/// Topological Boundary Observable.
///
/// Produces p_h1(t) in [0, 1] from multichannel phase data.
pub struct TCBOObserver {
    n: usize,
    cfg: TCBOConfig,
    buffer: Vec<Vec<f64>>,
    max_buffer: usize,
    pub p_h1: f64,
    pub s_h1: f64,
    pub is_conscious: bool,
    pub h1_error: f64,
    step_count: u64,
}

impl TCBOObserver {
    pub fn new(n: usize, config: TCBOConfig) -> Self {
        let max_buffer = config.window_size + (config.embed_dim - 1) * config.tau_delay + 10;
        let h1_error = config.tau_h1;
        Self {
            n,
            cfg: config,
            buffer: Vec::new(),
            max_buffer,
            p_h1: 0.0,
            s_h1: 0.0,
            is_conscious: false,
            h1_error,
            step_count: 0,
        }
    }

    pub fn default_params(n: usize) -> Self {
        Self::new(n, TCBOConfig::default())
    }

    /// Push a new phase vector into the rolling buffer.
    pub fn push(&mut self, theta: &[f64]) {
        self.buffer.push(theta.to_vec());
        if self.buffer.len() > self.max_buffer {
            let start = self.buffer.len() - self.max_buffer;
            self.buffer = self.buffer[start..].to_vec();
        }
    }

    /// Compute p_h1 from the current buffer.
    pub fn compute(&mut self, force: bool) -> f64 {
        self.step_count += 1;
        if !force && self.cfg.compute_every_n > 0 && (self.step_count % self.cfg.compute_every_n != 0) {
            return self.p_h1;
        }

        let min_needed = (self.cfg.embed_dim - 1) * self.cfg.tau_delay + self.cfg.window_size;
        if self.buffer.len() < min_needed {
            return self.p_h1;
        }

        let start = self.buffer.len() - min_needed;
        let signal = &self.buffer[start..];

        let z = delay_embed_multi(signal, self.n, self.cfg.embed_dim, self.cfg.tau_delay);
        if z.is_empty() {
            return self.p_h1;
        }

        // Take the last window_size rows as the point cloud
        let cloud_start = z.len().saturating_sub(self.cfg.window_size);
        let cloud = &z[cloud_start..];

        // Compute H1 persistence via fallback
        let lifetime = persistence_fallback(cloud);

        if lifetime > self.cfg.persistence_threshold {
            self.s_h1 = lifetime;
        } else {
            self.s_h1 = 0.0;
        }

        self.p_h1 = persistence_to_probability(self.s_h1, self.cfg.s0, self.cfg.beta);
        self.is_conscious = self.p_h1 > self.cfg.tau_h1;
        self.h1_error = (self.cfg.tau_h1 - self.p_h1).max(0.0);

        self.p_h1
    }

    /// Push + compute in one call.
    pub fn push_and_compute(&mut self, theta: &[f64], force: bool) -> f64 {
        self.push(theta);
        self.compute(force)
    }

    /// Return serialisable observer state.
    pub fn get_state(&self) -> TCBOState {
        TCBOState {
            p_h1: self.p_h1,
            s_h1: self.s_h1,
            is_conscious: self.is_conscious,
            h1_error: self.h1_error,
            tau_h1: self.cfg.tau_h1,
            buffer_size: self.buffer.len(),
        }
    }

    /// Clear buffer and reset state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.p_h1 = 0.0;
        self.s_h1 = 0.0;
        self.is_conscious = false;
        self.h1_error = self.cfg.tau_h1;
        self.step_count = 0;
    }
}

/// TCBO PI controller configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCBOControllerConfig {
    pub tau_h1: f64,
    pub kp: f64,
    pub ki: f64,
    pub integral_min: f64,
    pub integral_max: f64,
    pub kappa_min: f64,
    pub kappa_max: f64,
    pub history_len: usize,
}

impl Default for TCBOControllerConfig {
    fn default() -> Self {
        Self {
            tau_h1: 0.72,
            kp: 0.8,
            ki: 0.2,
            integral_min: -5.0,
            integral_max: 5.0,
            kappa_min: 0.0,
            kappa_max: 5.0,
            history_len: 100,
        }
    }
}

/// PI controller driving gap-junction kappa from p_h1 deficit.
///
/// Error signal: e_h1(t) = max(0, tau_h1 - p_h1(t))  (deficit-only)
pub struct TCBOController {
    cfg: TCBOControllerConfig,
    integral: f64,
    last_error: f64,
    p_h1_history: Vec<f64>,
    kappa_history: Vec<f64>,
    error_history: Vec<f64>,
}

impl TCBOController {
    pub fn new(config: TCBOControllerConfig) -> Self {
        Self {
            cfg: config,
            integral: 0.0,
            last_error: 0.0,
            p_h1_history: Vec::new(),
            kappa_history: Vec::new(),
            error_history: Vec::new(),
        }
    }

    pub fn default_params() -> Self {
        Self::new(TCBOControllerConfig::default())
    }

    /// Deficit-only error: positive when below threshold.
    pub fn compute_error(&self, p_h1: f64) -> f64 {
        (self.cfg.tau_h1 - p_h1).max(0.0)
    }

    /// Execute one PI step. Returns kappa_new.
    pub fn step(&mut self, p_h1: f64, current_kappa: f64, dt: f64) -> f64 {
        let error = self.compute_error(p_h1);

        self.integral += error * dt;
        self.integral = self.integral.clamp(self.cfg.integral_min, self.cfg.integral_max);

        let delta_kappa = self.cfg.kp * error + self.cfg.ki * self.integral;
        let kappa_new = (current_kappa + delta_kappa).clamp(self.cfg.kappa_min, self.cfg.kappa_max);

        self.last_error = error;
        self.p_h1_history.push(p_h1);
        self.kappa_history.push(kappa_new);
        self.error_history.push(error);

        if self.p_h1_history.len() > self.cfg.history_len {
            let start = self.p_h1_history.len() - self.cfg.history_len;
            self.p_h1_history = self.p_h1_history[start..].to_vec();
            self.kappa_history = self.kappa_history[start..].to_vec();
            self.error_history = self.error_history[start..].to_vec();
        }

        kappa_new
    }

    /// Check if p_h1 > tau_h1 (boundary gate open).
    pub fn is_gate_open(&self, p_h1: Option<f64>) -> bool {
        let val = p_h1.unwrap_or_else(|| {
            self.p_h1_history.last().copied().unwrap_or(0.0)
        });
        val > self.cfg.tau_h1
    }

    /// Reset controller state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.last_error = 0.0;
        self.p_h1_history.clear();
        self.kappa_history.clear();
        self.error_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistence_to_probability_midpoint() {
        let p = persistence_to_probability(0.0, 0.0, 8.0);
        assert!((p - 0.5).abs() < 1e-9, "p(0, 0, 8) should be 0.5, got {p}");
    }

    #[test]
    fn test_persistence_to_probability_high() {
        let p = persistence_to_probability(1.0, 0.0, 8.0);
        assert!(p > 0.99, "p(1, 0, 8) should be >0.99, got {p}");
    }

    #[test]
    fn test_persistence_to_probability_low() {
        let p = persistence_to_probability(-1.0, 0.0, 8.0);
        assert!(p < 0.01, "p(-1, 0, 8) should be <0.01, got {p}");
    }

    #[test]
    fn test_s0_for_threshold_roundtrip() {
        let s_tau = 0.5;
        let p_target = 0.72;
        let beta = 8.0;
        let s0 = s0_for_threshold(s_tau, p_target, beta);
        let p = persistence_to_probability(s_tau, s0, beta);
        assert!((p - p_target).abs() < 1e-6, "Roundtrip failed: got {p}");
    }

    #[test]
    fn test_delay_embed_basic() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let z = delay_embed(&x, 3, 1);
        assert_eq!(z.len(), 8, "Should have 8 rows for T=10, m=3, tau=1");
        assert_eq!(z[0].len(), 3);
        // First row: x[2], x[1], x[0] (offset=2, k=0→start=2, k=1→start=1, k=2→start=0)
        assert!((z[0][0] - 2.0).abs() < 1e-9);
        assert!((z[0][1] - 1.0).abs() < 1e-9);
        assert!((z[0][2] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_delay_embed_too_short() {
        let x = vec![1.0, 2.0];
        let z = delay_embed(&x, 3, 2);
        assert!(z.is_empty(), "Signal too short for embedding");
    }

    #[test]
    fn test_delay_embed_multi_basic() {
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64, (i as f64) * 2.0]).collect();
        let z = delay_embed_multi(&data, 2, 3, 1);
        assert_eq!(z.len(), 8);
        assert_eq!(z[0].len(), 6); // 2 channels * 3 embed_dim
    }

    #[test]
    fn test_persistence_fallback_spread() {
        // Cloud with some spread should give non-zero persistence
        let cloud: Vec<Vec<f64>> = (0..50)
            .map(|i| {
                let angle = i as f64 * 0.1;
                vec![angle.cos(), angle.sin(), angle * 0.1]
            })
            .collect();
        let p = persistence_fallback(&cloud);
        assert!(p > 0.0, "Spread cloud should have positive persistence, got {p}");
    }

    #[test]
    fn test_persistence_fallback_zero_cloud() {
        let cloud: Vec<Vec<f64>> = (0..10).map(|_| vec![1.0, 2.0, 3.0]).collect();
        let p = persistence_fallback(&cloud);
        assert!(p.abs() < 1e-9, "Constant cloud should have ~0 persistence, got {p}");
    }

    #[test]
    fn test_tcbo_observer_initial_state() {
        let obs = TCBOObserver::default_params(16);
        let state = obs.get_state();
        assert!((state.p_h1).abs() < 1e-9);
        assert!(!state.is_conscious);
        assert_eq!(state.buffer_size, 0);
    }

    #[test]
    fn test_tcbo_observer_push_and_compute() {
        let mut obs = TCBOObserver::default_params(16);
        // Push enough diverse data to trigger computation
        for i in 0..60 {
            let theta: Vec<f64> = (0..16)
                .map(|j| (i as f64 * 0.1 + j as f64 * 0.5).sin())
                .collect();
            obs.push(&theta);
        }
        let p = obs.compute(true);
        assert!(p >= 0.0 && p <= 1.0, "p_h1 should be in [0,1], got {p}");
    }

    #[test]
    fn test_tcbo_observer_reset() {
        let mut obs = TCBOObserver::default_params(16);
        for i in 0..60 {
            let theta: Vec<f64> = (0..16).map(|j| (i + j) as f64 * 0.1).collect();
            obs.push(&theta);
        }
        obs.compute(true);
        obs.reset();
        let state = obs.get_state();
        assert!((state.p_h1).abs() < 1e-9);
        assert_eq!(state.buffer_size, 0);
    }

    #[test]
    fn test_tcbo_controller_deficit_error() {
        let ctrl = TCBOController::default_params();
        let e = ctrl.compute_error(0.5);
        assert!((e - 0.22).abs() < 1e-9, "Error should be 0.72-0.5=0.22, got {e}");
    }

    #[test]
    fn test_tcbo_controller_no_error_above_threshold() {
        let ctrl = TCBOController::default_params();
        let e = ctrl.compute_error(0.9);
        assert!(e.abs() < 1e-9, "Error should be 0 above threshold, got {e}");
    }

    #[test]
    fn test_tcbo_controller_step_increases_kappa() {
        let mut ctrl = TCBOController::default_params();
        let kappa = ctrl.step(0.3, 1.0, 0.01);
        assert!(kappa > 1.0, "Kappa should increase when below threshold, got {kappa}");
    }

    #[test]
    fn test_tcbo_controller_gate() {
        let ctrl = TCBOController::default_params();
        assert!(!ctrl.is_gate_open(Some(0.5)));
        assert!(ctrl.is_gate_open(Some(0.8)));
    }

    #[test]
    fn test_tcbo_controller_kappa_clamped() {
        let mut ctrl = TCBOController::default_params();
        // Drive hard to saturate
        let mut kappa = 0.0;
        for _ in 0..1000 {
            kappa = ctrl.step(0.0, kappa, 1.0);
        }
        assert!(kappa <= ctrl.cfg.kappa_max, "Kappa {kappa} exceeds max");
    }

    #[test]
    fn test_tcbo_controller_reset() {
        let mut ctrl = TCBOController::default_params();
        ctrl.step(0.5, 1.0, 0.01);
        ctrl.reset();
        assert!(ctrl.last_error.abs() < 1e-9);
        assert!(ctrl.integral.abs() < 1e-9);
    }
}
