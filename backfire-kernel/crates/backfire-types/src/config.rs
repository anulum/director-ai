// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Configuration
// Mirrors scoring-relevant subset of: src/director_ai/core/config.py
// ─────────────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

use crate::error::{BackfireError, BackfireResult};

/// Runtime configuration for the Backfire Kernel.
///
/// Contains only the parameters needed for the hot-path safety gate.
/// The full `DirectorConfig` stays in Python; this is the subset
/// that crosses the FFI boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfireConfig {
    /// Approval threshold: coherence must be >= this to pass.
    /// Default: 0.6 (from Python DirectorConfig.coherence_threshold).
    pub coherence_threshold: f64,

    /// Absolute safety floor: token stream severed if score drops below.
    /// Default: 0.5 (from Python DirectorConfig.hard_limit).
    pub hard_limit: f64,

    /// Weight for logical divergence in composite score.
    /// Default: 0.6 (from Python CoherenceScorer.W_LOGIC).
    pub w_logic: f64,

    /// Weight for factual divergence in composite score.
    /// Default: 0.4 (from Python CoherenceScorer.W_FACT).
    pub w_fact: f64,

    /// Number of tokens in sliding coherence window.
    /// Default: 10 (from Python StreamingKernel.window_size).
    pub window_size: usize,

    /// Halt if sliding window average drops below this.
    /// Default: 0.55 (from Python StreamingKernel.window_threshold).
    pub window_threshold: f64,

    /// Tokens to check for downward trend.
    /// Default: 5 (from Python StreamingKernel.trend_window).
    pub trend_window: usize,

    /// Halt if coherence drops more than this over trend window.
    /// Default: 0.15 (from Python StreamingKernel.trend_threshold).
    pub trend_threshold: f64,

    /// Scoring history window size.
    /// Default: 5 (from Python CoherenceScorer.history_window).
    pub history_window: usize,

    /// Maximum scoring deadline in milliseconds.
    /// Default: 50 (from Backfire Prevention Protocols §2.2).
    pub deadline_ms: u64,

    /// Logit entropy limit (bits per token). Stream terminated above this.
    /// Default: 1.2 (from Backfire Prevention Protocols §2.1).
    pub logit_entropy_limit: f64,
}

impl Default for BackfireConfig {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.6,
            hard_limit: 0.5,
            w_logic: 0.6,
            w_fact: 0.4,
            window_size: 10,
            window_threshold: 0.55,
            trend_window: 5,
            trend_threshold: 0.15,
            history_window: 5,
            deadline_ms: 50,
            logit_entropy_limit: 1.2,
        }
    }
}

impl BackfireConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> BackfireResult<()> {
        if !(0.0..=1.0).contains(&self.coherence_threshold) {
            return Err(BackfireError::Config(format!(
                "coherence_threshold must be in [0, 1], got {}",
                self.coherence_threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.hard_limit) {
            return Err(BackfireError::Config(format!(
                "hard_limit must be in [0, 1], got {}",
                self.hard_limit
            )));
        }
        if (self.w_logic + self.w_fact - 1.0).abs() > 1e-9 {
            return Err(BackfireError::Config(format!(
                "w_logic + w_fact must equal 1.0, got {} + {} = {}",
                self.w_logic,
                self.w_fact,
                self.w_logic + self.w_fact
            )));
        }
        if self.window_size < 1 {
            return Err(BackfireError::Config(format!(
                "window_size must be >= 1, got {}",
                self.window_size
            )));
        }
        if self.trend_window < 2 {
            return Err(BackfireError::Config(format!(
                "trend_window must be >= 2, got {}",
                self.trend_window
            )));
        }
        if self.trend_threshold <= 0.0 {
            return Err(BackfireError::Config(format!(
                "trend_threshold must be > 0, got {}",
                self.trend_threshold
            )));
        }
        if self.deadline_ms == 0 {
            return Err(BackfireError::Config(
                "deadline_ms must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Load from JSON string.
    pub fn from_json(json: &str) -> BackfireResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| BackfireError::Config(format!("JSON parse error: {e}")))
    }
}
