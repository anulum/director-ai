// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Score Types
// Mirrors: src/director_ai/core/types.py + streaming.py
// ─────────────────────────────────────────────────────────────────────

use serde::{Deserialize, Serialize};

/// Clamp a value to [lo, hi], mapping NaN to lo and Inf to nearest bound.
///
/// Mirrors `_clamp()` from `types.py:16-25`.
#[inline]
pub fn clamp_score(value: f64, lo: f64, hi: f64) -> f64 {
    if value.is_nan() {
        log::warn!("clamp_score: NaN detected, clamping to {lo:.4}");
        return lo;
    }
    if value.is_infinite() {
        let boundary = if value > 0.0 { hi } else { lo };
        log::warn!("clamp_score: Inf detected, clamping to {boundary:.4}");
        return boundary;
    }
    value.clamp(lo, hi)
}

/// Result of a coherence check on generated output.
///
/// Mirrors `CoherenceScore` from `types.py:28-40`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceScore {
    /// Composite coherence score: 0.0 = incoherent, 1.0 = perfect.
    pub score: f64,
    /// Whether the output passes the threshold.
    pub approved: bool,
    /// Logical divergence (NLI contradiction probability).
    pub h_logical: f64,
    /// Factual divergence (ground truth deviation).
    pub h_factual: f64,
}

impl CoherenceScore {
    pub fn new(score: f64, approved: bool, h_logical: f64, h_factual: f64) -> Self {
        Self {
            score: clamp_score(score, 0.0, 1.0),
            approved,
            h_logical: clamp_score(h_logical, 0.0, 1.0),
            h_factual: clamp_score(h_factual, 0.0, 1.0),
        }
    }
}

/// Full review outcome from the CoherenceAgent pipeline.
///
/// Mirrors `ReviewResult` from `types.py:43-54`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewResult {
    /// Final output text (or halt message).
    pub output: String,
    /// Score of the selected candidate (None if halted with no candidate).
    pub coherence: Option<CoherenceScore>,
    /// True if the system refused to emit output.
    pub halted: bool,
    /// Number of candidates scored.
    pub candidates_evaluated: u32,
}

/// A single token event in the stream.
///
/// Mirrors `TokenEvent` from `streaming.py:29-36`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    /// The token string.
    pub token: String,
    /// Token index in the stream.
    pub index: u32,
    /// Coherence score at this token.
    pub coherence: f64,
    /// Monotonic timestamp (seconds since session start).
    pub timestamp_s: f64,
    /// Whether this token triggered a halt.
    pub halted: bool,
}

/// Tracks state of a streaming oversight session.
///
/// Mirrors `StreamSession` from `streaming.py:40-76`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamSession {
    pub tokens: Vec<String>,
    pub events: Vec<TokenEvent>,
    pub coherence_history: Vec<f64>,
    pub halted: bool,
    pub halt_index: i32,
    pub halt_reason: String,
    pub start_time_s: f64,
    pub end_time_s: f64,
}

impl StreamSession {
    /// Assembled output text (truncated at halt point if halted).
    pub fn output(&self) -> String {
        if self.halted && self.halt_index >= 0 {
            self.tokens[..self.halt_index as usize].join("")
        } else {
            self.tokens.join("")
        }
    }

    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn avg_coherence(&self) -> f64 {
        if self.coherence_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.coherence_history.iter().sum();
        clamp_score(sum / self.coherence_history.len() as f64, 0.0, 1.0)
    }

    pub fn min_coherence(&self) -> f64 {
        self.coherence_history
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .max(0.0)
    }

    pub fn duration_ms(&self) -> f64 {
        (self.end_time_s - self.start_time_s) * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_nan() {
        assert_eq!(clamp_score(f64::NAN, 0.0, 1.0), 0.0);
    }

    #[test]
    fn test_clamp_pos_inf() {
        assert_eq!(clamp_score(f64::INFINITY, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_clamp_neg_inf() {
        assert_eq!(clamp_score(f64::NEG_INFINITY, 0.0, 1.0), 0.0);
    }

    #[test]
    fn test_clamp_normal() {
        assert_eq!(clamp_score(0.75, 0.0, 1.0), 0.75);
    }

    #[test]
    fn test_clamp_above_hi() {
        assert_eq!(clamp_score(1.5, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_clamp_below_lo() {
        assert_eq!(clamp_score(-0.3, 0.0, 1.0), 0.0);
    }

    #[test]
    fn test_coherence_score_clamps() {
        let cs = CoherenceScore::new(1.5, true, -0.1, f64::NAN);
        assert_eq!(cs.score, 1.0);
        assert_eq!(cs.h_logical, 0.0);
        assert_eq!(cs.h_factual, 0.0);
    }

    #[test]
    fn test_stream_session_output_halted() {
        let session = StreamSession {
            tokens: vec!["Hello ".into(), "world".into(), " bad".into()],
            halted: true,
            halt_index: 2,
            ..Default::default()
        };
        assert_eq!(session.output(), "Hello world");
    }

    #[test]
    fn test_stream_session_output_normal() {
        let session = StreamSession {
            tokens: vec!["Hello ".into(), "world".into()],
            halted: false,
            halt_index: -1,
            ..Default::default()
        };
        assert_eq!(session.output(), "Hello world");
    }

    #[test]
    fn test_stream_session_avg_coherence() {
        let session = StreamSession {
            coherence_history: vec![0.8, 0.6, 0.7],
            ..Default::default()
        };
        assert!((session.avg_coherence() - 0.7).abs() < 1e-9);
    }
}
