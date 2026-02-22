// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Safety Kernel (Output Gate) + Streaming Kernel
// Mirrors: src/director_ai/core/kernel.py + streaming.py
// ─────────────────────────────────────────────────────────────────────
//! Token-level safety gate that monitors coherence in real-time
//! and severs the output stream if coherence drops below the hard
//! safety limit.
//!
//! Two variants:
//! - `SafetyKernel` — basic per-token hard limit check.
//! - `StreamingKernel` — sliding window + downward trend detection.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use backfire_types::score::TokenEvent;
use backfire_types::{BackfireConfig, StreamSession};

/// Coherence callback: given a token, returns the current coherence score.
pub type CoherenceCallback<'a> = &'a dyn Fn(&str) -> f64;

/// Basic safety kernel — per-token hard limit enforcement.
///
/// Mirrors `SafetyKernel` from `kernel.py:12-69`.
pub struct SafetyKernel {
    hard_limit: f64,
    is_active: AtomicBool,
}

impl SafetyKernel {
    pub fn new(hard_limit: f64) -> Self {
        Self {
            hard_limit,
            is_active: AtomicBool::new(true),
        }
    }

    pub fn from_config(config: &BackfireConfig) -> Self {
        Self::new(config.hard_limit)
    }

    /// Emit output tokens while monitoring coherence in real-time.
    ///
    /// Returns assembled output string, or interrupt message if halted.
    /// Mirrors `stream_output()` from `kernel.py:27-59`.
    pub fn stream_output(
        &self,
        tokens: &[&str],
        coherence_callback: CoherenceCallback<'_>,
    ) -> String {
        let mut output_buffer = Vec::with_capacity(tokens.len());

        for token in tokens {
            let current_score = match std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| coherence_callback(token)),
            ) {
                Ok(s) => {
                    if s.is_finite() {
                        s
                    } else {
                        0.0
                    }
                }
                Err(_) => {
                    log::error!("Coherence callback panicked — treating as score=0");
                    0.0
                }
            };

            if current_score < self.hard_limit {
                self.emergency_stop();
                return "[KERNEL INTERRUPT: COHERENCE LIMIT EXCEEDED]".to_string();
            }

            output_buffer.push(*token);
        }

        output_buffer.join("")
    }

    /// Halt the inference engine.
    pub fn emergency_stop(&self) {
        log::error!(">>> SAFETY KERNEL ACTIVATED: INFERENCE HALTED <<<");
        self.is_active.store(false, Ordering::SeqCst);
    }

    /// Re-enable the kernel after a previous halt.
    pub fn reactivate(&self) {
        self.is_active.store(true, Ordering::SeqCst);
        log::info!("Safety kernel reactivated");
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::SeqCst)
    }
}

/// Streaming token-by-token safety kernel with sliding window oversight.
///
/// Extends `SafetyKernel` with:
/// - Sliding window average check
/// - Downward trend detection
/// - Full `StreamSession` trace
///
/// Mirrors `StreamingKernel` from `streaming.py:79-214`.
pub struct StreamingKernel {
    config: BackfireConfig,
    is_active: AtomicBool,
}

impl StreamingKernel {
    pub fn new(config: BackfireConfig) -> Self {
        Self {
            config,
            is_active: AtomicBool::new(true),
        }
    }

    /// Process tokens with full streaming oversight.
    ///
    /// Returns a `StreamSession` with the complete oversight trace.
    /// Mirrors `stream_tokens()` from `streaming.py:118-207`.
    pub fn stream_tokens(
        &self,
        tokens: &[&str],
        coherence_callback: CoherenceCallback<'_>,
    ) -> StreamSession {
        let start = Instant::now();
        let mut session = StreamSession {
            start_time_s: 0.0,
            ..Default::default()
        };
        let mut window: VecDeque<f64> = VecDeque::with_capacity(self.config.window_size);

        for (i, token) in tokens.iter().enumerate() {
            if !self.is_active.load(Ordering::SeqCst) {
                session.halted = true;
                session.halt_index = i as i32;
                session.halt_reason = "kernel_inactive".to_string();
                break;
            }

            let score = match std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| coherence_callback(token)),
            ) {
                Ok(s) if s.is_finite() => s,
                _ => 0.0,
            };

            let elapsed = start.elapsed().as_secs_f64();

            let mut event = TokenEvent {
                token: token.to_string(),
                index: i as u32,
                coherence: score,
                timestamp_s: elapsed,
                halted: false,
            };

            session.tokens.push(token.to_string());
            session.coherence_history.push(score);
            window.push_back(score);
            if window.len() > self.config.window_size {
                window.pop_front();
            }

            // Check 1: Hard limit
            if score < self.config.hard_limit {
                event.halted = true;
                session.halted = true;
                session.halt_index = i as i32;
                session.halt_reason =
                    format!("hard_limit ({score:.4} < {})", self.config.hard_limit);
                self.is_active.store(false, Ordering::SeqCst);
                session.events.push(event);
                break;
            }

            // Check 2: Sliding window average
            if window.len() >= self.config.window_size {
                let avg: f64 = window.iter().sum::<f64>() / window.len() as f64;
                if avg < self.config.window_threshold {
                    event.halted = true;
                    session.halted = true;
                    session.halt_index = i as i32;
                    session.halt_reason =
                        format!("window_avg ({avg:.4} < {})", self.config.window_threshold);
                    session.events.push(event);
                    break;
                }
            }

            // Check 3: Downward trend
            if session.coherence_history.len() >= self.config.trend_window {
                let recent = &session.coherence_history
                    [session.coherence_history.len() - self.config.trend_window..];
                let drop = recent[0] - recent[recent.len() - 1];
                if drop > self.config.trend_threshold {
                    event.halted = true;
                    session.halted = true;
                    session.halt_index = i as i32;
                    session.halt_reason = format!(
                        "downward_trend ({drop:.4} > {})",
                        self.config.trend_threshold
                    );
                    session.events.push(event);
                    break;
                }
            }

            session.events.push(event);
        }

        session.end_time_s = start.elapsed().as_secs_f64();
        session
    }

    /// Backward-compatible string output.
    pub fn stream_output(
        &self,
        tokens: &[&str],
        coherence_callback: CoherenceCallback<'_>,
    ) -> String {
        let session = self.stream_tokens(tokens, coherence_callback);
        if session.halted {
            format!("[KERNEL INTERRUPT: {}]", session.halt_reason)
        } else {
            session.output()
        }
    }

    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::SeqCst)
    }

    pub fn reactivate(&self) {
        self.is_active.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    // ── SafetyKernel tests ────────────────────────────────────────

    #[test]
    fn test_safety_kernel_pass() {
        let kernel = SafetyKernel::new(0.5);
        let output = kernel.stream_output(&["Hello ", "world"], &|_| 0.8);
        assert_eq!(output, "Hello world");
    }

    #[test]
    fn test_safety_kernel_halt() {
        let kernel = SafetyKernel::new(0.5);
        let output = kernel.stream_output(&["Bad ", "output"], &|_| 0.3);
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    #[test]
    fn test_safety_kernel_custom_limit() {
        let kernel = SafetyKernel::new(0.7);
        let output = kernel.stream_output(&["test"], &|_| 0.6);
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    #[test]
    fn test_safety_kernel_reactivate() {
        let kernel = SafetyKernel::new(0.5);
        kernel.emergency_stop();
        assert!(!kernel.is_active());
        kernel.reactivate();
        assert!(kernel.is_active());
    }

    #[test]
    fn test_safety_kernel_callback_panic() {
        let kernel = SafetyKernel::new(0.5);
        let output = kernel.stream_output(&["test"], &|_| panic!("boom"));
        // Panic caught → score=0.0 < 0.5 → halt
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    #[test]
    fn test_safety_kernel_nan_score() {
        let kernel = SafetyKernel::new(0.5);
        let output = kernel.stream_output(&["test"], &|_| f64::NAN);
        // NaN → 0.0 < 0.5 → halt
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    // ── StreamingKernel tests ─────────────────────────────────────

    #[test]
    fn test_streaming_pass_all() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let session = kernel.stream_tokens(
            &["Hello ", "beautiful ", "world"],
            &|_| 0.8,
        );
        assert!(!session.halted);
        assert_eq!(session.output(), "Hello beautiful world");
        assert_eq!(session.token_count(), 3);
    }

    #[test]
    fn test_streaming_hard_limit_halt() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let scores = [0.8, 0.7, 0.3]; // Third token below hard_limit(0.5)
        let idx = Cell::new(0usize);
        let session = kernel.stream_tokens(
            &["Hello ", "world ", "bad"],
            &|_| {
                let i = idx.get();
                let s = scores[i];
                idx.set(i + 1);
                s
            },
        );
        assert!(session.halted);
        assert_eq!(session.halt_index, 2);
        assert!(session.halt_reason.contains("hard_limit"));
    }

    #[test]
    fn test_streaming_window_avg_halt() {
        let mut config = BackfireConfig::default();
        config.window_size = 3;
        config.window_threshold = 0.7;
        let kernel = StreamingKernel::new(config);
        // All tokens at 0.6 → window avg = 0.6 < 0.7 → halt after window fills
        let session = kernel.stream_tokens(
            &["a ", "b ", "c ", "d "],
            &|_| 0.6,
        );
        assert!(session.halted);
        assert!(session.halt_reason.contains("window_avg"));
    }

    #[test]
    fn test_streaming_downward_trend_halt() {
        let mut config = BackfireConfig::default();
        config.trend_window = 3;
        config.trend_threshold = 0.1;
        config.hard_limit = 0.1; // Low so hard limit doesn't trigger
        config.window_threshold = 0.1; // Low so window doesn't trigger
        let kernel = StreamingKernel::new(config);
        let scores = [0.9, 0.8, 0.7, 0.5]; // Drop from 0.8→0.5 = 0.3 > 0.1
        let idx = Cell::new(0usize);
        let session = kernel.stream_tokens(
            &["a ", "b ", "c ", "d "],
            &|_| {
                let i = idx.get();
                let s = scores[i.min(scores.len() - 1)];
                idx.set(i + 1);
                s
            },
        );
        assert!(session.halted);
        assert!(session.halt_reason.contains("downward_trend"));
    }

    #[test]
    fn test_streaming_duration_tracked() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let session = kernel.stream_tokens(&["hello"], &|_| 0.8);
        assert!(session.duration_ms() >= 0.0);
    }

    #[test]
    fn test_streaming_avg_coherence() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let scores = [0.8, 0.6, 0.7];
        let idx = Cell::new(0usize);
        let session = kernel.stream_tokens(
            &["a", "b", "c"],
            &|_| {
                let i = idx.get();
                let s = scores[i];
                idx.set(i + 1);
                s
            },
        );
        assert!((session.avg_coherence() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_streaming_string_output_halted() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let output = kernel.stream_output(&["ok ", "bad"], &|t| {
            if t == "bad" { 0.1 } else { 0.8 }
        });
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    #[test]
    fn test_streaming_string_output_pass() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let output = kernel.stream_output(&["Hello ", "world"], &|_| 0.8);
        assert_eq!(output, "Hello world");
    }
}
