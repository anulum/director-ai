// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — kernel
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Safety Kernel (Output Gate) + Streaming Kernel
// Mirrors: src/director_ai/core/kernel.py + streaming.py
// ─────────────────────────────────────────────────────────────────────
#![deny(missing_docs)]
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

use crate::signals::trend_drop;

/// Coherence callback: given a token, returns the current coherence score.
pub type CoherenceCallback<'a> = &'a dyn Fn(&str) -> f64;

/// Basic safety kernel — per-token hard limit enforcement.
///
/// Mirrors `SafetyKernel` from `kernel.py:12-69`.
pub struct SafetyKernel {
    /// Minimum accepted score for each token.
    hard_limit: f64,
    /// Atomic latch cleared by an interruption and reset by reactivation.
    is_active: AtomicBool,
}

impl SafetyKernel {
    /// Construct an active hard-limit gate for a host-supplied score floor.
    pub fn new(hard_limit: f64) -> Self {
        Self {
            hard_limit,
            is_active: AtomicBool::new(true),
        }
    }

    /// Use the hard limit from a caller-validated native configuration.
    pub fn from_config(config: &BackfireConfig) -> Self {
        Self::new(config.hard_limit)
    }

    /// Emit output tokens while monitoring coherence in real-time.
    ///
    /// Returns assembled output string, or interrupt message if halted.
    /// An inactive gate refuses before scoring another token. Nonfinite
    /// callback results and callback panics become zero before comparison.
    pub fn stream_output(
        &self,
        tokens: &[&str],
        coherence_callback: CoherenceCallback<'_>,
    ) -> String {
        let mut output_buffer = Vec::with_capacity(tokens.len());

        for token in tokens {
            if !self.is_active.load(Ordering::SeqCst) {
                return "[KERNEL INTERRUPT: KERNEL INACTIVE]".to_string();
            }
            let current_score = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                coherence_callback(token)
            })) {
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

    /// Stop subsequent token-gate iterations until explicit reactivation.
    pub fn emergency_stop(&self) {
        log::error!(">>> SAFETY KERNEL ACTIVATED: INFERENCE HALTED <<<");
        self.is_active.store(false, Ordering::SeqCst);
    }

    /// Re-enable the kernel after a previous halt.
    pub fn reactivate(&self) {
        self.is_active.store(true, Ordering::SeqCst);
        log::info!("Safety kernel reactivated");
    }

    /// Return whether this gate is armed to process another token.
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
    /// Thresholds and window sizes used for streaming decisions.
    config: BackfireConfig,
    /// Atomic latch that rejects later streams after any halt until reactivation.
    is_active: AtomicBool,
}

impl StreamingKernel {
    /// Construct an active stream gate from a caller-validated configuration.
    pub fn new(config: BackfireConfig) -> Self {
        Self {
            config,
            is_active: AtomicBool::new(true),
        }
    }

    /// Process tokens with full streaming oversight.
    ///
    /// Returns a `StreamSession` with the complete oversight trace.
    /// The trend uses the shared least-squares projected drop over the recent
    /// window. Every halt deactivates the gate until explicit reactivation.
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

            let score = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                coherence_callback(token)
            })) {
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
                    self.is_active.store(false, Ordering::SeqCst);
                    session.events.push(event);
                    break;
                }
            }

            // Check 3: Downward trend
            if session.coherence_history.len() >= self.config.trend_window {
                let recent = &session.coherence_history
                    [session.coherence_history.len() - self.config.trend_window..];
                let drop = trend_drop(recent);
                if drop > self.config.trend_threshold {
                    event.halted = true;
                    session.halted = true;
                    session.halt_index = i as i32;
                    session.halt_reason = format!(
                        "downward_trend ({drop:.4} > {})",
                        self.config.trend_threshold
                    );
                    self.is_active.store(false, Ordering::SeqCst);
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

    /// Return whether another stream can be processed without reactivation.
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::SeqCst)
    }

    /// Re-arm the gate so a new stream can start after any halt.
    pub fn reactivate(&self) {
        self.is_active.store(true, Ordering::SeqCst);
    }
}

/// Exercises callback failure, latch transitions and streaming decisions on the native API.
#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    /// Feed one generated token per score through the native callback interface.
    fn stream_scores(kernel: &StreamingKernel, scores: &[f64]) -> StreamSession {
        let tokens: Vec<String> = (0..scores.len()).map(|i| format!("token-{i}")).collect();
        let token_refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let index = Cell::new(0);
        kernel.stream_tokens(&token_refs, &|_| {
            let i = index.get();
            index.set(i + 1);
            scores[i]
        })
    }

    /// Check that the fitted decline can cross the threshold with equal endpoint scores.
    #[test]
    fn regression_detects_decline_with_equal_endpoints() {
        let kernel = StreamingKernel::new(BackfireConfig::default());
        let session = stream_scores(&kernel, &[0.8, 1.0, 0.8, 0.5, 0.8]);
        assert!(session.halted);
        assert!(session.halt_reason.starts_with("downward_trend (0.2000"));
        assert_eq!(session.halt_index, 4);
    }

    /// Check that falling endpoints can still pass when the fitted trend rises.
    #[test]
    fn regression_does_not_halt_on_declining_endpoints_with_rising_trend() {
        let kernel = StreamingKernel::new(BackfireConfig::default());
        let session = stream_scores(&kernel, &[0.9, 0.5, 0.8, 1.0, 0.7]);
        assert!(!session.halted);
        assert_eq!(session.token_count(), 5);
    }

    /// Halt on the latest low scores even when earlier high scores keep the full mean above the floor.
    #[test]
    fn recent_window_halts_even_when_full_history_average_passes() {
        let config = BackfireConfig {
            window_size: 3,
            window_threshold: 0.7,
            trend_threshold: 1.0,
            ..BackfireConfig::default()
        };
        let kernel = StreamingKernel::new(config);
        let scores = [0.9, 0.9, 0.9, 0.6, 0.6, 0.6];
        let session = stream_scores(&kernel, &scores);
        assert!(session.halted);
        assert_eq!(session.halt_index, 5);
        assert_eq!(session.coherence_history, scores);
        assert!(session.halt_reason.starts_with("window_avg (0.6000"));
        assert!(!kernel.is_active());
    }

    /// Capture logger output for a panicking callback, its interruption and reactivation.
    #[test]
    fn safety_kernel_emits_real_callback_failure_halt_and_reactivation_logs() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "backfire-kernel-{}-{stamp}.log",
            std::process::id()
        ));
        let sink = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp(None)
            .target(env_logger::Target::Pipe(Box::new(sink)))
            .try_init()
            .unwrap();
        let kernel = SafetyKernel::new(0.5);
        assert!(kernel
            .stream_output(&["unscored"], &|_| panic!("score producer failed"))
            .contains("KERNEL INTERRUPT"));
        kernel.reactivate();
        log::logger().flush();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Coherence callback panicked"));
        assert!(content.contains("SAFETY KERNEL ACTIVATED"));
        assert!(content.contains("Safety kernel reactivated"));
        eprintln!("Native kernel log {}:\n{content}", path.display());
    }

    /// Check that emergency stop prevents callback invocation until reactivation.
    #[test]
    fn safety_kernel_emergency_stop_blocks_callback_until_reactivation() {
        let kernel = SafetyKernel::new(0.5);
        let calls = Cell::new(0);
        let score = |_: &str| {
            calls.set(calls.get() + 1);
            0.9
        };
        kernel.emergency_stop();
        assert!(kernel
            .stream_output(&["blocked"], &score)
            .contains("KERNEL INTERRUPT"));
        assert_eq!(calls.get(), 0);
        kernel.reactivate();
        assert_eq!(kernel.stream_output(&["safe"], &score), "safe");
        assert_eq!(calls.get(), 1);
    }

    /// Check that a hard-limit interruption rejects the next stream even with a passing score.
    #[test]
    fn safety_kernel_hard_halt_blocks_a_later_passing_stream() {
        let kernel = SafetyKernel::new(0.5);
        assert!(kernel
            .stream_output(&["bad"], &|_| 0.1)
            .contains("KERNEL INTERRUPT"));
        assert!(!kernel.is_active());
        assert!(kernel
            .stream_output(&["must-not-pass"], &|_| 0.9)
            .contains("KERNEL INTERRUPT"));
    }

    /// Verify the latch and restart for hard-limit, average and trend interruptions.
    #[test]
    fn every_streaming_halt_requires_reactivation_before_another_stream() {
        for scores in [vec![0.1], vec![0.54, 0.54, 0.54], vec![0.95, 0.85, 0.55]] {
            let config = BackfireConfig {
                window_size: 3,
                trend_window: 3,
                trend_threshold: 0.2,
                ..BackfireConfig::default()
            };
            let kernel = StreamingKernel::new(config);
            let halted = stream_scores(&kernel, &scores);
            assert!(halted.halted);
            assert!(!kernel.is_active());
            let blocked = stream_scores(&kernel, &[0.9]);
            assert!(blocked.halted);
            assert!(blocked.tokens.is_empty());
            assert_eq!(blocked.halt_reason, "kernel_inactive");
            kernel.reactivate();
            let resumed = stream_scores(&kernel, &[0.9]);
            assert!(!resumed.halted);
            assert_eq!(resumed.token_count(), 1);
        }
    }

    // ── SafetyKernel tests ────────────────────────────────────────

    /// Join tokens whose scores satisfy the per-token floor.
    #[test]
    fn test_safety_kernel_pass() {
        let kernel = SafetyKernel::new(0.5);
        let output = kernel.stream_output(&["Hello ", "world"], &|_| 0.8);
        assert_eq!(output, "Hello world");
    }

    /// Return the interruption marker when a score falls below the floor.
    #[test]
    fn test_safety_kernel_halt() {
        let kernel = SafetyKernel::new(0.5);
        let output = kernel.stream_output(&["Bad ", "output"], &|_| 0.3);
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    /// Apply the supplied score floor to an otherwise passing score.
    #[test]
    fn test_safety_kernel_custom_limit() {
        let kernel = SafetyKernel::new(0.7);
        let output = kernel.stream_output(&["test"], &|_| 0.6);
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    /// Restore the active flag after an emergency stop.
    #[test]
    fn test_safety_kernel_reactivate() {
        let kernel = SafetyKernel::new(0.5);
        kernel.emergency_stop();
        assert!(!kernel.is_active());
        kernel.reactivate();
        assert!(kernel.is_active());
    }

    /// Interrupt when a panicking callback yields the zero fallback.
    #[test]
    fn test_safety_kernel_callback_panic() {
        let kernel = SafetyKernel::new(0.5);
        let output = kernel.stream_output(&["test"], &|_| panic!("boom"));
        // Panic caught → score=0.0 < 0.5 → halt
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    /// Interrupt when a NaN callback yields the zero fallback.
    #[test]
    fn test_safety_kernel_nan_score() {
        let kernel = SafetyKernel::new(0.5);
        let output = kernel.stream_output(&["test"], &|_| f64::NAN);
        // NaN → 0.0 < 0.5 → halt
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    // ── StreamingKernel tests ─────────────────────────────────────

    /// Retain the full output and token count for a passing stream.
    #[test]
    fn test_streaming_pass_all() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let session = kernel.stream_tokens(&["Hello ", "beautiful ", "world"], &|_| 0.8);
        assert!(!session.halted);
        assert_eq!(session.output(), "Hello beautiful world");
        assert_eq!(session.token_count(), 3);
    }

    /// Record the index and reason of the first sub-floor token.
    #[test]
    fn test_streaming_hard_limit_halt() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let scores = [0.8, 0.7, 0.3]; // Third token below hard_limit(0.5)
        let idx = Cell::new(0usize);
        let session = kernel.stream_tokens(&["Hello ", "world ", "bad"], &|_| {
            let i = idx.get();
            let s = scores[i];
            idx.set(i + 1);
            s
        });
        assert!(session.halted);
        assert_eq!(session.halt_index, 2);
        assert!(session.halt_reason.contains("hard_limit"));
    }

    /// Halt when a full window fails the average despite passing token scores.
    #[test]
    fn test_streaming_window_avg_halt() {
        let mut config = BackfireConfig::default();
        config.window_size = 3;
        config.window_threshold = 0.7;
        let kernel = StreamingKernel::new(config);
        // All tokens at 0.6 → window avg = 0.6 < 0.7 → halt after window fills
        let session = kernel.stream_tokens(&["a ", "b ", "c ", "d "], &|_| 0.6);
        assert!(session.halted);
        assert!(session.halt_reason.contains("window_avg"));
    }

    /// Isolate the trend halt by lowering the hard and average floors.
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
        let session = kernel.stream_tokens(&["a ", "b ", "c ", "d "], &|_| {
            let i = idx.get();
            let s = scores[i.min(scores.len() - 1)];
            idx.set(i + 1);
            s
        });
        assert!(session.halted);
        assert!(session.halt_reason.contains("downward_trend"));
    }

    /// Check that the session exposes a nonnegative elapsed duration.
    #[test]
    fn test_streaming_duration_tracked() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let session = kernel.stream_tokens(&["hello"], &|_| 0.8);
        assert!(session.duration_ms() >= 0.0);
    }

    /// Compute the arithmetic mean over the recorded scores.
    #[test]
    fn test_streaming_avg_coherence() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let scores = [0.8, 0.6, 0.7];
        let idx = Cell::new(0usize);
        let session = kernel.stream_tokens(&["a", "b", "c"], &|_| {
            let i = idx.get();
            let s = scores[i];
            idx.set(i + 1);
            s
        });
        assert!((session.avg_coherence() - 0.7).abs() < 1e-9);
    }

    /// Include the interruption marker in the string form of a halted stream.
    #[test]
    fn test_streaming_string_output_halted() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let output = kernel.stream_output(&["ok ", "bad"], &|t| {
            if t == "bad" {
                0.1
            } else {
                0.8
            }
        });
        assert!(output.contains("KERNEL INTERRUPT"));
    }

    /// Return the joined tokens from the passing string output path.
    #[test]
    fn test_streaming_string_output_pass() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let output = kernel.stream_output(&["Hello ", "world"], &|_| 0.8);
        assert_eq!(output, "Hello world");
    }

    /// Check that the basic gate applies the hard limit read from the shared config.
    #[test]
    fn test_safety_kernel_from_config() {
        let config = BackfireConfig {
            hard_limit: 0.42,
            ..BackfireConfig::default()
        };
        let kernel = SafetyKernel::from_config(&config);
        assert!(kernel.is_active());
        let output = kernel.stream_output(&["ok"], &|_| 0.45);
        assert_eq!(output, "ok");
        let output2 = kernel.stream_output(&["fail"], &|_| 0.41);
        assert!(output2.contains("KERNEL INTERRUPT"));
    }

    /// Check the active flag before a halt, after it and after reactivation.
    #[test]
    fn test_streaming_is_active_and_reactivate() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        assert!(kernel.is_active());
        let stopped = kernel.stream_tokens(&["stop"], &|_| 0.0);
        assert!(stopped.halted);
        assert!(!kernel.is_active());
        kernel.reactivate();
        assert!(kernel.is_active());
    }

    /// Report an inactive-session halt at index zero without accepting tokens.
    #[test]
    fn test_streaming_inactive_kernel_halts_immediately() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let stopped = kernel.stream_tokens(&["stop"], &|_| 0.0);
        assert!(stopped.halted);
        let session = kernel.stream_tokens(&["a", "b", "c"], &|_| 0.9);
        assert!(session.halted);
        assert_eq!(session.halt_index, 0);
        assert!(session.halt_reason.contains("kernel_inactive"));
        assert!(session.tokens.is_empty());
    }

    /// Route a panicking callback to a zero score and the hard-limit halt.
    #[test]
    fn test_streaming_callback_panic_halts() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let session = kernel.stream_tokens(&["trigger_panic"], &|_| panic!("callback exploded"));
        assert!(session.halted);
        assert!(session.halt_reason.contains("hard_limit"));
    }

    /// Route a NaN callback result to a zero score and the hard-limit halt.
    #[test]
    fn test_streaming_nan_callback_halts() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let session = kernel.stream_tokens(&["nan_token"], &|_| f64::NAN);
        assert!(session.halted);
        assert!(session.halt_reason.contains("hard_limit"));
    }

    /// Return an empty, passing session for an empty slice.
    #[test]
    fn test_streaming_empty_tokens() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let session = kernel.stream_tokens(&[], &|_| 0.8);
        assert!(!session.halted);
        assert_eq!(session.token_count(), 0);
        assert_eq!(session.output(), "");
    }

    /// Check ordered event tokens, zero-based indices and the passing flag.
    #[test]
    fn test_streaming_events_match_tokens() {
        let config = BackfireConfig::default();
        let kernel = StreamingKernel::new(config);
        let session = kernel.stream_tokens(&["a", "b"], &|_| 0.9);
        assert_eq!(session.events.len(), 2);
        assert_eq!(session.events[0].token, "a");
        assert_eq!(session.events[1].token, "b");
        assert_eq!(session.events[0].index, 0);
        assert_eq!(session.events[1].index, 1);
        assert!(!session.events[0].halted);
    }
}
