// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Coherence Scorer (Dual-Entropy Oversight)
// Mirrors: src/director_ai/core/scorer.py
// ─────────────────────────────────────────────────────────────────────
//! Dual-entropy coherence scorer for AI output verification.
//!
//! Computes a composite coherence score from two independent signals:
//! - **Logical divergence** (H_logical): NLI contradiction probability.
//! - **Factual divergence** (H_factual): Ground-truth deviation via RAG.
//!
//! The coherence score is `1 - (w_logic * H_logical + w_fact * H_factual)`.
//! When the score falls below `threshold`, the output is rejected.

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;

use backfire_types::score::{clamp_score, CoherenceScore};
use backfire_types::BackfireConfig;

use crate::knowledge::GroundTruthStore;
use crate::nli::NliBackend;

/// Dual-entropy coherence scorer.
///
/// Thread-safe: history mutations are guarded by a `parking_lot::Mutex`.
pub struct CoherenceScorer {
    config: BackfireConfig,
    nli: Arc<dyn NliBackend>,
    knowledge: Arc<dyn GroundTruthStore>,
    history: Mutex<VecDeque<String>>,
}

impl CoherenceScorer {
    pub fn new(
        config: BackfireConfig,
        nli: Arc<dyn NliBackend>,
        knowledge: Arc<dyn GroundTruthStore>,
    ) -> Self {
        Self {
            config,
            nli,
            knowledge,
            history: Mutex::new(VecDeque::new()),
        }
    }

    /// Calculate factual divergence against the ground truth store.
    ///
    /// Returns 0.0 (perfect alignment) to 1.0 (total hallucination).
    /// Mirrors `calculate_factual_divergence()` from `scorer.py:52-77`.
    pub fn calculate_factual_divergence(&self, prompt: &str, text_output: &str) -> f64 {
        let context = match self.knowledge.retrieve_context(prompt) {
            Some(ctx) => ctx,
            None => return 0.5, // Neutral when no store / no match
        };

        // Prototype heuristic checks (NLI replaces these in production)
        if context.contains("16")
            && !text_output.contains("16")
            && text_output.contains("layers")
        {
            return 0.9;
        }

        if context.contains("sky color") {
            if text_output.contains("blue") {
                return 0.1;
            }
            if text_output.contains("green") {
                return 1.0;
            }
        }

        0.1 // Default: consistent if no contradiction detected
    }

    /// Calculate logical divergence via NLI.
    ///
    /// Returns 0.0 (entailment) to 1.0 (contradiction).
    /// Mirrors `calculate_logical_divergence()` from `scorer.py:81-118`.
    pub fn calculate_logical_divergence(&self, prompt: &str, text_output: &str) -> f64 {
        let h = self.nli.score(prompt, text_output);
        if !h.is_finite() {
            log::warn!("NLI returned non-finite score, defaulting to 0.5");
            return 0.5;
        }
        h.clamp(0.0, 1.0)
    }

    /// Compute clamped divergences and heuristic coherence.
    ///
    /// Returns `(h_logic, h_fact, coherence)` with all values in [0, 1].
    /// Mirrors `_heuristic_coherence()` from `scorer.py:127-135`.
    fn compute_coherence(
        &self,
        prompt: &str,
        action: &str,
    ) -> (f64, f64, f64) {
        let h_logic = clamp_score(
            self.calculate_logical_divergence(prompt, action),
            0.0,
            1.0,
        );
        let h_fact = clamp_score(
            self.calculate_factual_divergence(prompt, action),
            0.0,
            1.0,
        );
        let coherence = clamp_score(
            1.0 - (self.config.w_logic * h_logic + self.config.w_fact * h_fact),
            0.0,
            1.0,
        );
        (h_logic, h_fact, coherence)
    }

    /// Score an action and decide whether to approve it.
    ///
    /// Returns `(approved, CoherenceScore)`.
    /// Mirrors `review()` from `scorer.py:181-189`.
    pub fn review(&self, prompt: &str, action: &str) -> (bool, CoherenceScore) {
        let (h_logic, h_fact, coherence) = self.compute_coherence(prompt, action);
        let approved = coherence >= self.config.coherence_threshold;

        if !approved {
            log::error!(
                "COHERENCE FAILURE. Score: {coherence:.4} < Threshold: {}",
                self.config.coherence_threshold
            );
        } else {
            let mut history = self.history.lock();
            history.push_back(action.to_string());
            if history.len() > self.config.history_window {
                history.pop_front();
            }
        }

        let score = CoherenceScore::new(coherence, approved, h_logic, h_fact);
        (approved, score)
    }

    /// Compute composite divergence (lower is better).
    ///
    /// Weighted sum: `0.6 * H_logical + 0.4 * H_factual`.
    /// Mirrors `compute_divergence()` from `scorer.py:165-179`.
    pub fn compute_divergence(&self, prompt: &str, action: &str) -> f64 {
        let h_logic = self.calculate_logical_divergence(prompt, action);
        let h_fact = self.calculate_factual_divergence(prompt, action);
        self.config.w_logic * h_logic + self.config.w_fact * h_fact
    }

    /// Read-only access to config.
    pub fn config(&self) -> &BackfireConfig {
        &self.config
    }

    /// Current history length.
    pub fn history_len(&self) -> usize {
        self.history.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::InMemoryKnowledge;
    use crate::nli::HeuristicNli;

    fn make_scorer() -> CoherenceScorer {
        CoherenceScorer::new(
            BackfireConfig::default(),
            Arc::new(HeuristicNli),
            Arc::new(InMemoryKnowledge::new()),
        )
    }

    #[test]
    fn test_review_approved() {
        let scorer = make_scorer();
        let (approved, score) = scorer.review(
            "What color is the sky?",
            "The sky is blue, consistent with reality",
        );
        assert!(approved);
        assert!(score.score >= 0.6);
        assert!(score.h_logical < 0.5);
    }

    #[test]
    fn test_review_rejected_contradiction() {
        let scorer = make_scorer();
        let (approved, score) = scorer.review(
            "What color is the sky?",
            "The sky is green and the opposite is true",
        );
        assert!(!approved);
        assert!(score.score < 0.6);
    }

    #[test]
    fn test_factual_divergence_hallucination() {
        let scorer = make_scorer();
        let h = scorer.calculate_factual_divergence(
            "How many SCPN layers?",
            "There are many layers in the system",
        );
        assert!((h - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_factual_divergence_correct() {
        let scorer = make_scorer();
        let h = scorer.calculate_factual_divergence(
            "How many SCPN layers?",
            "There are 16 layers",
        );
        assert!(h < 0.5);
    }

    #[test]
    fn test_history_grows_and_caps() {
        let scorer = make_scorer();
        for i in 0..10 {
            scorer.review("test", &format!("response {i} consistent with reality"));
        }
        assert!(scorer.history_len() <= scorer.config().history_window);
    }

    #[test]
    fn test_compute_divergence() {
        let scorer = make_scorer();
        let div = scorer.compute_divergence("test", "some text");
        assert!((0.0..=1.0).contains(&div));
    }

    #[test]
    fn test_nan_nli_fallback() {
        use crate::nli::ExternalNli;
        let scorer = CoherenceScorer::new(
            BackfireConfig::default(),
            Arc::new(ExternalNli::new(|_, _| f64::NAN)),
            Arc::new(InMemoryKnowledge::new()),
        );
        let h = scorer.calculate_logical_divergence("a", "b");
        assert_eq!(h, 0.5);
    }

    #[test]
    fn test_coherence_formula() {
        // coherence = 1 - (0.6 * h_logic + 0.4 * h_fact)
        // With h_logic=0.1, h_fact=0.1:
        //   coherence = 1 - (0.06 + 0.04) = 0.9
        use crate::nli::ExternalNli;
        let scorer = CoherenceScorer::new(
            BackfireConfig::default(),
            Arc::new(ExternalNli::new(|_, _| 0.1)),
            Arc::new(InMemoryKnowledge::with_facts(Default::default())),
        );
        let (_, score) = scorer.review("test", "test");
        // h_fact = 0.5 (no facts → neutral), h_logic = 0.1
        // coherence = 1 - (0.6*0.1 + 0.4*0.5) = 1 - 0.26 = 0.74
        assert!((score.score - 0.74).abs() < 1e-9);
    }
}
