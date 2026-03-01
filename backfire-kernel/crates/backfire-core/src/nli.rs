// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — NLI Backend Interface
// Mirrors: src/director_ai/core/nli.py (NLIScorer)
// ─────────────────────────────────────────────────────────────────────
//! NLI (Natural Language Inference) backend trait and heuristic
//! fallback implementation.
//!
//! In production, the real NLI model (DeBERTa or ONNX) runs behind
//! this trait — either via ONNX Runtime embedded in Rust, or via
//! gRPC/HTTP to an external inference server. The heuristic fallback
//! provides deterministic scoring for testing.

/// Trait for NLI backends.
///
/// Returns logical divergence ∈ [0, 1]:
/// - 0.0 = entailment (fully consistent)
/// - 0.5 = neutral
/// - 1.0 = contradiction
pub trait NliBackend: Send + Sync {
    fn score(&self, premise: &str, hypothesis: &str) -> f64;
}

/// Deterministic heuristic NLI scorer (no model required).
///
/// Mirrors `NLIScorer._heuristic_score()` from `nli.py:133-149`.
/// Used for testing and as a fallback when no model is available.
pub struct HeuristicNli;

impl NliBackend for HeuristicNli {
    fn score(&self, _premise: &str, hypothesis: &str) -> f64 {
        let h_lower = hypothesis.to_lowercase();

        if h_lower.contains("consistent with reality") {
            return 0.1;
        }
        if h_lower.contains("opposite is true") {
            return 0.9;
        }
        if h_lower.contains("depends on your perspective") {
            return 0.5;
        }

        let p_lower = _premise.to_lowercase();
        let h_lower_owned = hypothesis.to_lowercase();
        let p_words: std::collections::HashSet<&str> = p_lower.split_whitespace().collect();
        let h_words: std::collections::HashSet<&str> = h_lower_owned.split_whitespace().collect();

        if p_words.is_empty() {
            return 0.5;
        }

        let overlap = p_words.intersection(&h_words).count() as f64;
        let max_len = p_words.len().max(1) as f64;
        (0.5 - (overlap / max_len) * 0.3).clamp(0.1, 0.9)
    }
}

/// External NLI backend that calls a scoring function pointer.
///
/// Used by the PyO3 FFI layer to delegate NLI scoring back to Python
/// (where the DeBERTa model lives) while keeping the rest of the
/// hot path in Rust.
type NliScoreFn = Box<dyn Fn(&str, &str) -> f64 + Send + Sync>;

pub struct ExternalNli {
    score_fn: NliScoreFn,
}

impl ExternalNli {
    pub fn new(score_fn: impl Fn(&str, &str) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            score_fn: Box::new(score_fn),
        }
    }
}

impl NliBackend for ExternalNli {
    fn score(&self, premise: &str, hypothesis: &str) -> f64 {
        (self.score_fn)(premise, hypothesis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_entailment() {
        let nli = HeuristicNli;
        assert!((nli.score("test", "consistent with reality") - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_heuristic_contradiction() {
        let nli = HeuristicNli;
        assert!((nli.score("test", "the opposite is true") - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_heuristic_neutral() {
        let nli = HeuristicNli;
        assert!((nli.score("test", "it depends on your perspective") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_heuristic_default_range() {
        let nli = HeuristicNli;
        let score = nli.score("the sky is blue", "something unrelated");
        assert!((0.1..=0.9).contains(&score));
    }

    #[test]
    fn test_external_nli() {
        let nli = ExternalNli::new(|_p, _h| 0.42);
        assert!((nli.score("a", "b") - 0.42).abs() < 1e-9);
    }
}
