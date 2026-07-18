// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::heuristics
//! No-NLI heuristic divergence scorers: the lite scorer and the
//! logical/factual fallbacks, with the shared negation polarity-flip
//! contradiction floor (KIMI3-negation).
//!
//! Mirrors `LiteScorer.score()` from `lite_scorer.py` and
//! `CoherenceScorer._heuristic_logical()` / `_heuristic_factual()` from
//! `scorer.py`.

use std::collections::HashSet;

use once_cell::sync::Lazy;
use regex::Regex;

static LITE_WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\w+\b").unwrap());

static LITE_ENTITY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b").unwrap());

static LITE_NEGATION_WORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "not",
        "no",
        "never",
        "neither",
        "nobody",
        "nothing",
        "nowhere",
        "nor",
        "cannot",
        "can't",
        "don't",
        "doesn't",
        "didn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "hasn't",
        "haven't",
        "hadn't",
    ]
    .into_iter()
    .collect()
});

static LITE_STOP_WORDS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "can't",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "he'd",
        "he'll",
        "he's",
        "her",
        "here",
        "here's",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "how's",
        "i",
        "i'd",
        "i'll",
        "i'm",
        "i've",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "let's",
        "me",
        "more",
        "most",
        "mustn't",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "she'd",
        "she'll",
        "she's",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "that's",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "there's",
        "these",
        "they",
        "they'd",
        "they'll",
        "they're",
        "they've",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "we'd",
        "we'll",
        "we're",
        "we've",
        "were",
        "weren't",
        "what",
        "what's",
        "when",
        "when's",
        "where",
        "where's",
        "which",
        "while",
        "who",
        "who's",
        "whom",
        "why",
        "why's",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
    .into_iter()
    .collect()
});

/// Lightweight divergence scorer using word overlap, length ratio,
/// named entity heuristics, and negation asymmetry.
///
/// Returns divergence in [0, 1]. 0 = aligned, 1 = contradicted.
/// A negation polarity flip on near-identical content floors the
/// divergence at the contradiction level (KIMI3-negation).
/// Mirrors `LiteScorer.score()` from `lite_scorer.py`.
pub fn lite_score(premise: &str, hypothesis: &str) -> f64 {
    if premise.is_empty() || hypothesis.is_empty() {
        return 0.5;
    }

    let p_words: HashSet<String> = LITE_WORD_RE
        .find_iter(&premise.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();
    let h_words: HashSet<String> = LITE_WORD_RE
        .find_iter(&hypothesis.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();

    if p_words.is_empty() || h_words.is_empty() {
        return 0.5;
    }

    // Jaccard overlap
    let intersection = p_words.intersection(&h_words).count();
    let union = p_words.union(&h_words).count();
    let jaccard = intersection as f64 / union as f64;

    // Length ratio penalty
    let len_ratio =
        premise.len().min(hypothesis.len()) as f64 / premise.len().max(hypothesis.len()) as f64;

    // Named entity overlap
    let p_ents: HashSet<String> = LITE_ENTITY_RE
        .find_iter(premise)
        .map(|m| m.as_str().to_string())
        .collect();
    let h_ents: HashSet<String> = LITE_ENTITY_RE
        .find_iter(hypothesis)
        .map(|m| m.as_str().to_string())
        .collect();
    let ent_overlap = if !p_ents.is_empty() && !h_ents.is_empty() {
        let ei = p_ents.intersection(&h_ents).count();
        let eu = p_ents.union(&h_ents).count();
        ei as f64 / eu as f64
    } else if !p_ents.is_empty() || !h_ents.is_empty() {
        0.0
    } else {
        0.5
    };

    // Negation asymmetry
    let p_neg = p_words
        .iter()
        .filter(|w| LITE_NEGATION_WORDS.contains(w.as_str()))
        .count();
    let h_neg = h_words
        .iter()
        .filter(|w| LITE_NEGATION_WORDS.contains(w.as_str()))
        .count();
    let neg_mismatch = (p_neg == 0) != (h_neg == 0);
    let neg_penalty = if neg_mismatch { 0.3 } else { 0.0 };

    let similarity =
        0.4 * jaccard + 0.2 * len_ratio + 0.2 * ent_overlap + 0.2 * (1.0 - neg_penalty);
    let divergence = (1.0 - similarity).clamp(0.0, 1.0);

    // A polarity flip on near-identical content is a direct
    // contradiction: the weighted penalty moves this composite by at
    // most 0.06, so floor at the contradiction level.
    if neg_mismatch {
        let p_content: HashSet<&str> = p_words
            .iter()
            .filter(|w| !LITE_STOP_WORDS.contains(w.as_str()))
            .map(|w| w.as_str())
            .collect();
        let h_content: HashSet<&str> = h_words
            .iter()
            .filter(|w| !LITE_STOP_WORDS.contains(w.as_str()))
            .map(|w| w.as_str())
            .collect();
        if !p_content.is_empty() && !h_content.is_empty() {
            let overlap = p_content.intersection(&h_content).count() as f64;
            // Precision of the hypothesis side, mirroring the factual
            // heuristic: the flip must target grounded content.
            let content_precision = overlap / h_content.len() as f64;
            if content_precision >= NEGATION_FLIP_OVERLAP {
                return divergence.max(0.9);
            }
        }
    }
    divergence
}

/// Batch lite scoring for multiple (premise, hypothesis) pairs.
///
/// Mirrors `LiteScorer.score_batch()` from `lite_scorer.py`.
pub fn lite_score_batch(pairs: &[(String, String)]) -> Vec<f64> {
    pairs.iter().map(|(p, h)| lite_score(p, h)).collect()
}

/// Logical divergence fallback used when model-backed NLI is unavailable.
///
/// Mirrors `CoherenceScorer._heuristic_logical()` from `scorer.py`.
pub fn heuristic_logical_divergence(text_output: &str, prompt: &str) -> f64 {
    let out = text_output.to_lowercase();
    if out.contains("consistent with reality") {
        return 0.1;
    }
    if out.contains("opposite is true") {
        return 0.9;
    }
    if out.contains("depends on your perspective") {
        return 0.5;
    }
    if prompt.is_empty() {
        return 0.5;
    }

    let p_words: HashSet<String> = LITE_WORD_RE
        .find_iter(&prompt.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();
    let o_words: HashSet<String> = LITE_WORD_RE
        .find_iter(&out)
        .map(|m| m.as_str().to_string())
        .collect();

    if p_words.is_empty() || o_words.is_empty() {
        return 0.5;
    }

    let intersection = p_words.intersection(&o_words).count();
    let union = p_words.union(&o_words).count();
    if union == 0 {
        return 0.5;
    }
    (1.0 - intersection as f64 / union as f64).clamp(0.0, 1.0)
}

/// Content-precision gate above which a negation polarity flip is
/// treated as a direct contradiction rather than a mild divergence:
/// the fraction of the output's content words grounded in the premise.
///
/// Mirrors `NEGATION_FLIP_OVERLAP` from `_heuristics.py`.
pub const NEGATION_FLIP_OVERLAP: f64 = 0.8;

/// Factual divergence fallback used when model-backed NLI is unavailable.
///
/// Word-overlap scoring with negation and entity checks; a negation
/// polarity flip on near-identical content floors the divergence at the
/// contradiction level (KIMI3-negation).
///
/// Mirrors `CoherenceScorer._heuristic_factual()` from `scorer.py`.
pub fn heuristic_factual_divergence(context: &str, text_output: &str) -> f64 {
    let ctx_raw: HashSet<String> = LITE_WORD_RE
        .find_iter(&context.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();
    let out_raw: HashSet<String> = LITE_WORD_RE
        .find_iter(&text_output.to_lowercase())
        .map(|m| m.as_str().to_string())
        .collect();

    let ctx_words: HashSet<String> = ctx_raw
        .iter()
        .filter(|w| !LITE_STOP_WORDS.contains(w.as_str()))
        .cloned()
        .collect();
    let out_words: HashSet<String> = out_raw
        .iter()
        .filter(|w| !LITE_STOP_WORDS.contains(w.as_str()))
        .cloned()
        .collect();

    if ctx_words.is_empty() || out_words.is_empty() {
        return 0.5;
    }

    let overlap = ctx_words.intersection(&out_words).count() as f64;
    let recall = overlap / ctx_words.len() as f64;
    let precision = overlap / out_words.len() as f64;
    let similarity = recall.max(precision);
    let mut divergence = 1.0 - similarity;

    let ctx_neg = ctx_raw
        .iter()
        .any(|w| LITE_NEGATION_WORDS.contains(w.as_str()));
    let out_neg = out_raw
        .iter()
        .any(|w| LITE_NEGATION_WORDS.contains(w.as_str()));
    if ctx_neg != out_neg {
        divergence += 0.25;
        // A polarity flip on grounded content is a direct contradiction:
        // when nearly all of the output's content words come from the
        // context, the negation necessarily applies to that shared
        // content, and the flat asymmetry penalty cannot push such
        // restatements past the rejection threshold. Gate on precision,
        // not recall — an output that covers the context but adds its
        // own negated material may be negating the added material.
        if precision >= NEGATION_FLIP_OVERLAP {
            divergence = divergence.max(0.9);
        }
    }

    let ctx_ents: HashSet<String> = LITE_ENTITY_RE
        .find_iter(context)
        .map(|m| m.as_str().to_string())
        .collect();
    let out_ents: HashSet<String> = LITE_ENTITY_RE
        .find_iter(text_output)
        .map(|m| m.as_str().to_string())
        .collect();
    let novel_ents_count = out_ents.difference(&ctx_ents).count();
    if novel_ents_count > 0 {
        divergence += 0.15;
    }

    divergence.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lite_score_identical() {
        let s = lite_score("The sky is blue today.", "The sky is blue today.");
        assert!(s < 0.15, "identical texts should have low divergence: {s}");
    }

    #[test]
    fn test_lite_score_contradicted() {
        let s = lite_score(
            "The company never ships products late.",
            "The company always ships products extremely late.",
        );
        // Negation asymmetry should raise divergence above identical-text baseline
        assert!(s > 0.2, "contradicted should have higher divergence: {s}");
    }

    #[test]
    fn test_lite_score_empty() {
        assert!((lite_score("", "something") - 0.5).abs() < 1e-9);
        assert!((lite_score("hello", "") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_lite_score_negation_flip_high_overlap_contradicts() {
        // KIMI3-negation: the weighted penalty alone left this at 0.14.
        let s = lite_score(
            "Paris is the capital of France.",
            "Paris is not the capital of France.",
        );
        assert!(s >= 0.9, "expected contradiction floor, got {s}");
    }

    #[test]
    fn test_lite_score_negated_true_claim_below_gate_not_floored() {
        let s = lite_score(
            "The maximum single adult dose of ibuprofen is 400 mg.",
            "Adults should not exceed 400 mg of ibuprofen in a single dose.",
        );
        assert!(s < 0.9, "content below the gate must not be floored: {s}");
    }

    #[test]
    fn test_lite_score_matching_negation_polarity_no_floor() {
        let s = lite_score(
            "Phone support is not available on the free plan.",
            "Phone support is not available on the free plan.",
        );
        assert!(s < 0.15, "identical negated texts must align, got {s}");
    }

    #[test]
    fn test_lite_score_entity_mismatch() {
        let s = lite_score(
            "Apple released a new product.",
            "Samsung released a new product.",
        );
        // Same structure, different entity → entity overlap < 1
        assert!(s > 0.1, "entity mismatch should increase divergence: {s}");
    }

    #[test]
    fn test_lite_score_batch() {
        let pairs = vec![
            (
                "The sky is blue.".to_string(),
                "The sky is blue.".to_string(),
            ),
            (
                "Yes it works.".to_string(),
                "No it does not work.".to_string(),
            ),
        ];
        let results = lite_score_batch(&pairs);
        assert_eq!(results.len(), 2);
        assert!(results[0] < results[1], "identical < contradicted");
    }

    #[test]
    fn test_heuristic_logical_aligned_keyword() {
        let s = heuristic_logical_divergence("This is consistent with reality.", "q");
        assert_eq!(s, 0.1);
    }

    #[test]
    fn test_heuristic_logical_contradicted_keyword() {
        let s = heuristic_logical_divergence("The opposite is true.", "q");
        assert_eq!(s, 0.9);
    }

    #[test]
    fn test_heuristic_logical_neutral_keyword() {
        let s = heuristic_logical_divergence("depends on your perspective", "q");
        assert_eq!(s, 0.5);
    }

    #[test]
    fn test_heuristic_logical_no_prompt() {
        let s = heuristic_logical_divergence("some text", "");
        assert_eq!(s, 0.5);
    }

    #[test]
    fn test_heuristic_factual_empty_inputs() {
        let s = heuristic_factual_divergence("", "something");
        assert_eq!(s, 0.5);
    }

    #[test]
    fn test_heuristic_factual_negation_asymmetry() {
        let s = heuristic_factual_divergence("The sky is blue.", "The sky is not blue.");
        assert!(s > 0.2);
    }

    #[test]
    fn test_heuristic_factual_negation_flip_high_overlap_contradicts() {
        // KIMI3-negation: a polarity flip on near-identical content must
        // floor at the contradiction level, not the flat +0.25 penalty.
        let s = heuristic_factual_divergence(
            "Paris is the capital of France.",
            "Paris is not the capital of France.",
        );
        assert!(s >= 0.9, "expected contradiction floor, got {s}");
    }

    #[test]
    fn test_heuristic_factual_negation_flip_gate_inclusive() {
        // Content precision is exactly 0.8 here: the gate is inclusive.
        let s = heuristic_factual_divergence(
            "World War II ended in 1945.",
            "World War II did not end in 1945.",
        );
        assert!(
            s >= 0.9,
            "expected contradiction floor at precision 0.8, got {s}"
        );
    }

    #[test]
    fn test_heuristic_factual_negated_true_claim_below_gate_not_floored() {
        // A TRUE claim phrased with negation against a positive fact sits
        // below the overlap gate: flat penalty only, no contradiction floor.
        let s = heuristic_factual_divergence(
            "The maximum single adult dose of ibuprofen is 400 mg.",
            "Adults should not exceed 400 mg of ibuprofen in a single dose.",
        );
        assert!(
            s > 0.5 && s < 0.9,
            "expected penalty without floor, got {s}"
        );
    }

    #[test]
    fn test_heuristic_factual_matching_negation_polarity_no_penalty() {
        let s = heuristic_factual_divergence(
            "Phone support is not available on the free plan.",
            "Phone support is not available on the free plan.",
        );
        assert!(
            s.abs() < 1e-9,
            "identical negated texts must align, got {s}"
        );
    }

    #[test]
    fn test_heuristic_factual_novel_entities() {
        let s = heuristic_factual_divergence("The sky is blue.", "Planet Mars is red.");
        assert!(s > 0.3);
    }
}
