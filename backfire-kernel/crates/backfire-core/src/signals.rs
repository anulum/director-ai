// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — signals
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Verification Signals (Rust)
// (C) 1998-2026 Miroslav Šotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Fast verification signal functions for the VerifiedScorer.
//!
//! Ports of the four per-claim signal extraction functions from
//! `src/director_ai/core/scoring/verified_scorer.py`. Called once
//! per claim × source match in heuristic mode.
//!
//! Python fallbacks exist; these are optional accelerators.

use std::collections::HashSet;

/// Words to exclude from content-based comparisons.
const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or", "if", "then", "than",
    "that", "this", "these", "those", "it", "its",
];

const NEG_WORDS: &[&str] = &[
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "cannot",
    "can't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "doesn't",
    "didn't",
    "hasn't",
    "haven't",
    "hadn't",
    "without",
    "none",
    "nobody",
];

fn to_lower_words(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.to_lowercase()).collect()
}

fn is_capitalized_word(w: &str) -> bool {
    let mut chars = w.chars();
    match chars.next() {
        Some(c) if c.is_uppercase() => chars.next().is_some_and(|c2| c2.is_lowercase()),
        _ => false,
    }
}

/// Extract proper-noun-like entities (capitalised words).
fn extract_entities(text: &str) -> HashSet<String> {
    let mut entities = HashSet::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if is_capitalized_word(word) {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        } else if !current.is_empty() {
            entities.insert(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        entities.insert(current);
    }
    entities
}

/// Extract numbers (digit sequences with optional commas/dots).
fn extract_numbers(text: &str) -> HashSet<String> {
    let mut nums = HashSet::new();
    let mut current = String::new();
    let mut in_num = false;

    for ch in text.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
            in_num = true;
        } else if in_num && (ch == ',' || ch == '.') {
            current.push(ch);
        } else {
            if in_num && !current.is_empty() {
                // Trim trailing punctuation
                let trimmed = current.trim_end_matches([',', '.']);
                if !trimmed.is_empty() {
                    nums.insert(trimmed.to_string());
                }
                current.clear();
            }
            in_num = false;
        }
    }
    if in_num && !current.is_empty() {
        let trimmed = current.trim_end_matches([',', '.']);
        if !trimmed.is_empty() {
            nums.insert(trimmed.to_string());
        }
    }
    nums
}

/// Jaccard overlap of proper-noun entities between two texts.
///
/// Returns 1.0 if neither text has entities.
pub fn entity_overlap(text_a: &str, text_b: &str) -> f64 {
    let ents_a = extract_entities(text_a);
    let ents_b = extract_entities(text_b);

    if ents_a.is_empty() && ents_b.is_empty() {
        return 1.0;
    }
    let union_len = ents_a.union(&ents_b).count();
    if union_len == 0 {
        return 1.0;
    }
    let intersect_len = ents_a.intersection(&ents_b).count();
    intersect_len as f64 / union_len as f64
}

/// Check whether numbers in `text_a` match numbers in `text_b`.
///
/// Returns `None` if neither text contains numbers.
pub fn numerical_consistency(text_a: &str, text_b: &str) -> Option<bool> {
    let nums_a = extract_numbers(text_a);
    let nums_b = extract_numbers(text_b);

    if nums_a.is_empty() && nums_b.is_empty() {
        return None;
    }
    if nums_a.is_empty() || nums_b.is_empty() {
        return None;
    }
    Some(!nums_a.is_disjoint(&nums_b))
}

/// Detect if the claim negates something the source states positively
/// (or vice versa).
pub fn negation_flip(claim: &str, source: &str) -> bool {
    let neg_set: HashSet<&str> = NEG_WORDS.iter().copied().collect();

    let claim_words: HashSet<String> = to_lower_words(claim).into_iter().collect();
    let source_words: HashSet<String> = to_lower_words(source).into_iter().collect();

    let claim_has_neg = claim_words.iter().any(|w| neg_set.contains(w.as_str()));
    let source_has_neg = source_words.iter().any(|w| neg_set.contains(w.as_str()));

    if claim_has_neg == source_has_neg {
        return false;
    }

    // Count shared content words (excluding negation)
    let content_a: HashSet<&str> = claim_words
        .iter()
        .filter(|w| !neg_set.contains(w.as_str()))
        .map(|w| w.as_str())
        .collect();
    let content_b: HashSet<&str> = source_words
        .iter()
        .filter(|w| !neg_set.contains(w.as_str()))
        .map(|w| w.as_str())
        .collect();

    content_a.intersection(&content_b).count() >= 3
}

/// Fraction of claim's content words found in source.
///
/// Low traceability = claim contains info not in source = potential fabrication.
pub fn traceability(claim: &str, source: &str) -> f64 {
    let stop: HashSet<&str> = STOP_WORDS.iter().copied().collect();
    let neg: HashSet<&str> = NEG_WORDS.iter().copied().collect();

    let claim_words: HashSet<String> = to_lower_words(claim)
        .into_iter()
        .filter(|w| !stop.contains(w.as_str()) && !neg.contains(w.as_str()))
        .collect();

    if claim_words.is_empty() {
        return 1.0;
    }

    let source_words: HashSet<String> = to_lower_words(source)
        .into_iter()
        .filter(|w| !stop.contains(w.as_str()) && !neg.contains(w.as_str()))
        .collect();

    let matched = claim_words
        .iter()
        .filter(|w| source_words.contains(*w))
        .count();

    matched as f64 / claim_words.len() as f64
}

/// Linear regression trend drop over a window of coherence scores.
///
/// Returns the projected drop magnitude: -slope * (n - 1).
/// Positive values indicate downward trend.
pub fn trend_drop(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let x_mean = (nf - 1.0) / 2.0;
    let y_mean: f64 = values.iter().sum::<f64>() / nf;

    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &y) in values.iter().enumerate() {
        let xi = i as f64 - x_mean;
        num += xi * (y - y_mean);
        den += xi * xi;
    }

    let slope = if den.abs() > 1e-12 { num / den } else { 0.0 };
    -slope * (nf - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_overlap_identical() {
        let score = entity_overlap(
            "Paris is the capital of France",
            "The capital of France is Paris",
        );
        assert!(score > 0.5, "Expected high overlap, got {score}");
    }

    #[test]
    fn test_entity_overlap_no_entities() {
        assert_eq!(entity_overlap("hello world", "goodbye world"), 1.0);
    }

    #[test]
    fn test_numerical_consistency_match() {
        assert_eq!(
            numerical_consistency("There are 46 chromosomes", "Humans have 46 total"),
            Some(true),
        );
    }

    #[test]
    fn test_numerical_consistency_mismatch() {
        assert_eq!(
            numerical_consistency("Data retained for 90 days", "Data retained for 30 days"),
            Some(false),
        );
    }

    #[test]
    fn test_numerical_consistency_none() {
        assert_eq!(
            numerical_consistency("The sky is blue", "Blue is the sky"),
            None,
        );
    }

    #[test]
    fn test_negation_flip_detected() {
        assert!(negation_flip(
            "Phone support is not available for Team plan",
            "Phone support is available for all paid plans",
        ));
    }

    #[test]
    fn test_negation_flip_same_polarity() {
        assert!(!negation_flip(
            "Water boils at 100 degrees",
            "Water boils at 100 degrees Celsius",
        ));
    }

    #[test]
    fn test_traceability_high() {
        let score = traceability(
            "Water boils at 100 degrees Celsius",
            "Water boils at 100 degrees Celsius at standard pressure",
        );
        assert!(score > 0.8, "Expected high traceability, got {score}");
    }

    #[test]
    fn test_traceability_low() {
        let score = traceability(
            "HIPAA and FedRAMP certified",
            "SOC 2 Type II and ISO 27001 certified",
        );
        // "certified" is shared → 1/3 = 0.33; still low but not zero
        assert!(score < 0.5, "Expected low traceability, got {score}");
    }

    #[test]
    fn test_trend_drop_flat() {
        let values = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        assert!(trend_drop(&values).abs() < 1e-10);
    }

    #[test]
    fn test_trend_drop_declining() {
        let values = vec![0.9, 0.7, 0.5, 0.3, 0.1];
        assert!(trend_drop(&values) > 0.5);
    }

    #[test]
    fn test_trend_drop_single() {
        assert_eq!(trend_drop(&[0.5]), 0.0);
    }
}
