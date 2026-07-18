// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::reasoning
//! Reasoning-chain step extraction and word-overlap similarity.
//!
//! Mirrors `extract_steps()` / `_word_overlap()` from `reasoning_verifier.py`.

use std::collections::HashSet;

use once_cell::sync::Lazy;
use regex::Regex;

// Split on numbered step boundaries (works inline and multiline)
static NUMBERED_SPLIT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:^|\n)\s*(?:Step\s+)?\d+[.):]").unwrap());

static BULLET_STEP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\s*[-*•]\s+(.+)$").unwrap());

static NL_STEP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(?:^|\n)(?:First|Second|Third|Next|Then|Finally|Therefore|Thus|Hence|So)[,]?\s+",
    )
    .unwrap()
});

/// Extract reasoning steps from text.
///
/// Tries numbered steps, bullets, then natural language markers.
/// Mirrors `extract_steps()` from `reasoning_verifier.py`.
pub fn extract_reasoning_steps(text: &str) -> Vec<String> {
    // Try numbered steps — split text at step boundaries, keep content
    let splits: Vec<&str> = NUMBERED_SPLIT_RE.split(text).collect();
    // First element is text before any step marker (usually empty)
    let numbered: Vec<String> = splits
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if numbered.len() >= 2 {
        return numbered;
    }

    // Try bullet points
    let bullets: Vec<String> = BULLET_STEP_RE
        .captures_iter(text)
        .map(|c| c[1].trim().to_string())
        .collect();
    if bullets.len() >= 2 {
        return bullets;
    }

    // Try natural language markers — split on marker, keep content after
    let nl_splits: Vec<&str> = NL_STEP_RE.split(text).collect();
    let nl: Vec<String> = nl_splits
        .iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if nl.len() >= 2 {
        return nl;
    }

    // Sentence fallback
    let sentences: Vec<String> = text
        .split(['.', '!', '?'])
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() > 10)
        .collect();
    if sentences.len() >= 2 {
        return sentences;
    }

    vec![]
}

/// Jaccard word overlap between two texts.
///
/// Mirrors `_word_overlap()` from `reasoning_verifier.py`.
pub fn word_overlap(text_a: &str, text_b: &str) -> f64 {
    let words_a: HashSet<String> = text_a
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();
    let words_b: HashSet<String> = text_b
        .split_whitespace()
        .map(|w| w.to_lowercase())
        .collect();

    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_numbered() {
        let text = "1. First step\n2. Second step\n3. Third step";
        let steps = extract_reasoning_steps(text);
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0], "First step");
    }

    #[test]
    fn test_extract_bullets() {
        let text = "- Step A\n- Step B\n- Step C";
        let steps = extract_reasoning_steps(text);
        assert_eq!(steps.len(), 3);
    }

    #[test]
    fn test_word_overlap_identical() {
        let score = word_overlap("hello world", "hello world");
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_word_overlap_disjoint() {
        let score = word_overlap("hello world", "foo bar");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_word_overlap_partial() {
        let score = word_overlap("hello world foo", "hello bar baz");
        // intersection: {hello}, union: {hello, world, foo, bar, baz} = 1/5
        assert!((score - 0.2).abs() < 1e-9);
    }
}
