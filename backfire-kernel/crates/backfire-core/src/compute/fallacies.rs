// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::fallacies
//! Informal-fallacy marker detection.
//!
//! Mirrors `detect_fallacies()` from `core/verification/fallacy_detector.py`.

use once_cell::sync::Lazy;
use regex::Regex;

/// Informal-fallacy marker patterns, mirroring ``_FALLACY_SPECS`` in
/// `core/verification/fallacy_detector.py`. Lookaround- and backreference-free so
/// the Rust and Python regex engines produce identical matches.
static FALLACY_PATTERNS: Lazy<Vec<(&'static str, Regex)>> = Lazy::new(|| {
    vec![
        (
            "ad_hominem",
            Regex::new(
                r"(?i)\b(?:you|he|she|they)(?:'re| are| is|'s)?\s+(?:just\s+)?(?:an?\s+)?(?:too\s+)?(?:idiot|idiots|stupid|fool|fools|ignorant|incompetent|biased|liar|liars|clueless|moron|morons|dishonest)\b",
            )
            .unwrap(),
        ),
        (
            "appeal_to_authority",
            Regex::new(
                r"(?i)\b(?:because|since)\s+(?:an?\s+|the\s+)?(?:expert|experts|authority|authorities|professor|professors|doctor|doctors|scientist|scientists)\s+(?:say|says|said|claim|claims|agree|agrees|believe|believes)\b",
            )
            .unwrap(),
        ),
        (
            "bandwagon",
            Regex::new(
                r"(?i)\b(?:everyone|everybody|nobody|no one)\s+(?:knows|agrees|believes|thinks|does it)\b",
            )
            .unwrap(),
        ),
        (
            "false_dichotomy",
            Regex::new(
                r"(?i)\b(?:only\s+two\s+(?:options|choices|possibilities)|either\s+with\s+(?:us|me)\s+or\s+against|either\s+\w+\s+or\s+nothing)\b",
            )
            .unwrap(),
        ),
        (
            "hasty_generalization",
            Regex::new(
                r"(?i)\b(?:proves|shows|means)\s+(?:that\s+)?(?:all|every|everyone|no one|nobody|always|never)\b",
            )
            .unwrap(),
        ),
        (
            "slippery_slope",
            Regex::new(
                r"(?i)\b(?:will|would|could)\s+(?:inevitably|eventually|ultimately)\s+lead\s+to\b|\bnext\s+thing\s+you\s+know\b",
            )
            .unwrap(),
        ),
        (
            "appeal_to_emotion",
            Regex::new(
                r"(?i)\bthink\s+of\s+the\s+children\b|\byou\s+should\s+be\s+ashamed\b|\bhow\s+would\s+you\s+feel\b",
            )
            .unwrap(),
        ),
        (
            "post_hoc",
            Regex::new(
                r"(?i)\bcorrelat\w+[^.?!]{0,30}?\bcaus\w+|\bafter\s+\w+[^.?!]{0,40}?\btherefore\b",
            )
            .unwrap(),
        ),
    ]
});

/// Scan *text* for informal-fallacy markers.
///
/// Returns ``(fallacy_type, matched_span)`` pairs in scan order (pattern order,
/// then left-to-right). Mirrors `detect_fallacies()` from
/// `core/verification/fallacy_detector.py`.
pub fn detect_fallacies(text: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (name, re) in FALLACY_PATTERNS.iter() {
        for m in re.find_iter(text) {
            out.push((name.to_string(), m.as_str().to_string()));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_fallacies_families() {
        assert_eq!(detect_fallacies("You're just biased.")[0].0, "ad_hominem");
        assert_eq!(detect_fallacies("Everyone knows this.")[0].0, "bandwagon");
        assert_eq!(
            detect_fallacies("This proves that all of them lie.")[0].0,
            "hasty_generalization"
        );
        assert!(detect_fallacies("The capital of France is Paris.").is_empty());
    }
}
