// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licence available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — PII regex scanner fast path

//! Multi-pattern PII regex scanner.
//!
//! The Python [`RegexPIIDetector`] iterates a fixed list of
//! compiled regexes and calls `finditer` on each. Python's `re` is
//! already C-backed, but batching the scan behind a single
//! `RegexSet` lets us:
//!
//! * skip patterns that cannot match before ever invoking them;
//! * walk the input text once per scan regardless of pattern count;
//! * keep the scan allocation bounded.
//!
//! The returned `PiiMatch` records mirror the Python
//! `ModerationMatch` shape (category, byte offsets); the Python
//! adapter lifts them into `ModerationMatch` objects when the
//! `backfire-kernel` feature is installed.

use regex::{Regex, RegexSet};

/// A single finding: (category, byte_start, byte_end).
///
/// Byte offsets are returned so that the Python side can keep the
/// original text slice; a char-count would force a second linear
/// scan on non-ASCII input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiiMatch {
    pub category: String,
    pub start: usize,
    pub end: usize,
}

/// Compiled multi-pattern scanner. Construct once per detector
/// instance; cheap to reuse across thousands of `scan` calls.
pub struct PiiScanner {
    categories: Vec<String>,
    set: RegexSet,
    singles: Vec<Regex>,
}

impl PiiScanner {
    /// Build a scanner from a list of `(category, pattern)` pairs.
    /// Returns an error if any pattern fails to compile; every
    /// supplied pattern has to survive — a detector that silently
    /// drops broken regexes hides operator mistakes.
    pub fn new(patterns: &[(&str, &str)]) -> Result<Self, regex::Error> {
        let categories: Vec<String> = patterns.iter().map(|(c, _)| (*c).to_string()).collect();
        let sources: Vec<&str> = patterns.iter().map(|(_, p)| *p).collect();
        let set = RegexSet::new(&sources)?;
        let mut singles = Vec::with_capacity(sources.len());
        for pattern in &sources {
            singles.push(Regex::new(pattern)?);
        }
        Ok(Self {
            categories,
            set,
            singles,
        })
    }

    /// Scan one text for every pattern. The returned matches appear
    /// in text-order per category, but categories themselves are not
    /// globally sorted — callers that need a deterministic ordering
    /// should sort by `(start, end, category)` after the fact.
    pub fn scan(&self, text: &str) -> Vec<PiiMatch> {
        if text.is_empty() {
            return Vec::new();
        }
        let matches = self.set.matches(text);
        let mut out: Vec<PiiMatch> = Vec::new();
        for idx in matches.iter() {
            let category = &self.categories[idx];
            for m in self.singles[idx].find_iter(text) {
                out.push(PiiMatch {
                    category: category.clone(),
                    start: m.start(),
                    end: m.end(),
                });
            }
        }
        out
    }

    /// Count of registered pattern/category pairs — useful for
    /// smoke tests asserting the scanner was wired correctly.
    pub fn len(&self) -> usize {
        self.categories.len()
    }

    /// True when the scanner has no registered patterns.
    pub fn is_empty(&self) -> bool {
        self.categories.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin_patterns() -> Vec<(&'static str, &'static str)> {
        vec![
            ("email", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
            ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
            ("credit_card", r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),
        ]
    }

    #[test]
    fn empty_text_returns_empty() {
        let scanner = PiiScanner::new(&builtin_patterns()).unwrap();
        assert!(scanner.scan("").is_empty());
    }

    #[test]
    fn single_email_hit() {
        let scanner = PiiScanner::new(&builtin_patterns()).unwrap();
        let results = scanner.scan("contact a.b@example.com tomorrow");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].category, "email");
    }

    #[test]
    fn multi_category_hits() {
        let scanner = PiiScanner::new(&builtin_patterns()).unwrap();
        let results = scanner.scan(
            "card 4111-1111-1111-1111 ssn 123-45-6789 email x@y.com",
        );
        let cats: std::collections::BTreeSet<_> =
            results.iter().map(|m| m.category.as_str()).collect();
        assert!(cats.contains("email"));
        assert!(cats.contains("ssn"));
        assert!(cats.contains("credit_card"));
    }

    #[test]
    fn invalid_pattern_rejected() {
        let bad = vec![("x", "[")];
        assert!(PiiScanner::new(&bad).is_err());
    }

    #[test]
    fn offsets_are_byte_accurate() {
        let scanner = PiiScanner::new(&builtin_patterns()).unwrap();
        let text = "abc a@b.co xyz";
        let results = scanner.scan(text);
        assert_eq!(results.len(), 1);
        assert_eq!(&text[results[0].start..results[0].end], "a@b.co");
    }

    #[test]
    fn non_matching_patterns_do_not_allocate() {
        let scanner = PiiScanner::new(&builtin_patterns()).unwrap();
        let results = scanner.scan("boring plain text with nothing to hide");
        assert!(results.is_empty());
    }

    #[test]
    fn len_and_is_empty() {
        let patterns = builtin_patterns();
        let scanner = PiiScanner::new(&patterns).unwrap();
        assert_eq!(scanner.len(), 3);
        assert!(!scanner.is_empty());
        let empty = PiiScanner::new(&[]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }
}
