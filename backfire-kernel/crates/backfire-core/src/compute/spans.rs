// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::spans
//! Flagged-token span merging for hallucination span detection.
//!
//! Mirrors `merge_flagged_spans` from `span_detector.py`.

/// True for the characters Python's `str.isspace()` treats as whitespace.
///
/// Rust's `char::is_whitespace` (the Unicode `White_Space` property) matches
/// Python everywhere except the four C0 information separators U+001C..U+001F,
/// which Python counts and `White_Space` does not. Adding them makes the two
/// definitions identical, so the span-merge whitespace bridge is bit-exact with
/// the Python floor's `response[a:b].strip() == ""`.
fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// True when `chars[from..to]` is empty or entirely Python-whitespace.
///
/// Mirrors `response[from:to].strip() == ""`: an empty slice (`to <= from`) is
/// blank (Python's `""`), otherwise every character must be whitespace. Indices
/// are clamped so an out-of-range bound degrades to an empty (blank) slice rather
/// than panicking, matching Python's forgiving slice semantics.
fn gap_is_blank(chars: &[char], from: i64, to: i64) -> bool {
    let len = chars.len();
    let a = from.max(0) as usize;
    let b = to.max(0) as usize;
    let (a, b) = (a.min(len), b.min(len));
    if b <= a {
        return true;
    }
    chars[a..b].iter().all(|&c| is_python_space(c))
}

/// Merge contiguous flagged response tokens into character spans.
///
/// Mirrors `merge_flagged_spans` in `span_detector.py` exactly. `offsets` and
/// `scores` are aligned per response token (character offsets into the response
/// and `P(hallucinated)`); `response_chars` is the response as a `char` vector so
/// indexing is by code point, matching Python string slicing. A token is flagged
/// when its score reaches `threshold` and its offset range is non-empty
/// (`ce > cs`); adjacent flagged tokens whose intervening gap is blank collapse
/// into one span so a hallucinated phrase is not split on its spaces.
///
/// Returns `(spans, flagged, max_score)` where each span is
/// `(start, end, max_token_score)`; the caller slices the response for the text
/// so extraction stays identical to the Python floor.
pub fn merge_flagged_spans(
    offsets: &[(i64, i64)],
    scores: &[f64],
    response_chars: &[char],
    threshold: f64,
) -> (Vec<(i64, i64, f64)>, usize, f64) {
    let mut spans: Vec<(i64, i64, f64)> = Vec::new();
    let mut flagged = 0usize;
    let mut max_score = 0.0f64;
    let mut cur_start: i64 = -1;
    let mut cur_end: i64 = -1;
    let mut cur_max = 0.0f64;

    for (&(cs, ce), &score) in offsets.iter().zip(scores.iter()) {
        if score > max_score {
            max_score = score;
        }
        if score < threshold || ce <= cs {
            continue;
        }
        flagged += 1;
        if cur_end >= 0 && gap_is_blank(response_chars, cur_end, cs) {
            cur_end = ce;
            if score > cur_max {
                cur_max = score;
            }
        } else {
            if cur_end >= 0 {
                spans.push((cur_start, cur_end, cur_max));
            }
            cur_start = cs;
            cur_end = ce;
            cur_max = score;
        }
    }
    if cur_end >= 0 {
        spans.push((cur_start, cur_end, cur_max));
    }
    (spans, flagged, max_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn merge_bridges_blank_gap_and_breaks_on_text() {
        // "ab.cd ef": flag ab, cd, ef. The ab↔cd gap "." is non-blank → break;
        // the cd↔ef gap is a single space → bridge into one span.
        let response = chars("ab.cd ef");
        let offsets = [(0, 2), (2, 3), (3, 5), (5, 6), (6, 8)];
        let scores = [0.99, 0.10, 0.97, 0.10, 0.96];
        let (spans, flagged, max_score) = merge_flagged_spans(&offsets, &scores, &response, 0.95);
        assert_eq!(flagged, 3);
        assert!((max_score - 0.99).abs() < 1e-12);
        assert_eq!(spans, vec![(0, 2, 0.99), (3, 8, 0.97)]);
    }

    #[test]
    fn merge_skips_empty_offset_and_below_threshold() {
        let response = chars("x");
        // special-token (0,0) ce<=cs skipped; 0.5 below threshold skipped.
        let offsets = [(0, 0), (0, 1)];
        let scores = [0.99, 0.50];
        let (spans, flagged, max_score) = merge_flagged_spans(&offsets, &scores, &response, 0.95);
        assert_eq!(flagged, 0);
        assert!(spans.is_empty());
        assert!((max_score - 0.99).abs() < 1e-12);
    }

    #[test]
    fn gap_blank_matches_python_isspace_separators() {
        // U+001C..U+001F are Python whitespace but not Unicode White_Space.
        let response = chars("a\u{1c}\u{1f}b");
        assert!(gap_is_blank(&response, 1, 3));
        assert!(gap_is_blank(&response, 2, 2)); // empty slice is blank
        assert!(!gap_is_blank(&response, 0, 1)); // 'a' is not whitespace
    }
}
