// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::chunking
//! Sentence segmentation and token-budget chunk building for NLI scoring.
//!
//! Mirrors `NLIScorer._split_sentences()` / `NLIScorer._build_chunks()`.

static CHUNK_ABBREVIATIONS: &[&str] = &[
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "inc.", "ltd.", "corp.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.",
];

fn is_abbreviation_token(token: &str) -> bool {
    let trimmed = token
        .trim_matches(|c: char| "()[]{}\"'`".contains(c))
        .to_lowercase();
    CHUNK_ABBREVIATIONS.contains(&trimmed.as_str())
}

/// Split text into sentence-like units for chunked NLI scoring.
///
/// Mirrors `NLIScorer._split_sentences()` behaviour with abbreviation-aware
/// splitting while keeping whitespace-normalised sentence outputs.
pub fn split_sentences(text: &str) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();

    for token in trimmed.split_whitespace() {
        current.push(token.to_string());

        let boundary = if token.ends_with('?') || token.ends_with('!') {
            true
        } else if token.ends_with('.') {
            !is_abbreviation_token(token)
        } else {
            false
        };

        if boundary {
            out.push(current.join(" "));
            current.clear();
        }
    }

    if !current.is_empty() {
        out.push(current.join(" "));
    }

    out.into_iter().filter(|s| !s.trim().is_empty()).collect()
}

fn estimate_tokens(text: &str) -> usize {
    text.len() / 4 + 1
}

/// Build sentence chunks under a token budget with optional overlap ratio.
///
/// Mirrors `NLIScorer._build_chunks()` and overlap behaviour in
/// `NLIScorer._build_chunks_overlap()`.
pub fn build_chunks(sentences: &[String], budget: usize, overlap_ratio: f64) -> Vec<String> {
    if overlap_ratio > 0.0 {
        let mut chunks: Vec<String> = Vec::new();
        let mut i = 0usize;
        while i < sentences.len() {
            let mut current: Vec<String> = Vec::new();
            let mut current_tokens = 0usize;
            let mut j = i;

            while j < sentences.len() {
                let st = estimate_tokens(&sentences[j]);
                if !current.is_empty() && current_tokens + st > budget {
                    break;
                }
                current.push(sentences[j].clone());
                current_tokens += st;
                j += 1;
            }

            if current.is_empty() {
                current.push(sentences[i].clone());
            }

            chunks.push(current.join(" "));
            let stride = ((current.len() as f64) * (1.0 - overlap_ratio)).floor() as usize;
            i += stride.max(1);
        }

        if chunks.is_empty() {
            return vec![sentences.join(" ")];
        }
        return chunks;
    }

    let mut chunks: Vec<String> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut current_tokens = 0usize;

    for sent in sentences {
        let sent_tokens = estimate_tokens(sent);
        if !current.is_empty() && current_tokens + sent_tokens > budget {
            chunks.push(current.join(" "));
            let prev_last = current
                .last()
                .cloned()
                .expect("non-empty current must have a last sentence");
            current = vec![prev_last];
            current_tokens = estimate_tokens(&current[0]);
        }
        current.push(sent.clone());
        current_tokens += sent_tokens;
    }

    if !current.is_empty() {
        chunks.push(current.join(" "));
    }

    if chunks.is_empty() {
        vec![sentences.join(" ")]
    } else {
        chunks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_basic() {
        let s = split_sentences("Hello world. How are you? Fine!");
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn test_split_sentences_empty() {
        let s = split_sentences("");
        assert!(s.is_empty());
    }

    #[test]
    fn test_split_sentences_abbreviation_kept() {
        let s = split_sentences("Dr. Smith arrived. We started.");
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], "Dr. Smith arrived.");
    }

    #[test]
    fn test_build_chunks_non_overlap() {
        let sentences = vec![
            "Sentence one is long enough.".to_string(),
            "Sentence two is long enough.".to_string(),
            "Sentence three is long enough.".to_string(),
        ];
        let chunks = build_chunks(&sentences, 10, 0.0);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_build_chunks_overlap() {
        let sentences = vec![
            "One.".to_string(),
            "Two.".to_string(),
            "Three.".to_string(),
            "Four.".to_string(),
        ];
        let chunks = build_chunks(&sentences, 5, 0.5);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_build_chunks_empty() {
        let chunks = build_chunks(&[], 100, 0.0);
        assert_eq!(chunks, vec!["".to_string()]);
    }
}
