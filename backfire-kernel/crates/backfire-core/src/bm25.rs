// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — bm25
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel BM25 Retrieval Engine (Rust)
// (C) 1998-2026 Miroslav Šotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! BM25 sparse retrieval engine for the HybridBackend.
//!
//! Replaces the Python `HybridBackend._bm25_query()` inner loop.
//! Thread-safe via `parking_lot::RwLock`.

use std::collections::HashMap;

use parking_lot::RwLock;

/// A single indexed document.
struct Doc {
    id: String,
    term_freqs: HashMap<String, u32>,
    doc_len: u32,
}

/// BM25 retrieval engine with Okapi BM25 scoring (k1=1.2, b=0.75).
pub struct BM25Engine {
    docs: RwLock<Vec<Doc>>,
    doc_freq: RwLock<HashMap<String, u32>>,
    total_len: RwLock<u64>,
    k1: f64,
    b: f64,
}

/// A single BM25 result: document ID + relevance score.
#[derive(Debug, Clone)]
pub struct BM25Result {
    pub doc_id: String,
    pub score: f64,
}

impl Default for BM25Engine {
    fn default() -> Self {
        Self::new(1.2, 0.75)
    }
}

impl BM25Engine {
    pub fn new(k1: f64, b: f64) -> Self {
        Self {
            docs: RwLock::new(Vec::new()),
            doc_freq: RwLock::new(HashMap::new()),
            total_len: RwLock::new(0),
            k1,
            b,
        }
    }

    fn tokenize(text: &str) -> Vec<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_lowercase())
            .collect()
    }

    /// Add a document to the index.
    pub fn add_document(&self, doc_id: &str, text: &str) {
        let tokens = Self::tokenize(text);
        let doc_len = tokens.len() as u32;

        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_insert(0) += 1;
        }

        let unique_terms: Vec<String> = tf.keys().cloned().collect();

        let mut docs = self.docs.write();
        let mut df = self.doc_freq.write();
        let mut total = self.total_len.write();

        docs.push(Doc {
            id: doc_id.to_string(),
            term_freqs: tf,
            doc_len,
        });

        for term in unique_terms {
            *df.entry(term).or_insert(0) += 1;
        }

        *total += doc_len as u64;
    }

    /// Query the index, returning top-K results sorted by BM25 score.
    pub fn query(&self, query_text: &str, n_results: usize) -> Vec<BM25Result> {
        let query_tokens = Self::tokenize(query_text);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let docs = self.docs.read();
        let df = self.doc_freq.read();
        let total = *self.total_len.read();

        let n = docs.len();
        if n == 0 {
            return Vec::new();
        }

        let avgdl = total as f64 / n as f64;

        let mut scores: Vec<(f64, usize)> = Vec::with_capacity(n);

        for (i, doc) in docs.iter().enumerate() {
            let dl = doc.doc_len as f64;
            let mut score = 0.0_f64;

            for qt in &query_tokens {
                let f = *doc.term_freqs.get(qt).unwrap_or(&0) as f64;
                if f == 0.0 {
                    continue;
                }
                let doc_freq = *df.get(qt).unwrap_or(&0) as f64;
                // BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let idf = ((n as f64 - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0).ln();
                // BM25 TF component
                score += idf * (f * (self.k1 + 1.0))
                    / (f + self.k1 * (1.0 - self.b + self.b * dl / avgdl));
            }

            if score > 0.0 {
                scores.push((score, i));
            }
        }

        // Partial sort: only need top-K
        scores.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n_results);

        scores
            .into_iter()
            .map(|(score, idx)| BM25Result {
                doc_id: docs[idx].id.clone(),
                score,
            })
            .collect()
    }

    /// Number of indexed documents.
    pub fn count(&self) -> usize {
        self.docs.read().len()
    }

    /// Clear all documents from the index.
    pub fn clear(&self) {
        self.docs.write().clear();
        self.doc_freq.write().clear();
        *self.total_len.write() = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_test_engine() -> BM25Engine {
        let engine = BM25Engine::default();
        engine.add_document(
            "d1",
            "Water boils at 100 degrees Celsius at standard pressure",
        );
        engine.add_document("d2", "The speed of light is 299792 kilometers per second");
        engine.add_document("d3", "DNA has four bases adenine thymine guanine cytosine");
        engine.add_document(
            "d4",
            "Mitochondria produce ATP through oxidative phosphorylation",
        );
        engine.add_document("d5", "Iron rusts when it reacts with oxygen and moisture");
        engine
    }

    #[test]
    fn test_basic_query() {
        let engine = build_test_engine();
        let results = engine.query("water boiling temperature", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "d1");
    }

    #[test]
    fn test_empty_query() {
        let engine = build_test_engine();
        let results = engine.query("", 3);
        assert!(results.is_empty());
    }

    #[test]
    fn test_no_match() {
        let engine = build_test_engine();
        let results = engine.query("quantum entanglement teleportation", 3);
        assert!(results.is_empty());
    }

    #[test]
    fn test_count() {
        let engine = build_test_engine();
        assert_eq!(engine.count(), 5);
    }

    #[test]
    fn test_clear() {
        let engine = build_test_engine();
        engine.clear();
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn test_top_k_limiting() {
        let engine = build_test_engine();
        // "the" appears in multiple docs
        let results = engine.query("the", 2);
        assert!(results.len() <= 2);
    }

    #[test]
    fn test_scoring_order() {
        let engine = BM25Engine::default();
        engine.add_document("exact", "water boils at 100 degrees");
        engine.add_document("partial", "water is a liquid");
        engine.add_document("none", "the sky is blue today");

        let results = engine.query("water boils at 100 degrees", 3);
        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "exact");
        if results.len() > 1 {
            assert!(results[0].score >= results[1].score);
        }
    }
}
