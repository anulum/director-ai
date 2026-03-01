// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Ground Truth Store (RAG Interface)
// ─────────────────────────────────────────────────────────────────────
//! RAG (Retrieval-Augmented Generation) interface for fact retrieval.
//!
//! The in-memory backend provides keyword matching for testing and
//! low-latency (<1ms) retrieval. Production deployments can plug in
//! ChromaDB or any vector DB via the `GroundTruthStore` trait.

use std::collections::HashMap;

/// Trait for ground truth retrieval backends.
pub trait GroundTruthStore: Send + Sync {
    /// Retrieve relevant context for the given query.
    /// Returns `None` if no relevant facts are found.
    fn retrieve_context(&self, query: &str) -> Option<String>;
}

/// In-memory keyword-based ground truth store.
///
/// Mirrors `GroundTruthStore` from `knowledge.py:12-56`.
pub struct InMemoryKnowledge {
    facts: HashMap<String, String>,
}

impl Default for InMemoryKnowledge {
    fn default() -> Self {
        let mut facts = HashMap::new();
        facts.insert("sky color".into(), "blue".into());
        facts.insert("system layers".into(), "16".into());
        facts.insert("sec metric".into(), "sustainable ethical coherence".into());
        facts.insert("backfire limit".into(), "entropy threshold".into());
        Self { facts }
    }
}

impl InMemoryKnowledge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_facts(facts: HashMap<String, String>) -> Self {
        Self { facts }
    }

    pub fn add_fact(&mut self, key: String, value: String) {
        self.facts.insert(key, value);
    }
}

impl GroundTruthStore for InMemoryKnowledge {
    fn retrieve_context(&self, query: &str) -> Option<String> {
        let query_lower = query.to_lowercase();
        let mut context = Vec::new();

        for (key, value) in &self.facts {
            let key_words: Vec<&str> = key.split_whitespace().collect();
            if key_words.iter().any(|w| query_lower.contains(w)) {
                context.push(format!("{key} is {value}"));
            }
        }

        if context.is_empty() {
            None
        } else {
            Some(context.join("; "))
        }
    }
}

/// External ground truth store that calls a function pointer.
///
/// Used by the PyO3 FFI layer to delegate RAG retrieval to Python.
type RetrieveFn = Box<dyn Fn(&str) -> Option<String> + Send + Sync>;

pub struct ExternalKnowledge {
    retrieve_fn: RetrieveFn,
}

impl ExternalKnowledge {
    pub fn new(retrieve_fn: impl Fn(&str) -> Option<String> + Send + Sync + 'static) -> Self {
        Self {
            retrieve_fn: Box::new(retrieve_fn),
        }
    }
}

impl GroundTruthStore for ExternalKnowledge {
    fn retrieve_context(&self, query: &str) -> Option<String> {
        (self.retrieve_fn)(query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_facts() {
        let store = InMemoryKnowledge::new();
        let ctx = store.retrieve_context("What color is the sky?");
        assert!(ctx.is_some());
        assert!(ctx.unwrap().contains("blue"));
    }

    #[test]
    fn test_system_layers() {
        let store = InMemoryKnowledge::new();
        let ctx = store.retrieve_context("How many system layers?");
        assert!(ctx.is_some());
        assert!(ctx.unwrap().contains("16"));
    }

    #[test]
    fn test_truly_no_match() {
        let store = InMemoryKnowledge::new();
        let ctx = store.retrieve_context("xyzzy plugh");
        assert!(ctx.is_none());
    }

    #[test]
    fn test_custom_facts() {
        let mut facts = HashMap::new();
        facts.insert("test key".into(), "test value".into());
        let store = InMemoryKnowledge::with_facts(facts);
        let ctx = store.retrieve_context("test query with key");
        assert!(ctx.is_some());
        assert!(ctx.unwrap().contains("test value"));
    }

    #[test]
    fn test_external_knowledge() {
        let store = ExternalKnowledge::new(|q| {
            if q.contains("sky") {
                Some("sky is blue".into())
            } else {
                None
            }
        });
        assert!(store.retrieve_context("sky color").is_some());
        assert!(store.retrieve_context("unrelated").is_none());
    }

    #[test]
    fn test_add_fact() {
        let mut store = InMemoryKnowledge::with_facts(HashMap::new());
        assert!(store.retrieve_context("capital").is_none());
        store.add_fact("capital".into(), "Paris".into());
        let ctx = store.retrieve_context("What is the capital?");
        assert!(ctx.is_some());
        assert!(ctx.unwrap().contains("Paris"));
    }

    #[test]
    fn test_add_fact_overwrites() {
        let mut store = InMemoryKnowledge::new();
        store.add_fact("sky color".into(), "green".into());
        let ctx = store.retrieve_context("sky color").unwrap();
        assert!(ctx.contains("green"));
        assert!(!ctx.contains("blue"));
    }
}
