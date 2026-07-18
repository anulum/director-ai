// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute::task_detect
//! Task-type classification from prompt and response text.
//!
//! Mirrors `detect_task_type()` from `_task_scoring.py`.

use once_cell::sync::Lazy;
use regex::Regex;

static DIALOGUE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?i)(?:^|\s)(?:(?:User|Human|Customer|Student|Interviewer|Speaker|Assistant|AI|Bot|Agent|Interviewee|System)[\s\d]*:|\[(?:User|Human|Assistant|AI|System)\])"
    ).unwrap()
});

static SUMMARIZE_KW: &[&str] = &["summarize", "summary", "summarise", "tldr", "abstract"];

static RAG_KW: &[&str] = &[
    "based on the context",
    "based on the following",
    "given the document",
    "given the passage",
    "retrieved",
    "source document",
    "reference text",
];

static FACT_CHECK_KW: &[&str] = &["verify", "fact-check", "is it true", "claim", "support"];

static QA_KW: &[&str] = &["answer the question", "based on the", "according to"];

/// Detect task type from prompt content and response length ratio.
///
/// Returns one of: "dialogue", "summarization", "rag", "fact_check", "qa", "default".
/// Mirrors `detect_task_type()` from `_task_scoring.py`.
pub fn detect_task_type(prompt: &str, response: &str) -> String {
    let matches = DIALOGUE_RE.find_iter(prompt).count();
    if matches >= 2 {
        return "dialogue".to_string();
    }

    let lower = prompt.to_lowercase();

    if SUMMARIZE_KW.iter().any(|kw| lower.contains(kw)) {
        return "summarization".to_string();
    }

    if !response.is_empty() && prompt.len() > 1000 && response.len() > 20 {
        let ratio = response.len() as f64 / prompt.len() as f64;
        if ratio < 0.30 {
            return "summarization".to_string();
        }
    }

    if RAG_KW.iter().any(|kw| lower.contains(kw)) {
        return "rag".to_string();
    }

    if FACT_CHECK_KW.iter().any(|kw| lower.contains(kw)) {
        return "fact_check".to_string();
    }

    if prompt.contains('?') || QA_KW.iter().any(|kw| lower.contains(kw)) {
        return "qa".to_string();
    }

    "default".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_dialogue() {
        assert_eq!(
            detect_task_type("User: hello\nAssistant: hi\nUser: how are you?", ""),
            "dialogue"
        );
    }

    #[test]
    fn test_detect_summarization_keyword() {
        assert_eq!(
            detect_task_type("Please summarize the following article", ""),
            "summarization"
        );
    }

    #[test]
    fn test_detect_summarization_ratio() {
        let prompt = "x".repeat(2000);
        let response = "This is a short summary of the content above."; // > 20 chars
        assert_eq!(detect_task_type(&prompt, response), "summarization");
    }

    #[test]
    fn test_detect_rag() {
        assert_eq!(
            detect_task_type("Based on the context, what is X?", ""),
            "rag"
        );
    }

    #[test]
    fn test_detect_fact_check() {
        assert_eq!(
            detect_task_type("Verify this claim about climate", ""),
            "fact_check"
        );
    }

    #[test]
    fn test_detect_qa() {
        assert_eq!(detect_task_type("What is 2+2?", ""), "qa");
    }

    #[test]
    fn test_detect_default() {
        assert_eq!(detect_task_type("Tell me a joke", ""), "default");
    }
}
