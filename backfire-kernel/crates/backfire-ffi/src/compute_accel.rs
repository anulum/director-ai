// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::compute_accel
//! Compute-accelerator bindings: thin wrappers over
//! `backfire_core::compute` (sanitizer, task detection, verification,
//! chunking, NLI math, lite/heuristic scorers).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn rust_sanitizer_score(text: &str) -> (f64, Vec<String>) {
    backfire_core::compute::sanitizer_score(text)
}

#[pyfunction]
fn rust_has_suspicious_unicode(text: &str) -> bool {
    backfire_core::compute::has_suspicious_unicode(text)
}

#[pyfunction]
fn rust_detect_task_type(prompt: &str, response: &str) -> String {
    backfire_core::compute::detect_task_type(prompt, response)
}

type NumericIssuesTuple = Vec<(String, String, String, String)>;

#[pyfunction]
fn rust_verify_numeric(text: &str, current_year: i32) -> (usize, NumericIssuesTuple, bool) {
    let (claims, issues, valid) = backfire_core::compute::verify_numeric(text, current_year);
    let issues_tuples: Vec<(String, String, String, String)> = issues
        .into_iter()
        .map(|i| (i.issue_type, i.description, i.severity, i.context))
        .collect();
    (claims, issues_tuples, valid)
}

#[pyfunction]
fn rust_score_temporal_freshness(text: &str) -> (Vec<(String, String, f64)>, f64, bool) {
    let (claims, overall, has) = backfire_core::compute::score_temporal_freshness(text);
    let claims_tuples: Vec<(String, String, f64)> = claims
        .into_iter()
        .map(|c| (c.text, c.claim_type, c.staleness_risk))
        .collect();
    (claims_tuples, overall, has)
}

#[pyfunction]
fn rust_extract_reasoning_steps(text: &str) -> Vec<String> {
    backfire_core::compute::extract_reasoning_steps(text)
}

#[pyfunction]
fn rust_split_sentences(text: &str) -> Vec<String> {
    backfire_core::compute::split_sentences(text)
}

#[pyfunction]
fn rust_build_chunks(sentences: Vec<String>, budget: usize, overlap_ratio: f64) -> Vec<String> {
    backfire_core::compute::build_chunks(&sentences, budget, overlap_ratio)
}

#[pyfunction]
fn rust_word_overlap(text_a: &str, text_b: &str) -> f64 {
    backfire_core::compute::word_overlap(text_a, text_b)
}

#[pyfunction]
fn rust_eval_arithmetic(expr: &str) -> f64 {
    backfire_core::compute::eval_arithmetic(expr)
}

#[pyfunction]
fn rust_detect_fallacies(text: &str) -> Vec<(String, String)> {
    backfire_core::compute::detect_fallacies(text)
}

#[pyfunction]
fn rust_softmax(logits: Vec<f64>, cols: usize) -> Vec<f64> {
    backfire_core::compute::softmax(&logits, cols)
}

#[pyfunction]
fn rust_probs_to_divergence(
    probs: Vec<f64>,
    cols: usize,
    contradiction_idx: usize,
    neutral_idx: usize,
) -> Vec<f64> {
    backfire_core::compute::probs_to_divergence(&probs, cols, contradiction_idx, neutral_idx)
}

#[pyfunction]
fn rust_probs_to_confidence(probs: Vec<f64>, cols: usize) -> Vec<f64> {
    backfire_core::compute::probs_to_confidence(&probs, cols)
}

#[pyfunction]
fn rust_aggregate_chunk_scores(
    flat_scores: Vec<f64>,
    n_prem: usize,
    n_hyp: usize,
    inner_agg: &str,
    outer_agg: &str,
) -> (f64, Vec<f64>) {
    backfire_core::compute::aggregate_chunk_scores(
        &flat_scores,
        n_prem,
        n_hyp,
        inner_agg,
        outer_agg,
    )
}

#[pyfunction]
fn rust_merge_flagged_spans(
    offsets: Vec<(i64, i64)>,
    scores: Vec<f64>,
    response: &str,
    threshold: f64,
) -> (Vec<(i64, i64, f64)>, usize, f64) {
    let response_chars: Vec<char> = response.chars().collect();
    backfire_core::compute::merge_flagged_spans(&offsets, &scores, &response_chars, threshold)
}

#[pyfunction]
fn rust_aggregate_chunk_scores_confidence_weighted(
    flat_scores: Vec<f64>,
    flat_confidences: Vec<f64>,
    n_prem: usize,
    n_hyp: usize,
    inner_agg: &str,
) -> (f64, Vec<f64>) {
    backfire_core::compute::aggregate_chunk_scores_confidence_weighted(
        &flat_scores,
        &flat_confidences,
        n_prem,
        n_hyp,
        inner_agg,
    )
}

#[pyfunction]
fn rust_coverage_from_divergences(divergences: Vec<f64>, support_threshold: f64) -> (f64, usize) {
    backfire_core::compute::coverage_from_divergences(&divergences, support_threshold)
}

#[pyfunction]
fn rust_reduce_claim_attribution(
    flat_divergences: Vec<f64>,
    n_claims: usize,
    n_src: usize,
) -> PyResult<(Vec<f64>, Vec<usize>)> {
    if n_claims == 0 {
        return Err(PyValueError::new_err("n_claims must be >= 1"));
    }
    if n_src == 0 {
        return Err(PyValueError::new_err("n_src must be >= 1"));
    }
    let expected = n_claims
        .checked_mul(n_src)
        .ok_or_else(|| PyValueError::new_err("n_claims * n_src overflow"))?;
    if flat_divergences.len() != expected {
        return Err(PyValueError::new_err(format!(
            "flat_divergences length mismatch: expected {expected}, got {}",
            flat_divergences.len()
        )));
    }
    Ok(backfire_core::compute::reduce_claim_attribution(
        &flat_divergences,
        n_claims,
        n_src,
    ))
}

#[pyfunction]
fn rust_lite_score(premise: &str, hypothesis: &str) -> f64 {
    backfire_core::compute::lite_score(premise, hypothesis)
}

#[pyfunction]
fn rust_lite_score_batch(pairs: Vec<(String, String)>) -> Vec<f64> {
    backfire_core::compute::lite_score_batch(&pairs)
}

#[pyfunction]
fn rust_heuristic_logical_divergence(text_output: &str, prompt: &str) -> f64 {
    backfire_core::compute::heuristic_logical_divergence(text_output, prompt)
}

#[pyfunction]
fn rust_heuristic_factual_divergence(context: &str, text_output: &str) -> f64 {
    backfire_core::compute::heuristic_factual_divergence(context, text_output)
}

/// Register the compute accelerators on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_sanitizer_score, m)?)?;
    m.add_function(wrap_pyfunction!(rust_has_suspicious_unicode, m)?)?;
    m.add_function(wrap_pyfunction!(rust_detect_task_type, m)?)?;
    m.add_function(wrap_pyfunction!(rust_verify_numeric, m)?)?;
    m.add_function(wrap_pyfunction!(rust_score_temporal_freshness, m)?)?;
    m.add_function(wrap_pyfunction!(rust_extract_reasoning_steps, m)?)?;
    m.add_function(wrap_pyfunction!(rust_split_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(rust_build_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(rust_word_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(rust_eval_arithmetic, m)?)?;
    m.add_function(wrap_pyfunction!(rust_detect_fallacies, m)?)?;
    m.add_function(wrap_pyfunction!(rust_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(rust_probs_to_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_probs_to_confidence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_aggregate_chunk_scores, m)?)?;
    m.add_function(wrap_pyfunction!(rust_merge_flagged_spans, m)?)?;
    m.add_function(wrap_pyfunction!(
        rust_aggregate_chunk_scores_confidence_weighted,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(rust_coverage_from_divergences, m)?)?;
    m.add_function(wrap_pyfunction!(rust_reduce_claim_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(rust_lite_score, m)?)?;
    m.add_function(wrap_pyfunction!(rust_lite_score_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rust_heuristic_logical_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_heuristic_factual_divergence, m)?)?;
    Ok(())
}
