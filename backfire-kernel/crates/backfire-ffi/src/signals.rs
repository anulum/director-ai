// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — ffi::signals
//! Verification-signal and injection-detection bindings.

use pyo3::prelude::*;

/// Entity overlap between two texts (Jaccard of proper nouns).
#[pyfunction]
fn rust_entity_overlap(text_a: &str, text_b: &str) -> f64 {
    backfire_core::signals::entity_overlap(text_a, text_b)
}

/// Numerical consistency check between two texts.
#[pyfunction]
fn rust_numerical_consistency(text_a: &str, text_b: &str) -> Option<bool> {
    backfire_core::signals::numerical_consistency(text_a, text_b)
}

/// Detect negation flip between claim and source.
#[pyfunction]
fn rust_negation_flip(claim: &str, source: &str) -> bool {
    backfire_core::signals::negation_flip(claim, source)
}

/// Content word traceability (fraction of claim words in source).
#[pyfunction]
fn rust_traceability(claim: &str, source: &str) -> f64 {
    backfire_core::signals::traceability(claim, source)
}

/// Linear regression trend drop over coherence window.
#[pyfunction]
fn rust_trend_drop(values: Vec<f64>) -> f64 {
    backfire_core::signals::trend_drop(&values)
}

/// Per-claim bidirectional divergence scoring for injection detection.
///
/// For each claim, computes traceability, entity overlap, and
/// baseline-calibrated divergence against the intent.
///
/// Args:
///     claims: List of claim strings.
///     intent: The original intent string.
///     forward_scores: NLI scores (intent → claim) per claim.
///     reverse_scores: NLI scores (claim → intent) per claim.
///     baseline: Expected normal divergence (calibration baseline).
///
/// Returns:
///     List of (traceability, entity_match, calibrated_divergence) tuples.
#[pyfunction]
fn rust_bidirectional_divergence(
    claims: Vec<String>,
    intent: &str,
    forward_scores: Vec<f64>,
    reverse_scores: Vec<f64>,
    baseline: f64,
) -> Vec<(f64, f64, f64)> {
    let claim_refs: Vec<&str> = claims.iter().map(|s| s.as_str()).collect();
    backfire_core::signals::bidirectional_divergence(
        &claim_refs,
        intent,
        &forward_scores,
        &reverse_scores,
        baseline,
    )
}

/// Multi-signal injection verdict per claim.
///
/// Args:
///     calibrated_divs: Baseline-calibrated divergences per claim.
///     traceabilities: Content-word overlap per claim.
///     entity_matches: Entity Jaccard overlap per claim.
///     injection_threshold: Combined score threshold.
///     drift_threshold: Per-claim drift threshold.
///     injection_claim_threshold: High-divergence + low-trace threshold.
///     traceability_floor: Below this = fabrication override.
///     stage1_weight: Weight of sanitizer in combined score.
///     sanitizer_score: Stage 1 sanitizer score.
///
/// Returns:
///     Tuple of (verdicts, injection_risk, combined_score, detected) where
///     verdicts is list of (verdict_code, confidence) — 0=grounded, 1=drifted, 2=injected.
#[pyfunction]
#[pyo3(signature = (
    calibrated_divs,
    traceabilities,
    entity_matches,
    sanitizer_score,
    injection_threshold = 0.7,
    drift_threshold = 0.6,
    injection_claim_threshold = 0.75,
    traceability_floor = 0.15,
    stage1_weight = 0.3,
))]
#[allow(clippy::too_many_arguments)]
fn rust_injection_verdict(
    calibrated_divs: Vec<f64>,
    traceabilities: Vec<f64>,
    entity_matches: Vec<f64>,
    sanitizer_score: f64,
    injection_threshold: f64,
    drift_threshold: f64,
    injection_claim_threshold: f64,
    traceability_floor: f64,
    stage1_weight: f64,
) -> (Vec<(u8, f64)>, f64, f64, bool) {
    let cfg = backfire_core::signals::InjectionVerdictConfig {
        injection_threshold,
        drift_threshold,
        injection_claim_threshold,
        traceability_floor,
        stage1_weight,
    };
    let verdicts = backfire_core::signals::injection_verdicts(
        &calibrated_divs,
        &traceabilities,
        &entity_matches,
        &cfg,
    );
    let (risk, combined, detected) =
        backfire_core::signals::injection_aggregate(&verdicts, sanitizer_score, &cfg);
    (verdicts, risk, combined, detected)
}

/// Register the signal and injection functions on the Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Verification signals (Rust-accelerated)
    m.add_function(wrap_pyfunction!(rust_entity_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(rust_numerical_consistency, m)?)?;
    m.add_function(wrap_pyfunction!(rust_negation_flip, m)?)?;
    m.add_function(wrap_pyfunction!(rust_traceability, m)?)?;
    m.add_function(wrap_pyfunction!(rust_trend_drop, m)?)?;
    // Injection detection (Rust-accelerated)
    m.add_function(wrap_pyfunction!(rust_bidirectional_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(rust_injection_verdict, m)?)?;
    Ok(())
}
