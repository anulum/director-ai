// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — compute
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Compute Functions (Rust)
// ─────────────────────────────────────────────────────────────────────
//! Rust accelerators for CPU-bound Python functions.
//!
//! Each function mirrors a Python counterpart with identical semantics.
//! Python fallbacks exist; these are optional accelerators selected
//! automatically when `backfire_kernel` is importable.
//!
//! This is a facade: one submodule per responsibility, with every public
//! item re-exported here so callers keep the flat `compute::` paths.
//!
//! ## Functions
//!
//! - [`sanitizer_score`](crate::compute::sanitizer_score) — 11 regex injection patterns (InputSanitizer.score)
//! - [`has_suspicious_unicode`](crate::compute::has_suspicious_unicode) — suspicious Unicode character ratio check
//! - [`detect_task_type`](crate::compute::detect_task_type) — task classification from prompt text
//! - [`verify_numeric`](crate::compute::verify_numeric) — numeric consistency checks
//! - [`score_temporal_freshness`](crate::compute::score_temporal_freshness) — temporal claim staleness risk
//! - [`extract_reasoning_steps`](crate::compute::extract_reasoning_steps) — reasoning chain step extraction
//! - [`split_sentences`](crate::compute::split_sentences) — chunking sentence segmentation helper
//! - [`build_chunks`](crate::compute::build_chunks) — chunk builder with optional overlap
//! - [`word_overlap`](crate::compute::word_overlap) — Jaccard word overlap (heuristic NLI)
//! - [`softmax`](crate::compute::softmax) — row-wise softmax for NLI logits
//! - [`probs_to_divergence`](crate::compute::probs_to_divergence) — NLI probability → divergence score
//! - [`probs_to_confidence`](crate::compute::probs_to_confidence) — NLI probability → confidence score
//! - [`aggregate_chunk_scores`](crate::compute::aggregate_chunk_scores) — chunk score matrix aggregation
//! - [`aggregate_chunk_scores_confidence_weighted`](crate::compute::aggregate_chunk_scores_confidence_weighted) — confidence-weighted aggregation
//! - [`coverage_from_divergences`](crate::compute::coverage_from_divergences) — claim support/coverage reduction
//! - [`reduce_claim_attribution`](crate::compute::reduce_claim_attribution) — per-claim best source attribution reduction
//! - [`lite_score`](crate::compute::lite_score) — lightweight heuristic divergence (no-NLI fallback)
//! - [`lite_score_batch`](crate::compute::lite_score_batch) — batch version of lite_score
//! - [`heuristic_logical_divergence`](crate::compute::heuristic_logical_divergence) — keyword+overlap logical fallback
//! - [`heuristic_factual_divergence`](crate::compute::heuristic_factual_divergence) — overlap+negation+entity factual fallback
//! - [`eval_arithmetic`](crate::compute::eval_arithmetic) — arithmetic expression evaluation
//! - [`detect_fallacies`](crate::compute::detect_fallacies) — informal-fallacy marker detection
//! - [`merge_flagged_spans`](crate::compute::merge_flagged_spans) — flagged-token span merging

/// Parse and evaluate arithmetic expressions from output text.
mod arith;
/// Segment sentences and assemble overlapping passage windows.
mod chunking;
/// Detect text markers associated with informal fallacies.
mod fallacies;
/// Estimate logical and factual divergence without a model.
mod heuristics;
/// Convert and aggregate natural language inference scores.
mod nli_math;
/// Compare numeric expressions and report consistency issues.
mod numeric;
/// Extract reasoning steps and compute lexical overlap.
mod reasoning;
/// Score input-injection patterns and suspicious Unicode.
mod sanitize;
/// Merge adjacent flagged tokens into contiguous spans.
mod spans;
/// Infer the task category from prompt cues.
mod task_detect;
/// Estimate staleness risk for time-sensitive claims.
mod temporal;

pub use arith::eval_arithmetic;
pub use chunking::{build_chunks, split_sentences};
pub use fallacies::detect_fallacies;
pub use heuristics::{
    heuristic_factual_divergence, heuristic_logical_divergence, lite_score, lite_score_batch,
    NEGATION_FLIP_OVERLAP,
};
pub use nli_math::{
    aggregate_chunk_scores, aggregate_chunk_scores_confidence_weighted, coverage_from_divergences,
    probs_to_confidence, probs_to_divergence, reduce_claim_attribution, softmax,
};
pub use numeric::{verify_numeric, NumericIssue};
pub use reasoning::{extract_reasoning_steps, word_overlap};
pub use sanitize::{has_suspicious_unicode, sanitizer_score};
pub use spans::merge_flagged_spans;
pub use task_detect::detect_task_type;
pub use temporal::{score_temporal_freshness, TemporalClaim};
