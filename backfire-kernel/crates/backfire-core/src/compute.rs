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
//! - [`sanitizer_score`] — 11 regex injection patterns (InputSanitizer.score)
//! - [`has_suspicious_unicode`] — suspicious Unicode character ratio check
//! - [`detect_task_type`] — task classification from prompt text
//! - [`verify_numeric`] — numeric consistency checks
//! - [`score_temporal_freshness`] — temporal claim staleness risk
//! - [`extract_reasoning_steps`] — reasoning chain step extraction
//! - [`split_sentences`] — chunking sentence segmentation helper
//! - [`build_chunks`] — chunk builder with optional overlap
//! - [`word_overlap`] — Jaccard word overlap (heuristic NLI)
//! - [`softmax`] — row-wise softmax for NLI logits
//! - [`probs_to_divergence`] — NLI probability → divergence score
//! - [`probs_to_confidence`] — NLI probability → confidence score
//! - [`aggregate_chunk_scores`] — chunk score matrix aggregation
//! - [`aggregate_chunk_scores_confidence_weighted`] — confidence-weighted aggregation
//! - [`coverage_from_divergences`] — claim support/coverage reduction
//! - [`reduce_claim_attribution`] — per-claim best source attribution reduction
//! - [`lite_score`] — lightweight heuristic divergence (no-NLI fallback)
//! - [`lite_score_batch`] — batch version of lite_score
//! - [`heuristic_logical_divergence`] — keyword+overlap logical fallback
//! - [`heuristic_factual_divergence`] — overlap+negation+entity factual fallback
//! - [`eval_arithmetic`] — arithmetic expression evaluation
//! - [`detect_fallacies`] — informal-fallacy marker detection
//! - [`merge_flagged_spans`] — flagged-token span merging

mod arith;
mod chunking;
mod fallacies;
mod heuristics;
mod nli_math;
mod numeric;
mod reasoning;
mod sanitize;
mod spans;
mod task_detect;
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
