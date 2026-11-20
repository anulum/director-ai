// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-AI — native scoring and token gates
#![deny(unsafe_code)]
//! Core scoring engine, token gating, and safety kernel for
//! real-time AI output verification.
//!
//! The scoring and gating design budget is 50 ms. Deployment latency depends
//! on the host, callback and scoring backend and must be measured there.
//!
//! # Safety Invariants
//!
//! 1. **Halts persist until reactivation**: every streaming halt deactivates
//!    the gate. Subsequent calls refuse before scoring another token until
//!    the host explicitly calls `reactivate()`. Gate flags use sequentially
//!    consistent atomic loads and stores.
//!
//! 2. **Nonfinite scores become zero**: NaN, either infinity and callback
//!    panics use a zero fallback. The configured thresholds determine the
//!    decision; a zero hard limit alone permits zero.
//!
//! 3. **Checks have a defined order**: hard limit, window average, then
//!    downward trend. The trend uses the shared least-squares projected drop
//!    over the recent score window, also used by the Python and WASM paths.
//!
//! 4. **Traces allocate storage**: streaming retains token strings, scores
//!    and events. Window and trace storage contribute to runtime cost.

/// Sparse term-based retrieval with BM25 ranking.
pub mod bm25;
/// CPU kernels for scoring and signal processing.
pub mod compute;
/// Per-token and rolling-window coherence gates.
pub mod kernel;
/// Fact retrieval interfaces and in-memory ground truth.
pub mod knowledge;
/// Natural language inference interface and heuristic backend.
pub mod nli;
/// Pattern matching for personally identifiable information.
pub mod pii_scan;
/// Logical and factual coherence scoring.
pub mod scorer;
/// Verification signals, including the shared projected trend drop.
pub mod signals;
/// Reductions, confidence intervals and calibration statistics.
pub mod stats;

pub use bm25::BM25Engine;
pub use kernel::{SafetyKernel, StreamingKernel};
pub use knowledge::{GroundTruthStore, InMemoryKnowledge};
pub use nli::{HeuristicNli, NliBackend};
pub use pii_scan::{PiiMatch, PiiScanner};
pub use scorer::CoherenceScorer;
