// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Core Engine
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Core scoring engine, token gating, and safety kernel for
//! real-time AI output verification.
//!
//! The entire hot path (scoring + gating) must complete in ≤50ms
//! as specified in the Backfire Prevention Protocols §2.2.

pub mod kernel;
pub mod knowledge;
pub mod nli;
pub mod scorer;

pub use kernel::{SafetyKernel, StreamingKernel};
pub use knowledge::{GroundTruthStore, InMemoryKnowledge};
pub use nli::{HeuristicNli, NliBackend};
pub use scorer::CoherenceScorer;
