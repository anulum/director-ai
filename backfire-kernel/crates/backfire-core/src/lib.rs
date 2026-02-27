// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Core Engine
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
#![deny(unsafe_code)]
//! Core scoring engine, token gating, and safety kernel for
//! real-time AI output verification.
//!
//! The entire hot path (scoring + gating) must complete in ≤50ms
//! as specified in the Backfire Prevention Protocols §2.2.
//!
//! # Safety Invariants
//!
//! 1. **Halt is irreversible within a session**: once `is_active` transitions
//!    to `false` via `emergency_stop()`, no token passes the gate until
//!    `reactivate()` is called explicitly. The `AtomicBool` uses `SeqCst`
//!    ordering — no token decision can be reordered across a halt.
//!
//! 2. **Non-finite scores are lethal**: any NaN or Inf from a coherence
//!    callback is replaced with 0.0 (guaranteed halt). Callback panics
//!    are caught via `catch_unwind` and treated identically.
//!
//! 3. **All three halt mechanisms are independent**: hard limit, sliding
//!    window average, and downward trend each trigger independently.
//!    A single-token spike cannot mask a sustained coherence decline.
//!
//! 4. **No allocations in the hot path**: `VecDeque` and `Vec` are
//!    pre-allocated before the token loop. The scoring loop itself
//!    performs only arithmetic, comparisons, and atomic loads/stores.

pub mod kernel;
pub mod knowledge;
pub mod nli;
pub mod scorer;

pub use kernel::{SafetyKernel, StreamingKernel};
pub use knowledge::{GroundTruthStore, InMemoryKnowledge};
pub use nli::{HeuristicNli, NliBackend};
pub use scorer::CoherenceScorer;
