// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Consciousness Gate
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Consciousness gate: TCBO observer and PGBO engine.
//!
//! - TCBO: persistent-homology–based consciousness boundary observable
//! - PGBO: phase→geometry bridge operator (symmetric rank-2 tensor)

pub mod pgbo;
pub mod tcbo;

pub use pgbo::{PGBOConfig, PGBOEngine};
pub use tcbo::{TCBOConfig, TCBOController, TCBOControllerConfig, TCBOObserver};
