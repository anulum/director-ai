// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Boundary Observers (TCBO + PGBO)
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
#![deny(unsafe_code)]
//! Boundary observers: TCBO observer and PGBO engine.
//!
//! - TCBO: persistent-homology–based topological boundary observable
//! - PGBO: phase→geometry bridge operator (symmetric rank-2 tensor)

pub mod pgbo;
pub mod tcbo;

pub use pgbo::{PGBOConfig, PGBOEngine};
pub use tcbo::{TCBOConfig, TCBOController, TCBOControllerConfig, TCBOObserver};
