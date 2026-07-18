// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — lib
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel PyO3 FFI Bindings
// © 1998-2026 Miroslav Šotek. All rights reserved.
// License: Apache-2.0
// ─────────────────────────────────────────────────────────────────────
// Note: #[deny(unsafe_code)] not applied — PyO3 proc macros generate
// unsafe blocks internally. All hand-written code in this crate is safe.
//! Python-callable wrappers around the Rust Backfire Kernel.
//!
//! Exposes `RustSafetyKernel`, `RustStreamingKernel`, `RustCoherenceScorer`,
//! and supporting types to Python via PyO3.
//!
//! This is a facade: one binding submodule per subsystem, each with a
//! `register` hook called from the `#[pymodule]` entry point below, so
//! the flat `backfire_kernel.*` Python surface is unchanged.
//!
//! # FFI Safety
//!
//! - GIL acquired via `Python::attach` before every Python callback.
//! - Python exceptions → safe Rust defaults (0.0 for scores, None for strings).
//! - No borrowed references escape the GIL lock scope.
//! - All config validated before storage (`BackfireConfig::validate()`).
//!
//! Install: `cd backfire-kernel && pip install -e crates/backfire-ffi`
//! (requires maturin).
//!
//! Usage from Python:
//! ```python
//! from backfire_kernel import RustSafetyKernel, RustStreamingKernel
//!
//! kernel = RustSafetyKernel(hard_limit=0.5)
//! result = kernel.stream_output(["Hello ", "world"], lambda t: 0.8)
//! ```

use pyo3::prelude::*;

mod compute_accel;
mod core_gate;
mod observers;
mod physics;
mod pii;
mod retrieval;
mod safety_hooks;
mod signals;
mod ssgf;
mod stats;
mod zk_range;

// Backfire Kernel — Rust-accelerated safety gate for Director-Class AI.
//
// This module exposes the entire hot-path safety gate to Python:
// - BackfireConfig, RustSafetyKernel, RustStreamingKernel
// - RustCoherenceScorer, CoherenceScore, StreamSession
// - Verification signal functions (Rust-accelerated)
// - RustBM25 — BM25 sparse retrieval engine

#[pymodule]
fn backfire_kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Core safety gate
    core_gate::register(m)?;
    // Physics engine
    physics::register(m)?;
    // Boundary observers
    observers::register(m)?;
    // SSGF geometry engine
    ssgf::register(m)?;
    // Zero-knowledge range-proof attestation (Bulletproofs over Ristretto)
    m.add_function(wrap_pyfunction!(
        zk_range::rust_bulletproof_prove_threshold,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        zk_range::rust_bulletproof_verify_threshold,
        m
    )?)?;
    // Verification signals + injection detection (Rust-accelerated)
    signals::register(m)?;
    // BM25 retrieval engine
    retrieval::register(m)?;
    // Compute accelerators
    compute_accel::register(m)?;
    // Statistical helpers
    stats::register(m)?;
    // Heuristic parity constants (mirror _heuristics.py)
    m.add(
        "NEGATION_FLIP_OVERLAP",
        backfire_core::compute::NEGATION_FLIP_OVERLAP,
    )?;
    // PII regex multi-pattern scanner
    pii::register(m)?;
    // Safety-hook acceleration (cyber-physical geometry/IK +
    // zk-attestation Merkle + challenge derivation)
    safety_hooks::register(m)?;
    Ok(())
}
