// SPDX-License-Identifier: Apache-2.0
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — lib
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Boundary Observers (TCBO + PGBO)
// (C) 1998-2026 Miroslav Šotek. All rights reserved.
// License: Apache-2.0
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
