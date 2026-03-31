// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — lib
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SCPN Physics Engine
// (C) 1998-2026 Miroslav Šotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
#![deny(unsafe_code)]
//! SCPN physics: UPDE Kuramoto integrator, L16 cybernetic closure,
//! and SEC Lyapunov functional for the 16-layer hierarchy.

pub mod l16_closure;
pub mod params;
pub mod sec_functional;
pub mod upde;

pub use l16_closure::{L16Controller, L16ControllerState, PIState};
pub use params::{build_knm_matrix, LAYER_NAMES, N_LAYERS, OMEGA_N};
pub use sec_functional::{SECFunctional, SECResult};
pub use upde::{SimpleRng, UPDEState, UPDEStepper};
