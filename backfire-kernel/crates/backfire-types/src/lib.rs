// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — lib
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Types
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
#![deny(unsafe_code)]
//! Type definitions, configuration, and error hierarchy for the
//! Backfire Kernel — the real-time safety gate for Director-Class AI.

pub mod config;
pub mod error;
pub mod score;

pub use config::BackfireConfig;
pub use error::{BackfireError, BackfireResult};
pub use score::{CoherenceScore, ReviewResult, StreamSession, TokenEvent};
