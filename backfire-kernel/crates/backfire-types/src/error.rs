// SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — error
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Backfire Kernel Error Hierarchy
// Mirrors: src/director_ai/core/exceptions.py
// ─────────────────────────────────────────────────────────────────────

use thiserror::Error;

/// Root error type for all Backfire Kernel failures.
#[derive(Error, Debug)]
pub enum BackfireError {
    /// Coherence pipeline encountered a fatal scoring error.
    #[error("coherence error: {0}")]
    Coherence(String),

    /// Safety kernel triggered an emergency halt.
    #[error("kernel halt: {0}")]
    KernelHalt(String),

    /// Candidate generation failed unrecoverably.
    #[error("generator error: {0}")]
    Generator(String),

    /// Invalid input (prompt, parameters, config).
    #[error("validation error: {0}")]
    Validation(String),

    /// NLI model or inference bridge failed.
    #[error("NLI error: {0}")]
    Nli(String),

    /// RAG retrieval failed.
    #[error("knowledge store error: {0}")]
    Knowledge(String),

    /// Scoring timed out (exceeded 50ms deadline).
    #[error("timeout: scoring exceeded {deadline_ms}ms deadline")]
    Timeout { deadline_ms: u64 },

    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// Numerical error (NaN/Inf in computation).
    #[error("numerical error: {0}")]
    Numerical(String),
}

pub type BackfireResult<T> = Result<T, BackfireError>;
