// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SSGF Geometry Engine
// (C) 1998-2026 Miroslav Sotek. All rights reserved.
// License: GNU AGPL v3 | Commercial licensing available
// ─────────────────────────────────────────────────────────────────────
//! Stochastic Synthesis of Geometric Fields (SSGF).
//!
//! Two-timescale engine converting Kuramoto microcycles into stable
//! geometry carriers W(t), with spectral observables (Fiedler value,
//! spectral gap) feeding back to stabilise the oscillator dynamics.
//!
//! Architecture:
//!   - NodeSpace: pre-allocated state container
//!   - Decoder: z → W (gram_softplus or RBF)
//!   - MicroCycleEngine: Kuramoto + geometry feedback + PGBO coupling
//!   - SpectralBridge: normalised Laplacian → eigenpairs (Jacobi)
//!   - Costs: 8 cost terms → U_total
//!   - SSGFEngine: 10-step outer-cycle orchestrator

pub mod costs;
pub mod decoder;
pub mod engine;
pub mod micro;
pub mod node_space;
pub mod spectral;

pub use costs::{CostBreakdown, CostWeights};
pub use decoder::GradientMethod;
pub use engine::{AudioMapping, SSGFConfig, SSGFEngine, SSGFStepLog};
pub use micro::MicroCycleEngine;
pub use node_space::NodeSpace;
pub use spectral::{GaugeMethod, SpectralBridge};
