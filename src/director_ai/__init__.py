# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Package Initialisation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Director-Class AI: Coherence Engine & SCPN Research Extensions.

Consumer API (always available)::

    from director_ai.core import CoherenceAgent, CoherenceScorer, SafetyKernel

Research API (requires ``pip install director-ai[research]``)::

    from director_ai.research.consilium import ConsiliumAgent
"""

__version__ = "0.5.0"

# ── Consumer core (always available) ──────────────────────────────────
from .core import (
    AsyncStreamingKernel,
    CoherenceAgent,
    CoherenceScore,
    CoherenceScorer,
    GroundTruthStore,
    InMemoryBackend,
    LLMGenerator,
    MockGenerator,
    NLIScorer,
    PhysicsBackedScorer,
    ReviewResult,
    SafetyKernel,
    StreamingKernel,
    StreamSession,
    TokenEvent,
    VectorGroundTruthStore,
)

__all__ = [
    # Core (consumer)
    "CoherenceAgent",
    "CoherenceScorer",
    "SafetyKernel",
    "MockGenerator",
    "LLMGenerator",
    "GroundTruthStore",
    "CoherenceScore",
    "ReviewResult",
    # v0.4.0
    "NLIScorer",
    "VectorGroundTruthStore",
    "InMemoryBackend",
    "StreamingKernel",
    "StreamSession",
    "TokenEvent",
    "PhysicsBackedScorer",
    # v0.5.0
    "AsyncStreamingKernel",
]

# ── Research extensions (optional) ────────────────────────────────────
try:
    from .research import (
        ConsiliumAgent,
        EthicalFunctional,
        L16Controller,
        L16OversightLoop,
        PGBOConfig,
        PGBOEngine,
        ProofResult,
        # Track 1 — Physics
        SECFunctional,
        SECResult,
        SSGFConfig,
        SSGFEngine,
        SSGFState,
        SystemState,
        TCBOConfig,
        TCBOController,
        TCBOControllerConfig,
        # Track 2 — Consciousness
        TCBOObserver,
        TorchUPDEConfig,
        TorchUPDEStepper,
        UPDEState,
        UPDEStepper,
        run_all_proofs,
    )

    __all__ += [
        "ConsiliumAgent",
        "EthicalFunctional",
        "SystemState",
        "SECFunctional",
        "SECResult",
        "L16Controller",
        "L16OversightLoop",
        "UPDEState",
        "UPDEStepper",
        "TCBOObserver",
        "TCBOConfig",
        "TCBOController",
        "TCBOControllerConfig",
        "PGBOEngine",
        "PGBOConfig",
        "SSGFEngine",
        "SSGFConfig",
        "SSGFState",
        "ProofResult",
        "run_all_proofs",
        "TorchUPDEStepper",
        "TorchUPDEConfig",
    ]
except ImportError:
    pass  # Research extras not installed — consumer API still works
