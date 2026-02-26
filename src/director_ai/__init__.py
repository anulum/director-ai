# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Package Initialisation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Director-AI: Real-time LLM hallucination guardrail.

::

    from director_ai.core import CoherenceAgent, CoherenceScorer, SafetyKernel
"""

__version__ = "1.0.0"

from .core import (
    AsyncStreamingKernel,
    AuditEntry,
    AuditLogger,
    CoherenceAgent,
    CoherenceScore,
    CoherenceScorer,
    GroundTruthStore,
    InMemoryBackend,
    InputSanitizer,
    LLMGenerator,
    MockGenerator,
    NLIScorer,
    Policy,
    ReviewResult,
    SafetyKernel,
    SanitizeResult,
    StreamingKernel,
    StreamSession,
    TenantRouter,
    TokenEvent,
    VectorGroundTruthStore,
    Violation,
)

__all__ = [
    "CoherenceAgent",
    "CoherenceScorer",
    "SafetyKernel",
    "MockGenerator",
    "LLMGenerator",
    "GroundTruthStore",
    "CoherenceScore",
    "ReviewResult",
    "NLIScorer",
    "VectorGroundTruthStore",
    "InMemoryBackend",
    "StreamingKernel",
    "StreamSession",
    "TokenEvent",
    "AsyncStreamingKernel",
    "Policy",
    "Violation",
    "AuditLogger",
    "AuditEntry",
    "TenantRouter",
    "InputSanitizer",
    "SanitizeResult",
]
