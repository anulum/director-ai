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

__version__ = "2.2.0"

from .core import (
    AsyncStreamingKernel,
    CoherenceAgent,
    CoherenceScore,
    CoherenceScorer,
    EvidenceChunk,
    GroundTruthStore,
    HaltEvidence,
    InMemoryBackend,
    InputSanitizer,
    LLMGenerator,
    MockGenerator,
    NLIScorer,
    ReviewResult,
    SafetyKernel,
    SanitizeResult,
    ScoreCache,
    ScoringEvidence,
    SentenceTransformerBackend,
    StreamingKernel,
    StreamSession,
    TokenEvent,
    VectorGroundTruthStore,
)
from .core.exceptions import (
    CoherenceError,
    DependencyError,
    DirectorAIError,
    GeneratorError,
    HallucinationError,
    KernelHaltError,
    NumericalError,
    PhysicsError,
    ValidationError,
)
from .integrations.sdk_guard import get_score, guard

__all__ = [
    "CoherenceAgent",
    "CoherenceScorer",
    "SafetyKernel",
    "MockGenerator",
    "LLMGenerator",
    "GroundTruthStore",
    "CoherenceScore",
    "EvidenceChunk",
    "HaltEvidence",
    "ScoringEvidence",
    "ReviewResult",
    "NLIScorer",
    "VectorGroundTruthStore",
    "InMemoryBackend",
    "SentenceTransformerBackend",
    "ScoreCache",
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
    "guard",
    "get_score",
    "DirectorAIError",
    "CoherenceError",
    "KernelHaltError",
    "GeneratorError",
    "ValidationError",
    "DependencyError",
    "HallucinationError",
    "PhysicsError",
    "NumericalError",
]

_LAZY_ENTERPRISE = {
    "TenantRouter",
    "Policy",
    "Violation",
    "AuditLogger",
    "AuditEntry",
}


def __getattr__(name: str):
    if name in _LAZY_ENTERPRISE:
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
