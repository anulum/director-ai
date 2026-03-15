# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Package Initialisation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Director-AI: Real-time LLM hallucination guardrail.

::

    from director_ai.core import CoherenceAgent, CoherenceScorer, SafetyKernel
"""

__version__ = "3.9.0"

from .core import (
    AsyncStreamingKernel,
    ClaimAttribution,
    CoherenceAgent,
    CoherenceScore,
    CoherenceScorer,
    ConversationSession,
    EvidenceChunk,
    GroundTruthStore,
    HaltEvidence,
    InMemoryBackend,
    InputSanitizer,
    LiteScorer,
    LLMGenerator,
    MockGenerator,
    NLIScorer,
    ReviewResult,
    SafetyKernel,
    SanitizeResult,
    ScoreCache,
    ScorerBackend,
    ScoringEvidence,
    SentenceTransformerBackend,
    ShardedNLIScorer,
    StreamingKernel,
    StreamSession,
    TokenEvent,
    Turn,
    VectorGroundTruthStore,
    get_backend,
    get_vector_backend,
    list_backends,
    list_vector_backends,
    register_backend,
    register_vector_backend,
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
from .integrations.sdk_guard import get_score, guard, score

__all__ = [
    "ClaimAttribution",
    "CoherenceAgent",
    "CoherenceError",
    "CoherenceScore",
    "CoherenceScorer",
    "DependencyError",
    "DirectorAIError",
    "GeneratorError",
    "GroundTruthStore",
    "HallucinationError",
    "KernelHaltError",
    "LLMGenerator",
    "MockGenerator",
    "NumericalError",
    "PhysicsError",
    "ReviewResult",
    "SafetyKernel",
    "StreamingKernel",
    "ValidationError",
    "get_score",
    "guard",
    "score",
]

_MOVED_TO_ENTERPRISE = {
    "TenantRouter",
    "Policy",
    "Violation",
    "AuditLogger",
    "AuditEntry",
}


def __getattr__(name: str):
    if name in _MOVED_TO_ENTERPRISE:
        raise ImportError(
            f"{name} moved to director_ai.enterprise in v3.0. "
            f"Use: from director_ai.enterprise import {name}",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
