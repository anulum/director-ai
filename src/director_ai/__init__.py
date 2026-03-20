# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Package Initialisation

"""Director-AI: Real-time LLM hallucination guardrail.

::

    from director_ai.core import CoherenceAgent, CoherenceScorer, HaltMonitor
"""

__version__ = "3.9.4"

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
    HaltMonitor,
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
    "AsyncStreamingKernel",
    "ClaimAttribution",
    "CoherenceAgent",
    "CoherenceError",
    "CoherenceScore",
    "CoherenceScorer",
    "ConversationSession",
    "DependencyError",
    "DirectorAIError",
    "EvidenceChunk",
    "GeneratorError",
    "GroundTruthStore",
    "HallucinationError",
    "HaltEvidence",
    "HaltMonitor",
    "InMemoryBackend",
    "InputSanitizer",
    "KernelHaltError",
    "LLMGenerator",
    "LiteScorer",
    "MockGenerator",
    "NLIScorer",
    "NumericalError",
    "PhysicsError",
    "ReviewResult",
    "SafetyKernel",
    "SanitizeResult",
    "ScoreCache",
    "ScorerBackend",
    "ScoringEvidence",
    "SentenceTransformerBackend",
    "ShardedNLIScorer",
    "StreamSession",
    "StreamingKernel",
    "TokenEvent",
    "Turn",
    "ValidationError",
    "VectorGroundTruthStore",
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
