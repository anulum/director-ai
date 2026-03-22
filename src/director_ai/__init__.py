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
    ClaimVerdict,
    CoherenceAgent,
    CoherenceScore,
    CoherenceScorer,
    ConversationSession,
    DatasetTypeClassifier,
    DirectorConfig,
    EvidenceChunk,
    GroundTruthStore,
    HaltEvidence,
    HaltMonitor,
    InMemoryBackend,
    InputSanitizer,
    LiteScorer,
    LLMGenerator,
    MetaClassifier,
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
    TuneResult,
    Turn,
    VectorGroundTruthStore,
    VerificationResult,
    VerifiedScorer,
    clear_model_cache,
    export_onnx,
    export_tensorrt,
    get_backend,
    get_vector_backend,
    list_backends,
    list_vector_backends,
    nli_available,
    register_backend,
    register_vector_backend,
    tune,
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
    "ClaimVerdict",
    "CoherenceAgent",
    "CoherenceError",
    "CoherenceScore",
    "CoherenceScorer",
    "ConversationSession",
    "DatasetTypeClassifier",
    "DependencyError",
    "DirectorAIError",
    "DirectorConfig",
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
    "MetaClassifier",
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
    "TuneResult",
    "Turn",
    "ValidationError",
    "VectorGroundTruthStore",
    "VerificationResult",
    "VerifiedScorer",
    "clear_model_cache",
    "export_onnx",
    "export_tensorrt",
    "get_score",
    "guard",
    "list_backends",
    "nli_available",
    "score",
    "tune",
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
