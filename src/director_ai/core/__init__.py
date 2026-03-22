# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Core Package (Coherence Engine)

"""Coherence Engine â€” consumer-ready AI output verification.

Subpackages::

    core.scoring    â€” NLI, heuristic, and verified scorers
    core.retrieval  â€” knowledge stores, vector backends, chunking
    core.runtime    â€” streaming kernels, sessions, batching
    core.safety     â€” sanitizer, policy, audit
    core.training   â€” fine-tuning, threshold tuning, benchmarking

All public symbols are re-exported here for backward compatibility::

    from director_ai.core import CoherenceScorer, HaltMonitor, ...
"""

# --- Scoring ---
# --- Core-level modules (not moved) ---
from .actor import LLMGenerator, MockGenerator
from .agent import CoherenceAgent
from .cache import ScoreCache
from .config import DirectorConfig

# --- Retrieval ---
from .retrieval.knowledge import GroundTruthStore
from .retrieval.vector_store import (
    ChromaBackend,
    ColBERTBackend,
    ElasticsearchBackend,
    FAISSBackend,
    HybridBackend,
    InMemoryBackend,
    PineconeBackend,
    QdrantBackend,
    RerankedBackend,
    SentenceTransformerBackend,
    VectorBackend,
    VectorGroundTruthStore,
    WeaviateBackend,
    get_vector_backend,
    list_vector_backends,
    register_vector_backend,
)

# --- Runtime ---
from .runtime.async_streaming import AsyncStreamingKernel
from .runtime.batch import BatchProcessor, BatchResult
from .runtime.kernel import HaltMonitor, SafetyKernel
from .runtime.review_queue import ReviewQueue
from .runtime.session import ConversationSession, Turn
from .runtime.streaming import StreamingKernel, StreamSession, TokenEvent

# --- Safety ---
from .safety.sanitizer import InputSanitizer, SanitizeResult
from .scoring.backends import (
    ScorerBackend,
    get_backend,
    list_backends,
    register_backend,
)
from .scoring.lite_scorer import LiteScorer
from .scoring.meta_classifier import DatasetTypeClassifier, MetaClassifier
from .scoring.nli import (
    NLIScorer,
    clear_model_cache,
    export_onnx,
    export_tensorrt,
    nli_available,
)
from .scoring.scorer import CoherenceScorer
from .scoring.sharded_nli import ShardedNLIScorer
from .scoring.verified_scorer import ClaimVerdict, VerificationResult, VerifiedScorer

# --- Training ---
from .training.finetune import FinetuneConfig, FinetuneResult, finetune_nli
from .training.finetune_benchmark import RegressionReport, benchmark_finetuned_model
from .training.finetune_validator import DataQualityReport, validate_finetune_data
from .training.tuner import TuneResult, tune
from .types import (
    ClaimAttribution,
    CoherenceScore,
    EvidenceChunk,
    HaltEvidence,
    ReviewResult,
    ScoringEvidence,
)

__all__ = [
    # Scoring
    "ClaimVerdict",
    "CoherenceScorer",
    "DatasetTypeClassifier",
    "LiteScorer",
    "MetaClassifier",
    "NLIScorer",
    "ScorerBackend",
    "ShardedNLIScorer",
    "VerificationResult",
    "VerifiedScorer",
    # Retrieval
    "ChromaBackend",
    "ColBERTBackend",
    "ElasticsearchBackend",
    "FAISSBackend",
    "GroundTruthStore",
    "HybridBackend",
    "InMemoryBackend",
    "PineconeBackend",
    "QdrantBackend",
    "RerankedBackend",
    "SentenceTransformerBackend",
    "VectorBackend",
    "VectorGroundTruthStore",
    "WeaviateBackend",
    # Runtime
    "AsyncStreamingKernel",
    "BatchProcessor",
    "BatchResult",
    "HaltMonitor",
    "ReviewQueue",
    "SafetyKernel",
    "StreamSession",
    "StreamingKernel",
    "TokenEvent",
    # Safety
    "InputSanitizer",
    "SanitizeResult",
    # Training
    "DataQualityReport",
    "FinetuneConfig",
    "FinetuneResult",
    "RegressionReport",
    "TuneResult",
    # Types
    "ClaimAttribution",
    "CoherenceScore",
    "ConversationSession",
    "EvidenceChunk",
    "HaltEvidence",
    "ReviewResult",
    "ScoringEvidence",
    "Turn",
    # Orchestration
    "CoherenceAgent",
    "DirectorConfig",
    "LLMGenerator",
    "MockGenerator",
    "ScoreCache",
    # Functions
    "benchmark_finetuned_model",
    "clear_model_cache",
    "export_onnx",
    "export_tensorrt",
    "finetune_nli",
    "tune",
    "get_backend",
    "get_vector_backend",
    "list_backends",
    "list_vector_backends",
    "nli_available",
    "register_backend",
    "register_vector_backend",
    "validate_finetune_data",
]

_MOVED_TO_ENTERPRISE = {
    "TenantRouter": ".tenant",
    "Policy": ".safety.policy",
    "Violation": ".safety.policy",
    "AuditLogger": ".safety.audit",
    "AuditEntry": ".safety.audit",
}


def __getattr__(name: str):
    if name in _MOVED_TO_ENTERPRISE:
        raise ImportError(
            f"{name} moved to director_ai.enterprise in v3.0. "
            f"Use: from director_ai.enterprise import {name}",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
