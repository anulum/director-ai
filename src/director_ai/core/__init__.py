# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Core Package (Coherence Engine)

"""Coherence Engine — consumer-ready AI output verification.

Subpackages::

    core.scoring    — NLI, heuristic, and verified scorers
    core.retrieval  — knowledge stores, vector backends, chunking
    core.runtime    — streaming kernels, sessions, batching
    core.safety     — sanitizer, policy, audit
    core.training   — fine-tuning, threshold tuning, benchmarking

All public symbols are re-exported here for backward compatibility::

    from director_ai.core import CoherenceScorer, HaltMonitor, ...
"""

# --- Scoring ---
from .scoring.backends import ScorerBackend, get_backend, list_backends, register_backend
from .scoring.lite_scorer import LiteScorer
from .scoring.nli import NLIScorer, export_onnx, nli_available
from .scoring.scorer import CoherenceScorer
from .scoring.sharded_nli import ShardedNLIScorer

# --- Retrieval ---
from .retrieval.knowledge import GroundTruthStore
from .retrieval.vector_store import (
    ChromaBackend,
    InMemoryBackend,
    RerankedBackend,
    SentenceTransformerBackend,
    VectorBackend,
    VectorGroundTruthStore,
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

# --- Training ---
from .training.finetune import FinetuneConfig, FinetuneResult, finetune_nli
from .training.finetune_benchmark import RegressionReport, benchmark_finetuned_model
from .training.finetune_validator import DataQualityReport, validate_finetune_data

# --- Core-level modules (not moved) ---
from .actor import LLMGenerator, MockGenerator
from .agent import CoherenceAgent
from .cache import ScoreCache
from .config import DirectorConfig
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
    "CoherenceScorer",
    "LiteScorer",
    "NLIScorer",
    "ScorerBackend",
    "ShardedNLIScorer",
    # Retrieval
    "ChromaBackend",
    "GroundTruthStore",
    "InMemoryBackend",
    "RerankedBackend",
    "SentenceTransformerBackend",
    "VectorBackend",
    "VectorGroundTruthStore",
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
    "export_onnx",
    "finetune_nli",
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
