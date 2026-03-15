# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Core Package (Coherence Engine)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Coherence Engine — consumer-ready AI output verification.

Quick start::

    from director_ai.core import CoherenceAgent

    agent = CoherenceAgent()
    result = agent.process("What color is the sky?")
    print(result.output, result.coherence)
"""

from .actor import LLMGenerator, MockGenerator
from .agent import CoherenceAgent
from .async_streaming import AsyncStreamingKernel
from .backends import ScorerBackend, get_backend, list_backends, register_backend
from .cache import ScoreCache
from .config import DirectorConfig
from .finetune import FinetuneConfig, FinetuneResult, finetune_nli
from .finetune_benchmark import RegressionReport, benchmark_finetuned_model
from .finetune_validator import DataQualityReport, validate_finetune_data
from .kernel import HaltMonitor, SafetyKernel
from .knowledge import GroundTruthStore
from .lite_scorer import LiteScorer
from .nli import NLIScorer, export_onnx, nli_available
from .sanitizer import InputSanitizer, SanitizeResult
from .scorer import CoherenceScorer
from .session import ConversationSession, Turn
from .sharded_nli import ShardedNLIScorer
from .streaming import StreamingKernel, StreamSession, TokenEvent
from .types import (
    ClaimAttribution,
    CoherenceScore,
    EvidenceChunk,
    HaltEvidence,
    ReviewResult,
    ScoringEvidence,
)
from .vector_store import (
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

__all__ = [
    "AsyncStreamingKernel",
    "ChromaBackend",
    "ClaimAttribution",
    "CoherenceAgent",
    "CoherenceScore",
    "CoherenceScorer",
    "ConversationSession",
    "DataQualityReport",
    "DirectorConfig",
    "EvidenceChunk",
    "FinetuneConfig",
    "FinetuneResult",
    "GroundTruthStore",
    "HaltEvidence",
    "InMemoryBackend",
    "InputSanitizer",
    "LLMGenerator",
    "LiteScorer",
    "MockGenerator",
    "NLIScorer",
    "RegressionReport",
    "RerankedBackend",
    "ReviewResult",
    "HaltMonitor",
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
    "VectorBackend",
    "VectorGroundTruthStore",
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
    "Policy": ".policy",
    "Violation": ".policy",
    "AuditLogger": ".audit",
    "AuditEntry": ".audit",
}


def __getattr__(name: str):
    if name in _MOVED_TO_ENTERPRISE:
        raise ImportError(
            f"{name} moved to director_ai.enterprise in v3.0. "
            f"Use: from director_ai.enterprise import {name}",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
