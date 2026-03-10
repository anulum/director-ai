# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Core Package (Coherence Engine)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Coherence Engine — consumer-ready AI output verification.

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
from .kernel import SafetyKernel
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
    "ClaimAttribution",
    "CoherenceScore",
    "EvidenceChunk",
    "HaltEvidence",
    "ScoringEvidence",
    "ReviewResult",
    "CoherenceScorer",
    "SafetyKernel",
    "MockGenerator",
    "LLMGenerator",
    "GroundTruthStore",
    "CoherenceAgent",
    "NLIScorer",
    "nli_available",
    "export_onnx",
    "VectorGroundTruthStore",
    "VectorBackend",
    "InMemoryBackend",
    "ChromaBackend",
    "SentenceTransformerBackend",
    "ScoreCache",
    "StreamingKernel",
    "StreamSession",
    "TokenEvent",
    "AsyncStreamingKernel",
    "InputSanitizer",
    "SanitizeResult",
    "RerankedBackend",
    "ConversationSession",
    "Turn",
    "LiteScorer",
    "ShardedNLIScorer",
    "ScorerBackend",
    "register_backend",
    "get_backend",
    "list_backends",
    "register_vector_backend",
    "get_vector_backend",
    "list_vector_backends",
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
            f"Use: from director_ai.enterprise import {name}"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
