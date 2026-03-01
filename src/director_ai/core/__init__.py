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
from .audit import AuditEntry, AuditLogger
from .cache import ScoreCache
from .kernel import SafetyKernel
from .knowledge import GroundTruthStore
from .nli import NLIScorer, export_onnx, nli_available
from .policy import Policy, Violation
from .sanitizer import InputSanitizer, SanitizeResult
from .scorer import CoherenceScorer
from .streaming import StreamingKernel, StreamSession, TokenEvent
from .tenant import TenantRouter
from .types import (
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
)

__all__ = [
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
    # v0.4.0 additions
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
    "Policy",
    "Violation",
    "AuditLogger",
    "AuditEntry",
    "TenantRouter",
    "InputSanitizer",
    "SanitizeResult",
    "RerankedBackend",
]
