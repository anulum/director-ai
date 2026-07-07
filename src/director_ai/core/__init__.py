# SPDX-License-Identifier: Apache-2.0
# Commercial license available
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

from typing import NoReturn

# --- Scoring ---
# --- Core-level modules (not moved) ---
from .actor import LLMGenerator, MockGenerator
from .agent import CoherenceAgent
from .attribution import (
    AttributionEdge,
    AttributionNode,
    CausalAttributionGraph,
    build_causal_attribution_graph,
)
from .cache import ScoreCache
from .calibration.adaptive_conformal import AdaptiveConformalPredictor

# --- Calibration ---
from .calibration.adaptive_threshold import (
    AdaptiveThresholdArm,
    AdaptiveThresholdLearner,
    AdaptiveThresholdRecommendation,
    AdaptiveThresholdReport,
    ThresholdFeedback,
)
from .calibration.conformal import (
    ConformalPredictor,
    ConformalRoutingDecision,
    ConformalRoutingPolicy,
    PredictionInterval,
)
from .calibration.feedback_store import FeedbackStore
from .calibration.grader_validation import (
    GraderCase,
    GraderReport,
    GraderValidationError,
    StatusMetrics,
    assert_grader_admissible,
    validate_grader,
)
from .calibration.miscoverage import (
    CoverageHealthReport,
    MiscoverageMonitor,
    SegmentCoverageHealth,
)
from .calibration.online_calibrator import CalibrationReport, OnlineCalibrator
from .calibration.recall_correctness import (
    RecallOutcome,
    correctness_from_verdict,
    recall_outcome,
)
from .calibration.recall_correctness_client import (
    RemanentiaCorrectnessClient,
    RemanentiaCorrectnessError,
)
from .calibration.recall_ledger import (
    ColdStartSummary,
    RecallQuery,
    cold_start_from_ledger,
    default_ledger_path,
    read_recall_ledger,
)
from .calibration.segmented_threshold import (
    SegmentedThresholdLearner,
    SegmentRecommendation,
)
from .calibration.tuner import TuneResult, tune
from .config import DirectorConfig, ProfileMetadata
from .edge import (
    EdgeRuntimeCheck,
    EdgeRuntimeReadiness,
    build_edge_runtime_readiness,
)

# --- Evaluation ---
from .evaluation import (
    LabelledPolicySample,
    PolicyComparisonReport,
    PolicyEvaluationReport,
    PolicyVariant,
    PolicyVariantResult,
    compare_policy_variants,
    evaluate_policy_variants,
)
from .federated_privacy import (
    DifferentialPrivacyScoreReleaser,
    FederatedSafetySignalAggregator,
    FederatedSafetySignalRelease,
    PrivacyScoreRelease,
)
from .memory import (
    CrossDocumentConflict,
    CrossDocumentConsistencyMemory,
    CrossDocumentConsistencyReport,
    StoredDocument,
)

# --- Retrieval ---
from .retrieval.conflict_guard import (
    ConflictAwareKnowledgeGuard,
    KnowledgeConflict,
    KnowledgeConflictCheck,
    KnowledgeFact,
)
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

# --- Runtime (Phase 5) ---
from .runtime.contradiction_tracker import ContradictionReport, ContradictionTracker
from .runtime.correction import (
    CorrectionLoop,
    CorrectionProposal,
    GroundedCorrectionDraft,
    HaltCorrectionContext,
)
from .runtime.human_review import HumanReviewCase, HumanReviewDecision, HumanReviewQueue
from .runtime.kernel import HaltMonitor, SafetyKernel
from .runtime.review_queue import ReviewQueue
from .runtime.session import ConversationSession, Turn
from .runtime.streaming import StreamingKernel, StreamSession, TokenEvent
from .runtime.structured_recovery import (
    StructuredRecoveryConfig,
    StructuredRecoveryResult,
)

# --- Safety ---
from .safety.injection import InjectionDetector
from .safety.sanitizer import InputSanitizer, SanitizeResult
from .safety_event import (
    SAFETY_EVENT_JSON_SCHEMA,
    SafetyEvent,
    new_safety_event_id,
    utc_timestamp,
    validate_safety_event_payload,
)
from .safety_protocol import (
    DIRECTOR_SAFETY_PROTOCOL_VERSION,
    DirectorSafetySignal,
    director_safety_signal_from_event,
    new_director_safety_signal_id,
    validate_director_safety_signal,
)
from .scoring.backends import (
    ScorerBackend,
    get_backend,
    list_backends,
    register_backend,
)

# --- Scoring (Phase 5) ---
from .scoring.consensus import (
    BFTConsensusResult,
    BFTConsensusVote,
    ByzantineFaultTolerantConsensus,
    ConsensusScorer,
    CriticalConsensusProfile,
    CrossVerifierConsensus,
)
from .scoring.lite_scorer import LiteScorer
from .scoring.meta_classifier import DatasetTypeClassifier, MetaClassifier
from .scoring.meta_confidence import compute_meta_confidence
from .scoring.nli import (
    NLIScorer,
    clear_model_cache,
    export_onnx,
    export_tensorrt,
    nli_available,
)
from .scoring.scorer import CoherenceScorer
from .scoring.sharded_nli import ShardedNLIScorer
from .scoring.temporal_freshness import (
    CitationStatusSignal,
    CitationStatusVerdict,
    score_temporal_freshness,
)
from .scoring.verified_scorer import ClaimVerdict, VerificationResult, VerifiedScorer

# --- Training ---
from .training.finetune import FinetuneConfig, FinetuneResult, finetune_nli
from .training.finetune_benchmark import (
    ModelBenchmarkReport,
    ModelBenchmarkResult,
    RegressionReport,
    benchmark_finetuned_model,
    benchmark_model_candidates,
)
from .training.finetune_validator import DataQualityReport, validate_finetune_data
from .training.model_registry import (
    TrainingModelProfile,
    finetune_model_registry_to_dict,
    list_finetune_model_profiles,
    resolve_finetune_model,
)
from .types import (
    ClaimAttribution,
    CoherenceScore,
    CounterfactualFactChange,
    CounterfactualHaltDiagnostic,
    EvidenceChunk,
    HaltEvidence,
    HaltTraceAttribution,
    InjectedClaim,
    InjectionResult,
    ReviewResult,
    ScoringEvidence,
)

# --- Verification ---
from .verification.citation_tracer import (
    ClaimCitation,
    TraceResult,
    trace_citations,
)
from .verification.code_verifier import CodeCheckResult, verify_code
from .verification.fallacy_detector import (
    FallacyMatch,
    FallacyResult,
    detect_fallacies,
)
from .verification.json_verifier import StructuredVerificationResult, verify_json
from .verification.math_consistency import (
    MathConsistencyResult,
    verify_arithmetic,
)
from .verification.neuro_symbolic import (
    NeuroSymbolicVerificationResult,
    NeuroSymbolicVerifier,
    NeuroSymbolicVerifierInput,
)
from .verification.numeric_verifier import NumericVerificationResult, verify_numeric
from .verification.reasoning_verifier import verify_reasoning_chain
from .verification.tool_call_verifier import ToolCallResult, verify_tool_call
from .verification.types import FieldVerdict

__all__ = [
    # Scoring
    "ClaimVerdict",
    "CoherenceScorer",
    "ConflictAwareKnowledgeGuard",
    "DatasetTypeClassifier",
    "EdgeRuntimeCheck",
    "EdgeRuntimeReadiness",
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
    "KnowledgeConflict",
    "KnowledgeConflictCheck",
    "KnowledgeFact",
    "LabelledPolicySample",
    "PineconeBackend",
    "PolicyComparisonReport",
    "PolicyEvaluationReport",
    "PolicyVariant",
    "PolicyVariantResult",
    "QdrantBackend",
    "RerankedBackend",
    "SentenceTransformerBackend",
    "VectorBackend",
    "VectorGroundTruthStore",
    "WeaviateBackend",
    # Runtime
    "AdaptiveThresholdArm",
    "AdaptiveThresholdLearner",
    "AdaptiveThresholdRecommendation",
    "AdaptiveThresholdReport",
    "SegmentRecommendation",
    "SegmentedThresholdLearner",
    "AsyncStreamingKernel",
    "AttributionEdge",
    "AttributionNode",
    "BFTConsensusResult",
    "BFTConsensusVote",
    "BatchProcessor",
    "BatchResult",
    "CausalAttributionGraph",
    "ByzantineFaultTolerantConsensus",
    "CorrectionLoop",
    "CorrectionProposal",
    "CrossDocumentConflict",
    "CrossDocumentConsistencyMemory",
    "CrossDocumentConsistencyReport",
    "GroundedCorrectionDraft",
    "HaltCorrectionContext",
    "HaltMonitor",
    "HumanReviewCase",
    "HumanReviewDecision",
    "HumanReviewQueue",
    "ReviewQueue",
    "DirectorSafetySignal",
    "SAFETY_EVENT_JSON_SCHEMA",
    "SafetyEvent",
    "SafetyKernel",
    "StreamSession",
    "StreamingKernel",
    "StoredDocument",
    "StructuredRecoveryConfig",
    "StructuredRecoveryResult",
    "ThresholdFeedback",
    "TokenEvent",
    # Safety
    "InjectionDetector",
    "InputSanitizer",
    "SanitizeResult",
    # Training
    "DataQualityReport",
    "DifferentialPrivacyScoreReleaser",
    "FinetuneConfig",
    "FinetuneResult",
    "FederatedSafetySignalAggregator",
    "FederatedSafetySignalRelease",
    "ModelBenchmarkReport",
    "ModelBenchmarkResult",
    "NeuroSymbolicVerificationResult",
    "NeuroSymbolicVerifier",
    "NeuroSymbolicVerifierInput",
    "RegressionReport",
    "TrainingModelProfile",
    "TuneResult",
    # Types
    "ClaimAttribution",
    "CoherenceScore",
    "CounterfactualFactChange",
    "CounterfactualHaltDiagnostic",
    "ConversationSession",
    "EvidenceChunk",
    "HaltEvidence",
    "HaltTraceAttribution",
    "InjectedClaim",
    "InjectionResult",
    "ReviewResult",
    "ScoringEvidence",
    "Turn",
    # Orchestration
    "CoherenceAgent",
    "DirectorConfig",
    "LLMGenerator",
    "MockGenerator",
    "ProfileMetadata",
    "PrivacyScoreRelease",
    "ScoreCache",
    # Functions
    "benchmark_finetuned_model",
    "benchmark_model_candidates",
    "build_causal_attribution_graph",
    "compare_policy_variants",
    "clear_model_cache",
    "export_onnx",
    "export_tensorrt",
    "evaluate_policy_variants",
    "finetune_nli",
    "build_edge_runtime_readiness",
    "finetune_model_registry_to_dict",
    "tune",
    "get_backend",
    "get_vector_backend",
    "list_backends",
    "list_finetune_model_profiles",
    "list_vector_backends",
    "nli_available",
    "register_backend",
    "register_vector_backend",
    "resolve_finetune_model",
    "validate_finetune_data",
    "new_safety_event_id",
    "validate_safety_event_payload",
    "director_safety_signal_from_event",
    "new_director_safety_signal_id",
    "validate_director_safety_signal",
    "utc_timestamp",
    "DIRECTOR_SAFETY_PROTOCOL_VERSION",
    # Calibration
    "AdaptiveConformalPredictor",
    "CalibrationReport",
    "ConformalPredictor",
    "ConformalRoutingDecision",
    "ConformalRoutingPolicy",
    "CoverageHealthReport",
    "FeedbackStore",
    "GraderCase",
    "GraderReport",
    "GraderValidationError",
    "MiscoverageMonitor",
    "ColdStartSummary",
    "OnlineCalibrator",
    "PredictionInterval",
    "RecallOutcome",
    "RecallQuery",
    "RemanentiaCorrectnessClient",
    "RemanentiaCorrectnessError",
    "SegmentCoverageHealth",
    "StatusMetrics",
    "assert_grader_admissible",
    "cold_start_from_ledger",
    "correctness_from_verdict",
    "default_ledger_path",
    "read_recall_ledger",
    "recall_outcome",
    "validate_grader",
    # Verification (Phase 5)
    "ClaimCitation",
    "CodeCheckResult",
    "FallacyMatch",
    "FallacyResult",
    "FieldVerdict",
    "MathConsistencyResult",
    "TraceResult",
    "NumericVerificationResult",
    "StructuredVerificationResult",
    "ToolCallResult",
    "detect_fallacies",
    "trace_citations",
    "verify_arithmetic",
    "verify_code",
    "verify_json",
    "verify_numeric",
    "verify_reasoning_chain",
    "verify_tool_call",
    # Scoring (Phase 5)
    "ConsensusScorer",
    "CriticalConsensusProfile",
    "CrossVerifierConsensus",
    "CitationStatusSignal",
    "CitationStatusVerdict",
    "compute_meta_confidence",
    "score_temporal_freshness",
    # Runtime (Phase 5)
    "ContradictionReport",
    "ContradictionTracker",
]

_MOVED_TO_ENTERPRISE = {
    "TenantRouter": ".tenant",
    "Policy": ".safety.policy",
    "Violation": ".safety.policy",
    "AuditLogger": ".safety.audit",
    "AuditEntry": ".safety.audit",
}


def __getattr__(name: str) -> NoReturn:
    if name in _MOVED_TO_ENTERPRISE:
        raise ImportError(
            f"{name} moved to director_ai.enterprise in v3.0. "
            f"Use: from director_ai.enterprise import {name}",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
