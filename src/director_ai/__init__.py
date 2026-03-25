# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Package Initialisation (Lazy Imports)

"""Director-AI: Real-time LLM hallucination guardrail.

All public symbols are available via ``from director_ai import X``.
Imports are deferred until first access to keep ``import director_ai``
under 0.1 seconds.

::

    from director_ai import guard, score, CoherenceScorer
"""

__version__ = "3.10.0"

# Symbol → (module_path, attribute_name)
# Module paths are relative to director_ai package.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core classes
    "AsyncStreamingKernel": (".core", "AsyncStreamingKernel"),
    "BatchProcessor": (".core", "BatchProcessor"),
    "BatchResult": (".core", "BatchResult"),
    "ChromaBackend": (".core", "ChromaBackend"),
    "ClaimAttribution": (".core", "ClaimAttribution"),
    "ClaimVerdict": (".core", "ClaimVerdict"),
    "compute_meta_confidence": (
        ".core.scoring.meta_confidence",
        "compute_meta_confidence",
    ),
    "ContradictionReport": (
        ".core.runtime.contradiction_tracker",
        "ContradictionReport",
    ),
    "ContradictionTracker": (
        ".core.runtime.contradiction_tracker",
        "ContradictionTracker",
    ),
    "CoherenceAgent": (".core", "CoherenceAgent"),
    "CoherenceScore": (".core", "CoherenceScore"),
    "CoherenceScorer": (".core", "CoherenceScorer"),
    "ColBERTBackend": (".core", "ColBERTBackend"),
    "ConversationSession": (".core", "ConversationSession"),
    "DataQualityReport": (".core", "DataQualityReport"),
    "DatasetTypeClassifier": (".core", "DatasetTypeClassifier"),
    "DirectorConfig": (".core", "DirectorConfig"),
    "ElasticsearchBackend": (".core", "ElasticsearchBackend"),
    "EvidenceChunk": (".core", "EvidenceChunk"),
    "FAISSBackend": (".core", "FAISSBackend"),
    "FinetuneConfig": (".core", "FinetuneConfig"),
    "FinetuneResult": (".core", "FinetuneResult"),
    "GroundTruthStore": (".core", "GroundTruthStore"),
    "HaltEvidence": (".core", "HaltEvidence"),
    "HaltMonitor": (".core", "HaltMonitor"),
    "HybridBackend": (".core", "HybridBackend"),
    "InMemoryBackend": (".core", "InMemoryBackend"),
    "InputSanitizer": (".core", "InputSanitizer"),
    "LiteScorer": (".core", "LiteScorer"),
    "LLMGenerator": (".core", "LLMGenerator"),
    "MetaClassifier": (".core", "MetaClassifier"),
    "MockGenerator": (".core", "MockGenerator"),
    "NLIScorer": (".core", "NLIScorer"),
    "PineconeBackend": (".core", "PineconeBackend"),
    "QdrantBackend": (".core", "QdrantBackend"),
    "RegressionReport": (".core", "RegressionReport"),
    "RerankedBackend": (".core", "RerankedBackend"),
    "ReviewQueue": (".core", "ReviewQueue"),
    "ReviewResult": (".core", "ReviewResult"),
    "SafetyKernel": (".core", "SafetyKernel"),
    "SanitizeResult": (".core", "SanitizeResult"),
    "ScoreCache": (".core", "ScoreCache"),
    "ScorerBackend": (".core", "ScorerBackend"),
    "ScoringEvidence": (".core", "ScoringEvidence"),
    "SentenceTransformerBackend": (".core", "SentenceTransformerBackend"),
    "ShardedNLIScorer": (".core", "ShardedNLIScorer"),
    "StreamingKernel": (".core", "StreamingKernel"),
    "StreamSession": (".core", "StreamSession"),
    "TokenEvent": (".core", "TokenEvent"),
    "TuneResult": (".core", "TuneResult"),
    "Turn": (".core", "Turn"),
    "VectorBackend": (".core", "VectorBackend"),
    "VectorGroundTruthStore": (".core", "VectorGroundTruthStore"),
    "VerificationResult": (".core", "VerificationResult"),
    "VerifiedScorer": (".core", "VerifiedScorer"),
    "WeaviateBackend": (".core", "WeaviateBackend"),
    # Core functions
    "benchmark_finetuned_model": (".core", "benchmark_finetuned_model"),
    "clear_model_cache": (".core", "clear_model_cache"),
    "export_onnx": (".core", "export_onnx"),
    "export_tensorrt": (".core", "export_tensorrt"),
    "finetune_nli": (".core", "finetune_nli"),
    "get_backend": (".core", "get_backend"),
    "get_vector_backend": (".core", "get_vector_backend"),
    "list_backends": (".core", "list_backends"),
    "list_vector_backends": (".core", "list_vector_backends"),
    "nli_available": (".core", "nli_available"),
    "register_backend": (".core", "register_backend"),
    "register_vector_backend": (".core", "register_vector_backend"),
    "tune": (".core", "tune"),
    "validate_finetune_data": (".core", "validate_finetune_data"),
    # Exceptions
    "CoherenceError": (".core.exceptions", "CoherenceError"),
    "DependencyError": (".core.exceptions", "DependencyError"),
    "DirectorAIError": (".core.exceptions", "DirectorAIError"),
    "GeneratorError": (".core.exceptions", "GeneratorError"),
    "HallucinationError": (".core.exceptions", "HallucinationError"),
    "KernelHaltError": (".core.exceptions", "KernelHaltError"),
    "NumericalError": (".core.exceptions", "NumericalError"),
    "PhysicsError": (".core.exceptions", "PhysicsError"),
    "ValidationError": (".core.exceptions", "ValidationError"),
    # Integrations
    "get_score": (".integrations.sdk_guard", "get_score"),
    "guard": (".integrations.sdk_guard", "guard"),
    "score": (".integrations.sdk_guard", "score"),
    "VoiceGuard": (".integrations.voice", "VoiceGuard"),
    "VoiceToken": (".integrations.voice", "VoiceToken"),
    # Server factories
    "create_app": (".server", "create_app"),
    "create_grpc_server": (".grpc_server", "create_grpc_server"),
    "create_knowledge_router": (".knowledge_api", "create_knowledge_router"),
    # Structured output verification (stdlib only, no torch)
    "verify_json": (".core.verification.json_verifier", "verify_json"),
    "verify_tool_call": (".core.verification.tool_call_verifier", "verify_tool_call"),
    "verify_code": (".core.verification.code_verifier", "verify_code"),
    "StructuredVerificationResult": (
        ".core.verification.types",
        "StructuredVerificationResult",
    ),
    "ToolCallResult": (".core.verification.types", "ToolCallResult"),
    "CodeCheckResult": (".core.verification.types", "CodeCheckResult"),
    "FieldVerdict": (".core.verification.types", "FieldVerdict"),
    # Online calibration from production feedback
    "FeedbackStore": (".core.calibration.feedback_store", "FeedbackStore"),
    "OnlineCalibrator": (".core.calibration.online_calibrator", "OnlineCalibrator"),
    "CalibrationReport": (".core.calibration.online_calibrator", "CalibrationReport"),
    # EU AI Act compliance reporting
    "AuditLog": (".compliance.audit_log", "AuditLog"),
    "AuditEntry": (".compliance.audit_log", "AuditEntry"),
    "ComplianceReporter": (".compliance.reporter", "ComplianceReporter"),
    "Article15Report": (".compliance.reporter", "Article15Report"),
    "DriftDetector": (".compliance.drift_detector", "DriftDetector"),
    "DriftResult": (".compliance.drift_detector", "DriftResult"),
}

__all__ = sorted(_LAZY_IMPORTS)

_MOVED_TO_ENTERPRISE = {
    "TenantRouter",
    "Policy",
    "Violation",
    "AuditLogger",
    "AuditEntry",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        # Cache in module globals for subsequent access without __getattr__
        globals()[name] = val
        return val
    if name in _MOVED_TO_ENTERPRISE:
        raise ImportError(
            f"{name} moved to director_ai.enterprise in v3.0. "
            f"Use: from director_ai.enterprise import {name}",
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
