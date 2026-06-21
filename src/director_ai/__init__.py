# SPDX-License-Identifier: Apache-2.0
# Commercial license available
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

from typing import Any

__version__ = "3.16.0"

# Symbol → (module_path, attribute_name)
# Module paths are relative to director_ai package.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Core classes
    "AdaptiveThresholdArm": (".core", "AdaptiveThresholdArm"),
    "AdaptiveThresholdLearner": (".core", "AdaptiveThresholdLearner"),
    "AdaptiveThresholdRecommendation": (".core", "AdaptiveThresholdRecommendation"),
    "AdaptiveThresholdReport": (".core", "AdaptiveThresholdReport"),
    "AsyncStreamingKernel": (".core", "AsyncStreamingKernel"),
    "AttributionEdge": (".core", "AttributionEdge"),
    "AttributionNode": (".core", "AttributionNode"),
    "BFTConsensusResult": (".core", "BFTConsensusResult"),
    "BFTConsensusVote": (".core", "BFTConsensusVote"),
    "BatchProcessor": (".core", "BatchProcessor"),
    "BatchResult": (".core", "BatchResult"),
    "CausalAttributionGraph": (".core", "CausalAttributionGraph"),
    "ByzantineFaultTolerantConsensus": (".core", "ByzantineFaultTolerantConsensus"),
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
    "CounterfactualFactChange": (".core", "CounterfactualFactChange"),
    "CounterfactualHaltDiagnostic": (".core", "CounterfactualHaltDiagnostic"),
    "CoherenceAgent": (".core", "CoherenceAgent"),
    "CoherenceScore": (".core", "CoherenceScore"),
    "CoherenceScorer": (".core", "CoherenceScorer"),
    "ConflictAwareKnowledgeGuard": (".core", "ConflictAwareKnowledgeGuard"),
    "ColBERTBackend": (".core", "ColBERTBackend"),
    "ConversationSession": (".core", "ConversationSession"),
    "CrossDocumentConflict": (".core", "CrossDocumentConflict"),
    "CrossDocumentConsistencyMemory": (".core", "CrossDocumentConsistencyMemory"),
    "CrossDocumentConsistencyReport": (".core", "CrossDocumentConsistencyReport"),
    "DataQualityReport": (".core", "DataQualityReport"),
    "DatasetTypeClassifier": (".core", "DatasetTypeClassifier"),
    "DifferentialPrivacyScoreReleaser": (".core", "DifferentialPrivacyScoreReleaser"),
    "DirectorConfig": (".core", "DirectorConfig"),
    "ElasticsearchBackend": (".core", "ElasticsearchBackend"),
    "EvidenceChunk": (".core", "EvidenceChunk"),
    "FAISSBackend": (".core", "FAISSBackend"),
    "FinetuneConfig": (".core", "FinetuneConfig"),
    "FinetuneResult": (".core", "FinetuneResult"),
    "FederatedSafetySignalAggregator": (
        ".core",
        "FederatedSafetySignalAggregator",
    ),
    "FederatedSafetySignalRelease": (".core", "FederatedSafetySignalRelease"),
    "GroundTruthStore": (".core", "GroundTruthStore"),
    "HaltEvidence": (".core", "HaltEvidence"),
    "HaltMonitor": (".core", "HaltMonitor"),
    "HaltTraceAttribution": (".core", "HaltTraceAttribution"),
    "HybridBackend": (".core", "HybridBackend"),
    "HumanReviewCase": (".core", "HumanReviewCase"),
    "HumanReviewDecision": (".core", "HumanReviewDecision"),
    "HumanReviewQueue": (".core", "HumanReviewQueue"),
    "InMemoryBackend": (".core", "InMemoryBackend"),
    "InjectedClaim": (".core.types", "InjectedClaim"),
    "InjectionDetector": (".core.safety.injection", "InjectionDetector"),
    "InjectionResult": (".core.types", "InjectionResult"),
    "InputSanitizer": (".core", "InputSanitizer"),
    "KnowledgeConflict": (".core", "KnowledgeConflict"),
    "KnowledgeConflictCheck": (".core", "KnowledgeConflictCheck"),
    "KnowledgeFact": (".core", "KnowledgeFact"),
    "LabelledPolicySample": (".core", "LabelledPolicySample"),
    "LiteScorer": (".core", "LiteScorer"),
    "LLMGenerator": (".core", "LLMGenerator"),
    "MetaClassifier": (".core", "MetaClassifier"),
    "MockGenerator": (".core", "MockGenerator"),
    "NeuroSymbolicVerificationResult": (".core", "NeuroSymbolicVerificationResult"),
    "NeuroSymbolicVerifier": (".core", "NeuroSymbolicVerifier"),
    "NeuroSymbolicVerifierInput": (".core", "NeuroSymbolicVerifierInput"),
    "NLIScorer": (".core", "NLIScorer"),
    "PineconeBackend": (".core", "PineconeBackend"),
    "PolicyComparisonReport": (".core", "PolicyComparisonReport"),
    "PolicyEvaluationReport": (".core", "PolicyEvaluationReport"),
    "PolicyVariant": (".core", "PolicyVariant"),
    "PolicyVariantResult": (".core", "PolicyVariantResult"),
    "ProfileMetadata": (".core", "ProfileMetadata"),
    "PrivacyScoreRelease": (".core", "PrivacyScoreRelease"),
    "QdrantBackend": (".core", "QdrantBackend"),
    "RegressionReport": (".core", "RegressionReport"),
    "RerankedBackend": (".core", "RerankedBackend"),
    "ReviewQueue": (".core", "ReviewQueue"),
    "DIRECTOR_SAFETY_PROTOCOL_VERSION": (
        ".core",
        "DIRECTOR_SAFETY_PROTOCOL_VERSION",
    ),
    "DirectorSafetySignal": (".core", "DirectorSafetySignal"),
    "ReviewResult": (".core", "ReviewResult"),
    "SAFETY_EVENT_JSON_SCHEMA": (".core", "SAFETY_EVENT_JSON_SCHEMA"),
    "SafetyEvent": (".core", "SafetyEvent"),
    "SafetyKernel": (".core", "SafetyKernel"),
    "SanitizeResult": (".core", "SanitizeResult"),
    "ScoreCache": (".core", "ScoreCache"),
    "ScorerBackend": (".core", "ScorerBackend"),
    "ScoringEvidence": (".core", "ScoringEvidence"),
    "SentenceTransformerBackend": (".core", "SentenceTransformerBackend"),
    "ShardedNLIScorer": (".core", "ShardedNLIScorer"),
    "StreamGuard": (".lite", "StreamGuard"),
    "StreamingKernel": (".core", "StreamingKernel"),
    "StreamSession": (".core", "StreamSession"),
    "streaming_guard": (".lite", "streaming_guard"),
    "StoredDocument": (".core", "StoredDocument"),
    "StructuredRecoveryConfig": (".core", "StructuredRecoveryConfig"),
    "StructuredRecoveryResult": (".core", "StructuredRecoveryResult"),
    "ThresholdFeedback": (".core", "ThresholdFeedback"),
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
    "build_causal_attribution_graph": (".core", "build_causal_attribution_graph"),
    "compare_policy_variants": (".core", "compare_policy_variants"),
    "clear_model_cache": (".core", "clear_model_cache"),
    "evaluate_policy_variants": (".core", "evaluate_policy_variants"),
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
    "director_safety_signal_from_event": (
        ".core",
        "director_safety_signal_from_event",
    ),
    "new_director_safety_signal_id": (".core", "new_director_safety_signal_id"),
    "new_safety_event_id": (".core", "new_safety_event_id"),
    "validate_safety_event_payload": (".core", "validate_safety_event_payload"),
    "validate_director_safety_signal": (
        ".core",
        "validate_director_safety_signal",
    ),
    "utc_timestamp": (".core", "utc_timestamp"),
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
    "InferenceHookDecision": (
        ".integrations.inference_server_hooks",
        "InferenceHookDecision",
    ),
    "InferenceHookRequest": (
        ".integrations.inference_server_hooks",
        "InferenceHookRequest",
    ),
    "InferenceServerHook": (
        ".integrations.inference_server_hooks",
        "InferenceServerHook",
    ),
    "InferenceServerHookPolicy": (
        ".integrations.inference_server_hooks",
        "InferenceServerHookPolicy",
    ),
    "InterlockDecision": (".interlock", "InterlockDecision"),
    "InterlockKernel": (".interlock", "InterlockKernel"),
    "InterlockPolicy": (".interlock", "InterlockPolicy"),
    "build_inference_server_hook": (
        ".integrations.inference_server_hooks",
        "build_inference_server_hook",
    ),
    "get_score": (".integrations.sdk_guard", "get_score"),
    "guard": (".integrations.sdk_guard", "guard"),
    "score": (".integrations.sdk_guard", "score"),
    "VoiceGuard": (".integrations.voice", "VoiceGuard"),
    "VoiceToken": (".integrations.voice", "VoiceToken"),
    # Voice AI subpackage (async guard + TTS adapters + pipeline)
    "AsyncVoiceGuard": (".voice.guard", "AsyncVoiceGuard"),
    "TTSAdapter": (".voice.adapters", "TTSAdapter"),
    "ElevenLabsAdapter": (".voice.adapters", "ElevenLabsAdapter"),
    "OpenAITTSAdapter": (".voice.adapters", "OpenAITTSAdapter"),
    "DeepgramAdapter": (".voice.adapters", "DeepgramAdapter"),
    "voice_pipeline": (".voice.pipeline", "voice_pipeline"),
    "CoherenceCallbackHandler": (
        ".integrations.langchain_callback",
        "CoherenceCallbackHandler",
    ),
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
    # Enterprise moderation wrapper
    "ContentModerator": (".enterprise.moderation", "ContentModerator"),
    "ContentModerationResult": (
        ".enterprise.moderation",
        "ContentModerationResult",
    ),
    "ContentModerationFinding": (
        ".enterprise.moderation",
        "ContentModerationFinding",
    ),
    "ModerationAction": (".enterprise.moderation", "ModerationAction"),
    # Enterprise custom policy DSL
    "CustomRule": (".enterprise.rules_dsl", "CustomRule"),
    "CustomRuleset": (".enterprise.rules_dsl", "CustomRuleset"),
    "RulesDslError": (".enterprise.rules_dsl", "RulesDslError"),
    # Online calibration from production feedback
    "FeedbackStore": (".core.calibration.feedback_store", "FeedbackStore"),
    "OnlineCalibrator": (".core.calibration.online_calibrator", "OnlineCalibrator"),
    "CalibrationReport": (".core.calibration.online_calibrator", "CalibrationReport"),
    # EU AI Act compliance reporting
    "AuditLog": (".compliance.audit_log", "AuditLog"),
    "AuditEntry": (".compliance.audit_log", "AuditEntry"),
    "ComplianceReporter": (".compliance.reporter", "ComplianceReporter"),
    "Article15Report": (".compliance.reporter", "Article15Report"),
    "Article15TemplateContext": (
        ".compliance.reporter",
        "Article15TemplateContext",
    ),
    "ReadinessStatus": (".compliance.readiness", "ReadinessStatus"),
    "Soc2IsoControl": (".compliance.readiness", "Soc2IsoControl"),
    "Soc2IsoReadinessReport": (
        ".compliance.readiness",
        "Soc2IsoReadinessReport",
    ),
    "HipaaDeploymentObligation": (
        ".compliance.readiness",
        "HipaaDeploymentObligation",
    ),
    "HipaaDocumentationPacket": (
        ".compliance.readiness",
        "HipaaDocumentationPacket",
    ),
    "build_soc2_iso_readiness_report": (
        ".compliance.readiness",
        "build_soc2_iso_readiness_report",
    ),
    "build_hipaa_documentation_packet": (
        ".compliance.readiness",
        "build_hipaa_documentation_packet",
    ),
    "DriftDetector": (".compliance.drift_detector", "DriftDetector"),
    "DriftResult": (".compliance.drift_detector", "DriftResult"),
    # Numeric verification (Phase 5)
    "verify_numeric": (".core.verification.numeric_verifier", "verify_numeric"),
    # Arithmetic-equation verification
    "verify_arithmetic": (
        ".core.verification.math_consistency",
        "verify_arithmetic",
    ),
    # Informal-fallacy detection
    "detect_fallacies": (
        ".core.verification.fallacy_detector",
        "detect_fallacies",
    ),
    # Citation tracing
    "trace_citations": (
        ".core.verification.citation_tracer",
        "trace_citations",
    ),
    # Agentic loop monitoring (Phase 5)
    "LoopMonitor": (".agentic.loop_monitor", "LoopMonitor"),
    # Conformal prediction (Phase 5)
    "ConformalPredictor": (".core.calibration.conformal", "ConformalPredictor"),
    "ConformalRoutingDecision": (
        ".core.calibration.conformal",
        "ConformalRoutingDecision",
    ),
    "ConformalRoutingPolicy": (
        ".core.calibration.conformal",
        "ConformalRoutingPolicy",
    ),
    "PredictionInterval": (".core.calibration.conformal", "PredictionInterval"),
    # Cross-model consensus (Phase 5)
    "ConsensusScorer": (".core.scoring.consensus", "ConsensusScorer"),
    "CriticalConsensusProfile": (
        ".core.scoring.consensus",
        "CriticalConsensusProfile",
    ),
    "CrossVerifierConsensus": (".core.scoring.consensus", "CrossVerifierConsensus"),
    "ModelResponse": (".core.scoring.consensus", "ModelResponse"),
    # Adversarial robustness testing (Phase 5)
    "AdversarialTester": (".testing.adversarial_suite", "AdversarialTester"),
    # Temporal freshness (Phase 5)
    "score_temporal_freshness": (
        ".core.scoring.temporal_freshness",
        "score_temporal_freshness",
    ),
    "CitationStatusSignal": (
        ".core.scoring.temporal_freshness",
        "CitationStatusSignal",
    ),
    "CitationStatusVerdict": (
        ".core.scoring.temporal_freshness",
        "CitationStatusVerdict",
    ),
    # Reasoning chain verification (Phase 5)
    "verify_reasoning_chain": (
        ".core.verification.reasoning_verifier",
        "verify_reasoning_chain",
    ),
    # Feedback loop detection (Phase 5 — EU AI Act Art 15(4))
    "FeedbackLoopDetector": (
        ".compliance.feedback_loop_detector",
        "FeedbackLoopDetector",
    ),
    "NumericVerificationResult": (
        ".core.verification.numeric_verifier",
        "NumericVerificationResult",
    ),
    # Agentic swarm guardian (v3.14+)
    "AgentProfile": (".agentic.agent_profile", "AgentProfile"),
    "SwarmGuardian": (".agentic.swarm_guardian", "SwarmGuardian"),
    "HandoffScorer": (".agentic.handoff_scorer", "HandoffScorer"),
    "SwarmMetrics": (".agentic.swarm_metrics", "SwarmMetrics"),
    # Compliance cost analyser (v4.0+)
    "CostAnalyser": (".compliance.cost_analyser", "CostAnalyser"),
    # KB health diagnostics (v3.14+)
    "KBHealthCheck": (".core.retrieval.kb_health", "KBHealthCheck"),
    "KBHealthReport": (".core.retrieval.kb_health", "KBHealthReport"),
    # Advanced RAG backends (v3.14+)
    "ParentChildBackend": (
        ".core.retrieval.parent_child",
        "ParentChildBackend",
    ),
    "AdaptiveRouter": (
        ".core.retrieval.adaptive_router",
        "AdaptiveRouter",
    ),
    "HyDEBackend": (".core.retrieval.hyde", "HyDEBackend"),
    "QueryDecompositionBackend": (
        ".core.retrieval.query_decomposition",
        "QueryDecompositionBackend",
    ),
    "ContextualCompressionBackend": (
        ".core.retrieval.contextual_compression",
        "ContextualCompressionBackend",
    ),
    "MultiVectorBackend": (
        ".core.retrieval.multi_vector",
        "MultiVectorBackend",
    ),
}

__all__ = sorted(_LAZY_IMPORTS)

_MOVED_TO_ENTERPRISE = {
    "TenantRouter",
    "Policy",
    "Violation",
    "AuditLogger",
    "AuditEntry",
}


def __getattr__(name: str) -> Any:
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
