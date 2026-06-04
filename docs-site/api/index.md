# API Reference

Primary reference for the supported public classes, functions, dataclasses, and
operator-facing APIs in Director-AI. Internal helpers, generated protobuf
stubs, and compatibility shims are intentionally excluded unless they define a
supported integration boundary.

## Pick The Right API First

| Need | API | Start |
|---|---|---|
| Protect an existing SDK client | `guard()` | [guard / score / get_score](guard.md) |
| Score one prompt and answer | `score()` | [guard / score / get_score](guard.md#score) |
| Halt streamed tokens | `StreamingKernel` | [StreamingKernel](streaming.md) |
| Build an agent workflow | `CoherenceAgent` | [CoherenceAgent](agent.md) |
| Process evaluation batches | `BatchProcessor` | [BatchProcessor](batch.md) |
| Serve over HTTP | REST server | [REST Server](server.md) |
| Serve over gRPC | CoherenceScoring service | [gRPC Server](grpc.md) |
| Ground answers in documents | `DocumentIngestionPipeline` and `VectorGroundTruthStore` | [Ingestion](ingestion.md) |
| Package customer evidence | Customer Model Factory | [Customer Model Factory](customer-model-factory.md) |

For the product and pilot context around these APIs, see
[Product Overview](../guide/product-overview.md) and
[Evaluation Onboarding](../guide/onboarding.md).

## Quick Navigation

### Entry Points

| Symbol | Module | Purpose |
|--------|--------|---------|
| [`guard()`](guard.md) | `director_ai` | Wrap an LLM SDK client with coherence scoring |
| [`score()`](guard.md#score) | `director_ai` | Score a single prompt/response pair |
| [`get_score()`](guard.md#get_score) | `director_ai` | Retrieve last score from `on_fail="metadata"` |
| [`VoiceGuard`](../guide/voice-ai.md) | `director_ai.integrations.voice` | Real-time token filter for voice AI / TTS pipelines |

### Core Classes

| Class | Module | Purpose |
|-------|--------|---------|
| [`CoherenceScorer`](scorer.md) | `director_ai.core.scoring.scorer` | Dual-entropy coherence scoring engine |
| [`StreamingKernel`](streaming.md) | `director_ai.core.runtime.streaming` | Token-level streaming halt |
| [`AsyncStreamingKernel`](streaming.md#async) | `director_ai.core.runtime.async_streaming` | Async variant of StreamingKernel |
| [`HumanReviewQueue`](human-review.md) | `director_ai.core.runtime.human_review` | Durable reviewer approval, retry, and release gate |
| [`NoGoPolicy`](guard-control.md) | `director_ai.core.guard_control` | Calibrated no-go policy with default irreversibility forecasting |
| [`evaluate_policy_variants()`](policy-evaluation.md) | `director_ai.core.evaluation.policy` | Controlled profile and threshold comparison on labelled data |
| [`build_causal_attribution_graph()`](causal-attribution.md) | `director_ai.core.attribution.causal_graph` | Evidence, claim, halt-trace, and counterfactual attribution DAGs |
| [`AdaptiveThresholdLearner`](adaptive-threshold.md) | `director_ai.core.calibration.adaptive_threshold` | Human-gated Thompson-sampling threshold recommendations |
| [`DefenseUpdatePipeline`](defence-update-pipeline.md) | `director_ai.core.defense_genome` | Reviewed defence promotion across feedback, adversarial mining, and registry hot-swap |
| [`AutoRedteamDefenceLoop`](defence-update-pipeline.md#auto-redteam-defence-loop) | `director_ai.core.defense_genome` | Repeated adversarial-mining cycles with detection-uplift gates before promotion |
| [`build_edge_runtime_readiness()`](edge-runtime-readiness.md) | `director_ai.core.edge` | Edge/mobile readiness profile for WASM, Rust, ONNX, and smoke evidence |
| [`CrossDocumentConsistencyMemory`](cross-document-memory.md) | `director_ai.core.memory.consistency` | Tenant-scoped long-term consistency checks with retention and delete controls |
| [`DifferentialPrivacyScoreReleaser`](private-score-release.md) | `director_ai.core.federated_privacy.score_release` | Laplace-noised score disclosure with privacy accounting |
| [`FederatedSafetySignalAggregator`](federated-safety-signals.md) | `director_ai.core.federated_privacy.signal_sharing` | Anonymous DP aggregate sharing for tenant-safe guard signals |
| [`NeuroSymbolicVerifier`](neuro-symbolic-verifier.md) | `director_ai.core.verification.neuro_symbolic` | Neural score fusion with numeric and formal symbolic checks |
| [`ByzantineFaultTolerantConsensus`](byzantine-consensus.md) | `director_ai.core.scoring.consensus` | PBFT-style quorum over independent verifier votes |
| [`InferenceServerHook`](../guide/streaming.md#pre-sampling-inference-server-hooks) | `director_ai.integrations.inference_server_hooks` | Server-neutral pre-sampling hook for vLLM, TGI, and llama.cpp |
| [`createDirectorAiMiddleware`](../integrations/vercel-ai.md) | `@director-ai/vercel-ai` | Vercel AI SDK `LanguageModelV3Middleware` bridge to `/v1/review` |
| [`AutoGenReplyGuard`](../integrations/autogen.md) | `director_ai.integrations.autogen_swarm` | AutoGen `register_reply()` hook for guarded group-chat messages |
| [`build_guardrails_validator`](../integrations/guardrails-ai.md) | `director_ai.integrations.guardrails_ai` | Guardrails AI custom validator for coherence checks |
| [`MetaGuard`](meta-guard.md) | `director_ai.core.meta_guard` | Recursive guard drift monitor with production evasion gates |
| [`SafetyEvent`](safety-event-schema.md) | `director_ai.core.safety_event` | Tenant-safe telemetry schema and validator for guard decisions |
| [`CoherenceAgent`](agent.md) | `director_ai.core.agent` | Orchestrator: generator + scorer + kernel |
| [`BatchProcessor`](batch.md) | `director_ai.core.runtime.batch` | Concurrent batch scoring |

### Knowledge & Retrieval

| Class | Module | Purpose |
|-------|--------|---------|
| [`GroundTruthStore`](guard.md) | `director_ai.core.retrieval.knowledge` | Key-value fact store (prototype) |
| [`VectorGroundTruthStore`](vector-store.md) | `director_ai.core.retrieval.vector_store` | Semantic vector store with pluggable backends |
| [`VectorBackend`](vector-store.md#vectorbackend) | `director_ai.core.retrieval.vector_store` | Abstract backend protocol |
| [`DocumentIngestionPipeline`](ingestion.md) | `director_ai.core.ingestion` | Parse, chunk, update, and delete documents for vector grounding |

### Configuration

| Class | Module | Purpose |
|-------|--------|---------|
| [`DirectorConfig`](config.md) | `director_ai.core.config` | Env var / YAML / profile configuration |

### Data Types

| Class | Module | Purpose |
|-------|--------|---------|
| [`CoherenceScore`](types.md) | `director_ai.core.types` | Score result with H_logical, H_factual, evidence, task type, confidence |
| [`ReviewResult`](types.md#reviewresult) | `director_ai.core.types` | Agent review output |
| [`ScoringEvidence`](types.md#scoringevidence) | `director_ai.core.types` | Retrieved chunks + NLI details |
| [`HaltEvidence`](types.md#haltevidence) | `director_ai.core.types` | Structured halt reason with evidence |
| [`CounterfactualHaltDiagnostic`](types.md#counterfactualhaltdiagnostic) | `director_ai.core.types` | Single-fact halt diagnostic |
| [`CausalAttributionGraph`](causal-attribution.md) | `director_ai.core.attribution.causal_graph` | DAG representation of scorer and halt causal pathways |
| [`SafetyEvent`](types.md#safetyevent) | `director_ai.core.safety_event` | Tenant-safe halt and policy event schema |
| [`DirectorSafetySignal`](director-safety-protocol.md) | `director_ai.core.safety_protocol` | Cross-runtime safety protocol envelope |
| [`ConflictAwareKnowledgeGuard`](conflict-aware-knowledge.md) | `director_ai.core.retrieval.conflict_guard` | Pre-ingestion KB conflict checks |
| [`InterlockKernel`](interlock-kernel.md) | `director_ai.interlock` | Standalone bring-your-own-scorer halt kernel |
| [`TokenEvent`](streaming.md#tokenevent) | `director_ai.core.runtime.streaming` | Per-token stream event |
| [`StreamSession`](streaming.md#streamsession) | `director_ai.core.runtime.streaming` | Complete stream session state |
| [`SustainabilityPolicyAdapter`](sustainability-scoring.md) | `director_ai.core.sustainability` | Token, cost, energy, carbon, quota, and forecast policy decisions |
| [`SustainabilityTelemetry`](sustainability-scoring.md) | `director_ai.core.sustainability` | Per-tenant sustainability summaries and threshold alerts |
| [`AgentPassportRegistry`](agent-passport-registry.md) | `director_ai.core.agent_identity` | Signed agent identity, capability policy, revocation, and coherence history |
| [`TrajectoryRollbackManager`](trajectory-preflight.md) | `director_ai.core.trajectory` | Tenant-safe rollback hook registry for preflight halt and escalation outcomes |

### Verification And Guard APIs

| Symbol | Module | Purpose |
|--------|--------|---------|
| `verify_numeric()` | `director_ai.core.verification.numeric_verifier` | Numeric consistency checks (arithmetic, dates, probabilities) |
| `NeuroSymbolicVerifier` | `director_ai.core.verification.neuro_symbolic` | Neural + symbolic contradiction gate |
| `verify_reasoning_chain()` | `director_ai.core.verification.reasoning_verifier` | Reasoning chain logic (non-sequiturs, circularity) |
| `score_temporal_freshness()` | `director_ai.core.scoring.temporal_freshness` | Staleness risk for date-sensitive claims |
| `ConsensusScorer` | `director_ai.core.scoring.consensus` | Cross-model factual agreement |
| `CrossVerifierConsensus` | `director_ai.core.scoring.consensus` | Critical-domain verifier fusion with required coverage and calibrated risk interval |
| `ByzantineFaultTolerantConsensus` | `director_ai.core.scoring.consensus` | PBFT-style verifier vote quorum |
| `ConformalPredictor` | `director_ai.core.calibration.conformal` | Calibrated P(hallucination) intervals |
| `ConformalRoutingPolicy` | `director_ai.core.calibration.conformal` | Conservative allow, human-review, stronger-model, and reject routing from conformal risk bounds |
| `ConformalRoutingDecision` | `director_ai.core.calibration.conformal` | Evidence-carrying route decision for calibrated uncertainty |
| `FeedbackLoopDetector` | `director_ai.compliance.feedback_loop_detector` | EU AI Act Art 15(4) feedback loop detection |
| `LoopMonitor` | `director_ai.agentic.loop_monitor` | Agent loop safety (circular, drift, budget) |
| `AdversarialTester` | `director_ai.testing.adversarial_suite` | 25-pattern adversarial robustness test |

### Interfaces

| Interface | Purpose |
|-----------|---------|
| [REST Server](server.md) | FastAPI endpoints (`/v1/review`, `/v1/health`, `/v1/metrics`, 8 gem endpoints) |
| [gRPC Server](grpc.md) | Protocol Buffers service (4 RPC methods) |
| [CLI](cli.md) | 22 command-line subcommands |

### Exceptions

| Exception | Raised When |
|-----------|-------------|
| [`HallucinationError`](exceptions.md) | `guard()` with `on_fail="raise"` detects low coherence |
| [`KernelHaltError`](exceptions.md#kernelhalterror) | SafetyKernel halts the output stream |
| [`ValidationError`](exceptions.md#validationerror) | Invalid configuration or input |
| [`DependencyError`](exceptions.md#dependencyerror) | Required optional package missing |

### Meta-Confidence And Contradiction APIs

| Symbol | Module | Purpose |
|--------|--------|---------|
| `compute_meta_confidence()` | `core.scoring.meta_confidence` | Verdict confidence from margin + signal agreement |
| `ContradictionTracker` | `core.runtime.contradiction_tracker` | Pairwise cross-turn contradiction matrix |
| `ContradictionReport` | `core.runtime.contradiction_tracker` | Contradiction summary (worst pair, trend) |
| `CrossDocumentConsistencyMemory` | `core.memory.consistency` | Durable tenant-scoped contradiction memory across documents |
| `DifferentialPrivacyScoreReleaser` | `core.federated_privacy.score_release` | Optional DP layer for public score release |
| `FederatedSafetySignalAggregator` | `core.federated_privacy.signal_sharing` | Anonymous DP aggregate sharing for tenant-safe guard signals |

See [Meta-Confidence Guide](../guide/meta-confidence.md).

### Structured Output Verification APIs

| Function | Module | Purpose |
|----------|--------|---------|
| `verify_json()` | `core.verification.json_verifier` | JSON Schema validation + value grounding |
| `verify_tool_call()` | `core.verification.tool_call_verifier` | Tool existence, arg validation, fabrication detection |
| `verify_code()` | `core.verification.code_verifier` | Python syntax, import existence, hallucinated API detection |

Result types: `StructuredVerificationResult`, `ToolCallResult`, `CodeCheckResult`, `FieldVerdict`.

See [Structured Verification Guide](../guide/structured-verification.md).

### Online Calibration APIs

| Class | Module | Purpose |
|-------|--------|---------|
| `FeedbackStore` | `core.calibration.feedback_store` | SQLite-backed human correction store |
| `OnlineCalibrator` | `core.calibration.online_calibrator` | Threshold sweep + FPR/FNR with Wilson CIs |
| `CalibrationReport` | `core.calibration.online_calibrator` | Calibration metrics dataclass |
| `HumanReviewQueue` | `core.runtime.human_review` | Reviewer case queue with append-only decisions |
| `AdaptiveThresholdLearner` | `core.calibration.adaptive_threshold` | Human-gated Thompson-sampling threshold recommender |

See [Online Calibration Guide](../guide/online-calibration.md).

### Compliance APIs

| Class | Module | Purpose |
|-------|--------|---------|
| `AuditLog` | `compliance.audit_log` | SQLite audit trail for every scored interaction |
| `AuditEntry` | `compliance.audit_log` | Single interaction record |
| `ComplianceReporter` | `compliance.reporter` | Article 15 report generator |
| `Article15Report` | `compliance.reporter` | Structured report with metrics, drift, incidents |
| `Article15TemplateContext` | `compliance.reporter` | Operator-supplied Article 15 technical documentation context |
| `Soc2IsoReadinessReport` | `compliance.readiness` | Tenant-safe SOC 2 / ISO 27001 readiness report |
| `Soc2IsoControl` | `compliance.readiness` | One readiness control mapped to SOC 2 and ISO 27001 evidence |
| `build_soc2_iso_readiness_report()` | `compliance.readiness` | Default or operator-supplied readiness report builder |
| `build_observability_operations_report()` | `ui.safety_dashboard` | Tenant-safe halt forensics, drift alerts, controls, and compliance-export packet |
| `ComplianceExportRef` | `ui.safety_dashboard` | Compliance export reference included without serialising artefact contents |
| `DriftDetector` | `compliance.drift_detector` | Statistical drift detection (two-proportion z-test) |
| `DriftResult` | `compliance.drift_detector` | Drift analysis result with z-score, p-value, severity |

See [Compliance Reporting Guide](../guide/compliance-reporting.md).

### Enterprise Moderation

| Class | Module | Purpose |
|-------|--------|---------|
| `ContentModerator` | `enterprise.moderation` | One-call PII redaction plus toxicity allow/warn/block wrapper |
| `ContentModerationResult` | `enterprise.moderation` | Tenant-safe moderation decision and metadata |
| `ModerationAction` | `enterprise.moderation` | `allow`, `redact`, `warn`, or `block` decision enum |
| `CustomRuleset` | `enterprise.rules_dsl` | Strict JSON/YAML custom policy loader for operator-owned rules |
| `CustomRule` | `enterprise.rules_dsl` | Validated compiled-rule input record |
| `RulesDslError` | `enterprise.rules_dsl` | Stable validation error for invalid ruleset documents |
| `build_trust_console_report()` | `ui.safety_dashboard` | Tenant-safe Trust Console JSON/Markdown report builder |
| `TrustControl` | `ui.safety_dashboard` | Readiness control row for customer-facing trust reports |
| `build_observability_operations_report()` | `ui.safety_dashboard` | Tenant-safe operations packet for halt forensics, drift alerts, and compliance exports |
| `ComplianceExportRef` | `ui.safety_dashboard` | Operator-owned compliance export reference |

See [Enterprise API](enterprise.md).

## Import Patterns

```python
# Top-level convenience imports
from director_ai import guard, score, get_score
from director_ai import CoherenceScorer, StreamingKernel, CoherenceAgent
from director_ai import InferenceHookRequest, build_inference_server_hook

# v3.10.0: Structured verification (stdlib only, no torch)
from director_ai import verify_json, verify_tool_call, verify_code

# v3.10.0: Online calibration
from director_ai import FeedbackStore, OnlineCalibrator, CalibrationReport

# v3.10.0: EU AI Act compliance
from director_ai import AuditLog, AuditEntry, ComplianceReporter, Article15Report, Article15TemplateContext

# Commercial readiness
from director_ai import build_soc2_iso_readiness_report

# v3.10.0: Meta-confidence
from director_ai import compute_meta_confidence, ContradictionTracker

# Direct module imports (for type hints and advanced use)
from director_ai.core.config import DirectorConfig
from director_ai.core.types import CoherenceScore, ReviewResult
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore, ChromaBackend
from director_ai.core.runtime.batch import BatchProcessor

# Enterprise (lazy-loaded)
from director_ai.enterprise import TenantRouter, Policy, AuditLogger, CustomRuleset
```
