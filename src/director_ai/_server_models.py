# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Server Pydantic Models
"""Request and response models for the Director-AI REST server.

Extracted from server.py to reduce module size.  All models are
re-imported into server.py to preserve backward compatibility.
"""

from __future__ import annotations

from typing import Any
from typing import Literal as _Literal

from pydantic import BaseModel, Field

_MAX_PROMPT_CHARS = 100_000
_MAX_RESPONSE_CHARS = 500_000


class ReviewRequest(BaseModel):
    """Request body for scoring one prompt/response pair."""

    prompt: str = Field(
        ..., min_length=1, max_length=_MAX_PROMPT_CHARS, description="Input prompt"
    )
    response: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_RESPONSE_CHARS,
        description="LLM response to review",
    )
    session_id: str | None = Field(None, description="Conversation session ID")


class ProcessRequest(BaseModel):
    """Request body for guarded generation from one prompt."""

    prompt: str = Field(
        ..., min_length=1, max_length=_MAX_PROMPT_CHARS, description="Input prompt"
    )


class BatchRequest(BaseModel):
    """Request body for batched process or review operations."""

    task: _Literal["process", "review"] = Field(
        "process", description="Task type: process or review"
    )
    prompts: list[str] = Field(
        ..., min_length=1, max_length=1000, description="List of prompts"
    )
    responses: list[str] = Field(default_factory=list, description="Optional responses")


class ReviewResponse(BaseModel):
    """Coherence review result returned by ``/v1/review``."""

    approved: bool
    coherence: float
    h_logical: float
    h_factual: float
    warning: bool = False
    evidence: dict | None = None


class FeedbackRequest(BaseModel):
    """Human feedback record for threshold calibration."""

    prompt: str = Field(
        ..., min_length=1, max_length=_MAX_PROMPT_CHARS, description="Original prompt"
    )
    response: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_RESPONSE_CHARS,
        description="Reviewed response",
    )
    guardrail_approved: bool
    human_approved: bool
    guardrail_score: float = Field(0.0, ge=0.0, le=1.0)
    domain: str = Field("", max_length=128)
    review_id: str = Field("", max_length=128)


class FeedbackResponse(BaseModel):
    """Acknowledgement for an accepted feedback correction."""

    accepted: bool
    correction_count: int
    disagreement: bool
    tenant_id: str = ""
    review_id: str = ""


class FeedbackCalibrationResponse(BaseModel):
    """Current calibration metrics derived from feedback records."""

    correction_count: int
    optimal_threshold: float | None
    current_accuracy: float
    tpr: float
    tnr: float
    fpr: float
    fnr: float
    fpr_ci: float
    fnr_ci: float


class ProcessResponse(BaseModel):
    """Guarded generation result returned by ``/v1/process``."""

    output: str
    coherence: float | None
    halted: bool
    candidates_evaluated: int
    warning: bool = False
    fallback_used: bool = False
    evidence: dict | None = None
    halt_evidence: dict | None = None


class BatchResponse(BaseModel):
    """Aggregate result for a batch request."""

    results: list[dict[str, Any]]
    errors: list[dict]
    total: int
    succeeded: int
    failed: int
    duration_seconds: float


class InjectionRequest(BaseModel):
    """Request body for output-side prompt-injection detection."""

    system_prompt: str = Field(
        "", max_length=_MAX_PROMPT_CHARS, description="System prompt / task description"
    )
    user_query: str = Field("", max_length=_MAX_PROMPT_CHARS, description="User query")
    response: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_RESPONSE_CHARS,
        description="LLM response to check for injection effects",
    )
    intent: str = Field(
        "",
        max_length=_MAX_PROMPT_CHARS,
        description="Direct intent (used if system_prompt/user_query empty)",
    )


class InjectionClaimResponse(BaseModel):
    """Per-claim injection-drift evidence."""

    claim: str
    claim_index: int
    intent_divergence: float
    reverse_divergence: float
    bidirectional_divergence: float
    traceability: float
    entity_match: float
    verdict: str
    confidence: float


class InjectionResponse(BaseModel):
    """Injection detection summary with per-claim evidence."""

    injection_detected: bool
    injection_risk: float
    intent_coverage: float
    total_claims: int
    grounded_claims: int
    drifted_claims: int
    injected_claims: int
    claims: list[dict] = []
    input_sanitizer_score: float
    combined_score: float


class VerifyResponse(BaseModel):
    """Generic verification response for structured guard checks."""

    approved: bool
    overall_score: float
    confidence: str = ""
    reason: str = ""
    claims: list[dict] = []


class TextRequest(BaseModel):
    """Request body carrying one text blob for verification."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_RESPONSE_CHARS,
        description="Text to analyze",
    )


class NumericIssueResponse(BaseModel):
    """One numeric consistency issue found in text."""

    issue_type: str
    description: str
    severity: str
    context: str


class NumericVerifyResponse(BaseModel):
    """Numeric consistency verification result."""

    claims_found: int
    issues: list[NumericIssueResponse]
    valid: bool
    error_count: int
    warning_count: int


class ReasoningVerdictResponse(BaseModel):
    """Verdict for one reasoning-chain step."""

    step_index: int
    step_text: str
    verdict: str
    confidence: float
    reason: str = ""
    premise_text: str = ""


class ReasoningVerifyResponse(BaseModel):
    """Reasoning-chain verification summary."""

    steps_found: int
    verdicts: list[ReasoningVerdictResponse]
    chain_valid: bool
    issues_found: int


class FreshnessClaimResponse(BaseModel):
    """Temporal freshness risk for one dated claim."""

    text: str
    claim_type: str
    staleness_risk: float
    reason: str
    source_id: str = ""
    external_status: str = ""


class FreshnessStatusResponse(BaseModel):
    """External citation or source-status risk for one source."""

    source_id: str
    status: str
    risk: float
    reason: str
    status_source: str = ""


class FreshnessResponse(BaseModel):
    """Temporal and external-status freshness summary."""

    claims: list[FreshnessClaimResponse]
    citation_statuses: list[FreshnessStatusResponse] = Field(default_factory=list)
    overall_staleness_risk: float
    external_status_risk: float = 0.0
    has_temporal_claims: bool
    stale_claim_count: int
    risky_status_count: int = 0


class ConsensusResponseItem(BaseModel):
    """One model response supplied to consensus scoring."""

    model: str
    response: str


class PairwiseAgreementResponse(BaseModel):
    """Pairwise divergence between two model responses."""

    model_a: str
    model_b: str
    divergence: float
    agreed: bool


class ConsensusResponse(BaseModel):
    """Cross-model consensus summary."""

    responses: list[ConsensusResponseItem]
    pairs: list[PairwiseAgreementResponse]
    agreement_score: float
    lowest_pair_agreement: float
    has_consensus: bool
    num_models: int


class ConsensusRequest(BaseModel):
    """Request body for cross-model consensus scoring."""

    responses: list[ConsensusResponseItem] = Field(
        ..., min_length=2, description="Responses from different models"
    )


class AdversarialPatternResponse(BaseModel):
    """Detection outcome for one adversarial test pattern."""

    name: str
    category: str
    transform: str
    detected: bool
    score: float
    original_score: float


class AdversarialResponse(BaseModel):
    """Adversarial robustness test summary."""

    total_patterns: int
    detected: int
    bypassed: int
    detection_rate: float
    is_robust: bool
    vulnerable_categories: list[str]
    results: list[AdversarialPatternResponse]


class ConformalRequest(BaseModel):
    """Request body for conformal uncertainty estimation."""

    score: float = Field(..., ge=0.0, le=1.0, description="Guardrail coherence score")
    calibration_scores: list[float] = Field(
        default_factory=list, description="Historical scores for calibration"
    )
    calibration_labels: list[bool] = Field(
        default_factory=list,
        description="True if the response was actually a hallucination",
    )
    coverage: float = Field(0.95, gt=0.0, lt=1.0)


class ConformalResponse(BaseModel):
    """Conformal prediction interval for hallucination risk."""

    point_estimate: float
    lower: float
    upper: float
    coverage: float
    calibration_size: int
    is_reliable: bool


class FeedbackLoopCheckRequest(BaseModel):
    """Request body for detecting repeated-output feedback loops."""

    input_text: str = Field(..., min_length=1, description="Current input to check")
    previous_outputs: list[str] = Field(
        default_factory=list, description="Previous AI outputs to match against"
    )
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0)


class FeedbackLoopResponse(BaseModel):
    """Feedback-loop detection result."""

    loop_detected: bool
    similarity: float
    severity: str = ""
    matched_output: str = ""


class AgenticStepRequest(BaseModel):
    """Request body for one agent action safety check."""

    goal: str = Field(..., min_length=1, description="Agent's original objective")
    action: str = Field(..., min_length=1, description="Current tool/function name")
    args: str = Field("", description="Serialized arguments")
    result: str = Field("", description="Tool output")
    tokens: int = Field(0, ge=0, description="Tokens consumed")
    step_history: list[dict] = Field(
        default_factory=list,
        description="Previous steps [{action, args}] for circular detection",
    )
    max_steps: int = Field(50, ge=1)


class AgenticStepResponse(BaseModel):
    """Agent action safety verdict."""

    step_number: int
    should_halt: bool
    should_warn: bool
    reasons: list[str]
    goal_drift_score: float
    budget_remaining_pct: float


class HealthResponse(BaseModel):
    """Operational health response for the REST service."""

    status: str = "ok"
    version: str
    mode: str
    profile: str
    nli_loaded: bool
    uptime_seconds: float
    routers: dict[str, str] = Field(default_factory=dict)
    model_revisions: dict = Field(default_factory=dict)


class ReadyResponse(BaseModel):
    """Readiness response for deployment probes."""

    ready: bool
    reason: str = ""


class SourceResponse(BaseModel):
    """AGPL source-compliance response."""

    license: str
    version: str
    licensee: str = ""
    tier: str = ""
    repository_url: str = ""
    instructions: str = ""
    agpl_obligation: str = ""
    agpl_section: str = ""


class ConfigResponse(BaseModel):
    """Sanitised runtime configuration response."""

    config: dict


class TenantFactRequest(BaseModel):
    """Request body for writing one tenant-scoped fact."""

    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    signature: str = Field("", max_length=256)
    signature_key_id: str = Field("", max_length=128)


class TenantVectorFactRequest(BaseModel):
    """Request body for writing one tenant-scoped vector fact."""

    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)
    backend_type: str = Field("memory", description="Vector backend type")
    signature: str = Field("", max_length=256)
    signature_key_id: str = Field("", max_length=128)


class TenantInfo(BaseModel):
    """Tenant inventory row."""

    id: str
    fact_count: int


class TenantListResponse(BaseModel):
    """Tenant inventory response."""

    tenants: list[TenantInfo]


class StatusResponse(BaseModel):
    """Generic status/count response for tenant operations."""

    status: str
    tenant_id: str = ""
    key: str = ""
    backend_type: str = ""
    count: int = 0


class TurnInfo(BaseModel):
    """One stored conversation turn."""

    prompt: str
    response: str
    score: float
    turn_index: int


class SessionResponse(BaseModel):
    """Conversation session state response."""

    session_id: str
    turn_count: int
    turns: list[TurnInfo]


class DeletedResponse(BaseModel):
    """Deletion acknowledgement for session operations."""

    status: str
    session_id: str


class StatsResponse(BaseModel):
    """Aggregate runtime scoring statistics."""

    total: int = 0
    approved: int = 0
    rejected: int = 0
    halted: int = 0
    avg_score: float | None = None
    avg_latency_ms: float | None = None


class HourlyDataPoint(BaseModel):
    """Hourly statistics row."""

    hour: str = ""
    total: int = 0
    approved: int = 0
    rejected: int = 0


class HourlyResponse(BaseModel):
    """Hourly statistics response."""

    data: list[dict] = []
    note: str = ""


class ModelMetricsResponse(BaseModel):
    """Per-model compliance metrics row."""

    model: str
    total_requests: int
    hallucination_rate: float
    hallucination_rate_ci: float | list[float] = 0.0
    avg_score: float
    avg_confidence: float
    avg_latency_ms: float


class ComplianceReportResponse(BaseModel):
    """Compliance report summary returned by REST endpoints."""

    report_timestamp: float
    period_start: float
    period_end: float
    total_interactions: int
    overall_hallucination_rate: float
    overall_hallucination_rate_ci: float | list[float] = 0.0
    avg_score: float
    avg_verdict_confidence: float
    avg_latency_ms: float
    human_override_count: int
    human_override_rate: float
    model_metrics: list[ModelMetricsResponse] = []
    drift_detected: bool
    drift_severity: float | str = 0.0
    incident_count: int = 0


class WindowStats(BaseModel):
    """One comparison window used for drift detection."""

    start: float
    end: float
    total: int
    rejected: int
    hallucination_rate: float


class DriftResponse(BaseModel):
    """Drift detection result across comparison windows."""

    detected: bool
    severity: str
    z_score: float
    p_value: float
    rate_change: float
    windows: list[WindowStats] = []


class PeriodMetrics(BaseModel):
    """Compliance metrics for one reporting period."""

    total: int
    hallucination_rate: float
    avg_score: float


class ComplianceDashboardResponse(BaseModel):
    """24h / 7d / 30d compliance metrics."""

    period_24h: PeriodMetrics = PeriodMetrics(
        total=0, hallucination_rate=0, avg_score=0
    )
    period_7d: PeriodMetrics = PeriodMetrics(total=0, hallucination_rate=0, avg_score=0)
    period_30d: PeriodMetrics = PeriodMetrics(
        total=0, hallucination_rate=0, avg_score=0
    )
