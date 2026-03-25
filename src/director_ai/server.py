# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Ă‚Â© Concepts 1996Ă˘â‚¬â€ś2026 Miroslav ÄąÂ otek. All rights reserved.
# Ă‚Â© Code 2020Ă˘â‚¬â€ś2026 Miroslav ÄąÂ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI Ă˘â‚¬â€ť FastAPI Server

"""Production-ready FastAPI server for Director-Class AI.

Usage::

    # Programmatic
    from director_ai.server import create_app
    app = create_app()

    # CLI
    director-ai serve --port 8080
"""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import json as _json_mod
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from typing import Literal as _Literal

from .core.config import DirectorConfig
from .core.metrics import metrics

__all__ = ["create_app"]

REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="",
)

logger = logging.getLogger("DirectorAI.Server")

_WS_MAX_PROMPT_LENGTH = 100_000
_WS_MAX_CONCURRENT = 8
_AUTH_EXEMPT_PATHS_BASE = frozenset({"/v1/health", "/v1/ready", "/v1/source"})

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    import slowapi
    from slowapi import Limiter
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    _SLOWAPI_AVAILABLE = True
except ImportError:
    slowapi = None  # type: ignore
    Limiter = None  # type: ignore
    _SLOWAPI_AVAILABLE = False


def _check_fastapi() -> None:
    if not _FASTAPI_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )


# Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Pydantic request/response models Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

_MAX_PROMPT_CHARS = 100_000
_MAX_RESPONSE_CHARS = 500_000

if _FASTAPI_AVAILABLE:  # pragma: no branch

    class ReviewRequest(BaseModel):
        prompt: str = Field(
            ...,
            min_length=1,
            max_length=_MAX_PROMPT_CHARS,
            description="Input prompt",
        )
        response: str = Field(
            ...,
            min_length=1,
            max_length=_MAX_RESPONSE_CHARS,
            description="LLM response to review",
        )
        session_id: str | None = Field(None, description="Conversation session ID")

    class ProcessRequest(BaseModel):
        prompt: str = Field(
            ...,
            min_length=1,
            max_length=_MAX_PROMPT_CHARS,
            description="Input prompt",
        )

    class BatchRequest(BaseModel):
        task: _Literal["process", "review"] = Field(
            "process", description="Task type: process or review"
        )
        prompts: list[str] = Field(
            ...,
            min_length=1,
            max_length=1000,
            description="List of prompts",
        )
        responses: list[str] = Field(
            default_factory=list,
            description="Optional responses",
        )

    class ReviewResponse(BaseModel):
        approved: bool
        coherence: float
        h_logical: float
        h_factual: float
        warning: bool = False
        evidence: dict | None = None

    class ProcessResponse(BaseModel):
        output: str
        coherence: float | None
        halted: bool
        candidates_evaluated: int
        warning: bool = False
        fallback_used: bool = False
        evidence: dict | None = None
        halt_evidence: dict | None = None

    class BatchResponse(BaseModel):
        results: list[dict[str, Any]]
        errors: list[dict]
        total: int
        succeeded: int
        failed: int
        duration_seconds: float

    class HealthResponse(BaseModel):
        status: str = "ok"
        version: str
        mode: str
        profile: str
        nli_loaded: bool
        uptime_seconds: float

    class ConfigResponse(BaseModel):
        config: dict

    class TenantFactRequest(BaseModel):
        key: str = Field(..., min_length=1)
        value: str = Field(..., min_length=1)

    class TenantVectorFactRequest(BaseModel):
        key: str = Field(..., min_length=1)
        value: str = Field(..., min_length=1)
        backend_type: str = Field("memory", description="Vector backend type")

    class TextRequest(BaseModel):
        text: str = Field(
            ...,
            min_length=1,
            max_length=_MAX_RESPONSE_CHARS,
            description="Text to analyze",
        )

    class NumericIssueResponse(BaseModel):
        issue_type: str
        description: str
        severity: str
        context: str

    class NumericVerifyResponse(BaseModel):
        claims_found: int
        issues: list[NumericIssueResponse]
        valid: bool
        error_count: int
        warning_count: int

    class ReasoningVerdictResponse(BaseModel):
        step_index: int
        step_text: str
        verdict: str
        confidence: float
        reason: str = ""
        premise_text: str = ""

    class ReasoningVerifyResponse(BaseModel):
        steps_found: int
        verdicts: list[ReasoningVerdictResponse]
        chain_valid: bool
        issues_found: int

    class FreshnessClaimResponse(BaseModel):
        text: str
        claim_type: str
        staleness_risk: float
        reason: str

    class FreshnessResponse(BaseModel):
        claims: list[FreshnessClaimResponse]
        overall_staleness_risk: float
        has_temporal_claims: bool
        stale_claim_count: int

    # -- Consensus models --

    class ConsensusResponseItem(BaseModel):
        model: str
        response: str

    class PairwiseAgreementResponse(BaseModel):
        model_a: str
        model_b: str
        divergence: float
        agreed: bool

    class ConsensusResponse(BaseModel):
        responses: list[ConsensusResponseItem]
        pairs: list[PairwiseAgreementResponse]
        agreement_score: float
        lowest_pair_agreement: float
        has_consensus: bool
        num_models: int

    class ConsensusRequest(BaseModel):
        responses: list[ConsensusResponseItem] = Field(
            ..., min_length=2, description="Responses from different models"
        )

    # -- Adversarial models --

    class AdversarialPatternResponse(BaseModel):
        name: str
        category: str
        transform: str
        detected: bool
        score: float
        original_score: float

    class AdversarialResponse(BaseModel):
        total_patterns: int
        detected: int
        bypassed: int
        detection_rate: float
        is_robust: bool
        vulnerable_categories: list[str]
        results: list[AdversarialPatternResponse]

    # -- Conformal models --

    class ConformalRequest(BaseModel):
        score: float = Field(
            ..., ge=0.0, le=1.0, description="Guardrail coherence score"
        )
        calibration_scores: list[float] = Field(
            default_factory=list, description="Historical scores for calibration"
        )
        calibration_labels: list[bool] = Field(
            default_factory=list,
            description="True if the response was actually a hallucination",
        )
        coverage: float = Field(0.95, gt=0.0, lt=1.0)

    class ConformalResponse(BaseModel):
        point_estimate: float
        lower: float
        upper: float
        coverage: float
        calibration_size: int
        is_reliable: bool

    # -- Feedback loop models --

    class FeedbackLoopCheckRequest(BaseModel):
        input_text: str = Field(..., min_length=1, description="Current input to check")
        previous_outputs: list[str] = Field(
            default_factory=list, description="Previous AI outputs to match against"
        )
        similarity_threshold: float = Field(0.5, ge=0.0, le=1.0)

    class FeedbackLoopResponse(BaseModel):
        loop_detected: bool
        similarity: float
        severity: str = ""
        matched_output: str = ""

    # -- Agentic loop monitor models --

    class AgenticStepRequest(BaseModel):
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
        step_number: int
        should_halt: bool
        should_warn: bool
        reasons: list[str]
        goal_drift_score: float
        budget_remaining_pct: float

    # -- Operational response models --

    class ReadyResponse(BaseModel):
        ready: bool
        reason: str = ""

    class SourceResponse(BaseModel):
        license: str
        version: str
        licensee: str = ""
        tier: str = ""
        repository_url: str = ""
        instructions: str = ""
        agpl_obligation: str = ""
        agpl_section: str = ""

    class TenantInfo(BaseModel):
        id: str
        fact_count: int

    class TenantListResponse(BaseModel):
        tenants: list[TenantInfo]

    class StatusResponse(BaseModel):
        status: str
        tenant_id: str = ""
        key: str = ""
        backend_type: str = ""
        count: int = 0

    class TurnInfo(BaseModel):
        prompt: str
        response: str
        score: float
        turn_index: int

    class SessionResponse(BaseModel):
        session_id: str
        turn_count: int
        turns: list[TurnInfo]

    class DeletedResponse(BaseModel):
        status: str
        session_id: str

    class StatsResponse(BaseModel):
        total: int = 0
        approved: int = 0
        rejected: int = 0
        halted: int = 0
        avg_score: float | None = None
        avg_latency_ms: float | None = None

    class VerifyResponse(BaseModel):
        approved: bool
        overall_score: float
        confidence: str = ""
        reason: str = ""
        claims: list[dict] = []

    class HourlyDataPoint(BaseModel):
        hour: str = ""
        total: int = 0
        approved: int = 0
        rejected: int = 0

    class HourlyResponse(BaseModel):
        data: list[dict] = []
        note: str = ""

    class ModelMetricsResponse(BaseModel):
        model: str
        total_requests: int
        hallucination_rate: float
        hallucination_rate_ci: float | list[float] = 0.0
        avg_score: float
        avg_confidence: float
        avg_latency_ms: float

    class ComplianceReportResponse(BaseModel):
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
        start: float
        end: float
        total: int
        rejected: int
        hallucination_rate: float

    class DriftResponse(BaseModel):
        detected: bool
        severity: str
        z_score: float
        p_value: float
        rate_change: float
        windows: list[WindowStats] = []

    class PeriodMetrics(BaseModel):
        total: int
        hallucination_rate: float
        avg_score: float

    class ComplianceDashboardResponse(BaseModel):
        """24h / 7d / 30d compliance metrics."""

        period_24h: PeriodMetrics = PeriodMetrics(
            total=0, hallucination_rate=0, avg_score=0
        )
        period_7d: PeriodMetrics = PeriodMetrics(
            total=0, hallucination_rate=0, avg_score=0
        )
        period_30d: PeriodMetrics = PeriodMetrics(
            total=0, hallucination_rate=0, avg_score=0
        )


def _halt_evidence_to_dict(halt_ev) -> dict | None:
    if halt_ev is None:
        return None
    return {
        "reason": halt_ev.reason,
        "last_score": halt_ev.last_score,
        "evidence_chunks": [
            {"text": c.text, "distance": c.distance, "source": c.source}
            for c in halt_ev.evidence_chunks
        ],
        "nli_scores": halt_ev.nli_scores,
        "suggested_action": halt_ev.suggested_action,
    }


def _evidence_to_dict(evidence) -> dict | None:
    if evidence is None:
        return None
    d = {
        "chunks": [
            {"text": c.text, "distance": c.distance, "source": c.source}
            for c in evidence.chunks
        ],
        "nli_premise": evidence.nli_premise,
        "nli_hypothesis": evidence.nli_hypothesis,
        "nli_score": evidence.nli_score,
        "premise_chunk_count": evidence.premise_chunk_count,
        "hypothesis_chunk_count": evidence.hypothesis_chunk_count,
    }
    if evidence.claim_coverage is not None:
        d["claim_coverage"] = evidence.claim_coverage
        d["per_claim_divergences"] = evidence.per_claim_divergences
        d["claims"] = evidence.claims
    if evidence.attributions is not None:
        d["attributions"] = [
            {
                "claim": a.claim,
                "claim_index": a.claim_index,
                "source_sentence": a.source_sentence,
                "source_index": a.source_index,
                "divergence": a.divergence,
                "supported": a.supported,
            }
            for a in evidence.attributions
        ]
    if evidence.token_count is not None:
        d["token_count"] = evidence.token_count
        d["estimated_cost_usd"] = evidence.estimated_cost_usd
    return d


def create_app(config: DirectorConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    _check_fastapi()

    if config is None:
        import os

        profile = os.environ.get("DIRECTOR_PROFILE", "")
        if profile and profile != "default":
            cfg = DirectorConfig.from_profile(profile)
        else:
            cfg = DirectorConfig.from_env()
    else:
        cfg = config
    _start_time = time.monotonic()
    _state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifecycle events for the FastAPI server."""
        cfg = app.state.config

        from .core.license import load_license

        lic = load_license()
        app.state._license = lic
        if lic.is_commercial:
            logger.info(
                "Director-AI v%s Ă˘â‚¬â€ť Licensed to %s (%s tier)",
                __import__("director_ai").__version__,
                lic.licensee or lic.key[:20],
                lic.tier,
            )
        elif lic.is_trial:
            logger.info("Director-AI Ă˘â‚¬â€ť Trial license (expires %s)", lic.expires)
        else:
            logger.info("Director-AI Ă˘â‚¬â€ť AGPL-3.0-or-later (community)")

        logger.info("Starting Director-AI server")

        app.state._state = {}  # Initialize _state on app.state

        from .core.agent import CoherenceAgent
        from .core.runtime.batch import BatchProcessor
        from .core.safety.audit import AuditLogger
        from .core.safety.sanitizer import InputSanitizer
        from .core.tenant import TenantRouter

        if cfg.sanitize_inputs:
            app.state._state["sanitizer"] = InputSanitizer(
                block_threshold=cfg.sanitizer_block_threshold,
            )

        from .enterprise.redactor import PIIRedactor

        app.state._state["redactor"] = PIIRedactor(enabled=cfg.redact_pii)
        if cfg.redact_pii:
            logger.info("Enterprise PII Redaction enabled")

        store = cfg.build_store()
        scorer = cfg.build_scorer(store=store)
        agent_kwargs: dict = {"_scorer": scorer, "_store": store}
        if cfg.llm_provider == "local":
            agent_kwargs["llm_api_url"] = cfg.llm_api_url
        elif cfg.llm_provider in ("openai", "anthropic"):
            agent_kwargs["provider"] = cfg.llm_provider
            if cfg.llm_api_key:
                import os as _os

                _env_keys = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                }
                _os.environ.setdefault(_env_keys[cfg.llm_provider], cfg.llm_api_key)
            logger.info("LLM provider: %s", cfg.llm_provider)
        agent = CoherenceAgent(**agent_kwargs)
        batch_proc = BatchProcessor(agent, max_concurrency=cfg.batch_max_concurrency)

        stats = None
        if cfg.stats_backend == "sqlite":
            from .core.stats import StatsStore

            stats = StatsStore(db_path=cfg.stats_db_path)
            logger.info("SQLite stats backend: %s", cfg.stats_db_path)

        app.state._state["agent"] = agent
        app.state._state["scorer"] = scorer
        app.state._state["batch"] = batch_proc
        app.state._state["config"] = cfg
        app.state._state["stats"] = stats
        app.state._state["sessions"] = {}
        app.state._state["session_owners"] = {}
        app.state._state["sessions_lock"] = asyncio.Lock()
        app.state._state["max_sessions"] = getattr(cfg, "max_sessions", 10000)

        review_queue = None
        if cfg.review_queue_enabled:
            from .core.runtime.review_queue import ReviewQueue

            review_queue = ReviewQueue(
                scorer,
                max_batch=cfg.review_queue_max_batch,
                flush_timeout_ms=cfg.review_queue_flush_timeout_ms,
            )
            await review_queue.start()
        app.state._state["review_queue"] = review_queue

        if cfg.audit_log_path or cfg.audit_postgres_url:
            audit_logger = AuditLogger(path=cfg.audit_log_path)
            if cfg.audit_postgres_url:
                from .enterprise.audit_pg import PostgresAuditSink

                audit_logger.add_sink(PostgresAuditSink(db_url=cfg.audit_postgres_url))

            app.state._state["audit"] = audit_logger
            logger.info(
                "Audit logging initialized (path: %s, db: %s)",
                bool(cfg.audit_log_path),
                bool(cfg.audit_postgres_url),
            )

        if cfg.compliance_db_path:
            from .compliance.audit_log import AuditLog as ComplianceAuditLog
            from .compliance.drift_detector import DriftDetector
            from .compliance.reporter import ComplianceReporter

            c_log = ComplianceAuditLog(cfg.compliance_db_path)
            app.state._state["compliance_log"] = c_log
            app.state._state["compliance_reporter"] = ComplianceReporter(c_log)
            app.state._state["compliance_drift"] = DriftDetector(c_log)
            logger.info("Compliance audit log: %s", cfg.compliance_db_path)

        if cfg.tenant_routing:
            app.state._state["tenant_router"] = TenantRouter()
            logger.info("Tenant routing enabled")

        from .core.retrieval.doc_registry import DocRegistry

        app.state._state["doc_registry"] = DocRegistry()

        cfg.configure_logging()

        if cfg.otel_enabled:
            from .core.otel import setup_otel

            setup_otel()

        if cfg.use_nli:  # pragma: no cover Ă˘â‚¬â€ť lifespan only runs under ASGI
            metrics.gauge_set("nli_model_loaded", 1.0)

        logger.info(
            "Director AI server started (profile=%s, nli=%s)",
            cfg.profile,
            cfg.use_nli,
        )
        yield
        logger.info("Director AI server shutting down")
        if review_queue:
            await review_queue.stop()
        if stats:
            try:
                stats.close()
            except Exception:  # pragma: no cover Ă˘â‚¬â€ť defensive
                logger.warning("Failed to close stats database")
        c_log_shutdown = app.state._state.get("compliance_log")
        if c_log_shutdown is not None:
            c_log_shutdown.close()

    app = FastAPI(
        title="Director-Class AI",
        version="3.10.1",
        description="Real-time multi-agent orchestration and coherence scoring.",
        lifespan=lifespan,
    )
    app.state.config = cfg

    # Fine-tuning API router (Phase C)
    try:
        from .finetune_api import create_finetune_router

        app.include_router(create_finetune_router(), prefix="/v1/finetune")
    except ImportError:
        pass

    # Knowledge ingestion API
    try:
        from .knowledge_api import create_knowledge_router

        app.include_router(create_knowledge_router(), prefix="/v1/knowledge")
    except ImportError:
        pass

    _origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]
    if len(_origins) > 100:
        raise ValueError(f"Too many CORS origins: {len(_origins)} (max 100)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-API-Key",
            "X-Request-ID",
            "X-Tenant-ID",
        ],
    )

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Rate limiting Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    _rate_str = f"{cfg.rate_limit_rpm}/minute" if cfg.rate_limit_rpm > 0 else ""

    limiter = None
    if cfg.rate_limit_rpm > 0:
        if not _SLOWAPI_AVAILABLE:
            if cfg.rate_limit_strict:
                raise ImportError(
                    "rate_limit_strict=True but slowapi not installed. "
                    "Install with: pip install director-ai[server]",
                )
            logger.warning(
                "rate_limit_rpm=%d but slowapi not installed. "
                "Install with: pip install director-ai[server]",
                cfg.rate_limit_rpm,
            )
        else:
            storage_uri = None
            if cfg.redis_url:
                storage_uri = cfg.redis_url
                logger.info(
                    "Rate limiting backed by Redis: %s",
                    cfg.redis_url.split("@")[-1]
                    if "@" in cfg.redis_url
                    else cfg.redis_url,
                )
            limiter = Limiter(
                key_func=get_remote_address,
                default_limits=[_rate_str],
                storage_uri=storage_uri,
            )
            app.state.limiter = limiter
            from slowapi.errors import RateLimitExceeded

            app.add_middleware(SlowAPIMiddleware)

            @app.exception_handler(RateLimitExceeded)
            async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
                return JSONResponse(  # pragma: no cover Ă˘â‚¬â€ť ASGI runtime handler
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Middleware: correlation IDs + API key auth + metrics Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    _auth_exempt = (
        _AUTH_EXEMPT_PATHS_BASE
        if cfg.metrics_require_auth
        else _AUTH_EXEMPT_PATHS_BASE | {"/v1/metrics/prometheus"}
    )

    _api_key_tenant_map: dict[str, str] = {}
    if cfg.api_key_tenant_map:
        _api_key_tenant_map = _json_mod.loads(cfg.api_key_tenant_map)

    @app.middleware("http")
    async def _http_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        REQUEST_ID_CTX.set(request_id)

        api_key_hash = ""
        if cfg.api_keys and request.url.path not in _auth_exempt:
            provided = request.headers.get("X-API-Key", "")
            if not any(hmac.compare_digest(provided, k) for k in cfg.api_keys):
                logger.warning(
                    "Auth failed from %s on %s",
                    request.client.host if request.client else "unknown",
                    request.url.path,
                )
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                    headers={"X-Request-ID": request_id},
                )
            import hashlib

            # Truncated SHA-256 fingerprint for audit logs only — NOT used for
            # authentication or password storage. The API key is verified via
            # constant-time HMAC comparison above.
            api_key_hash = hashlib.sha256(provided.encode()).hexdigest()[:16]

            # Tenant binding: enforce API key Ă˘â€ â€™ tenant mapping if configured
            if _api_key_tenant_map:
                if provided not in _api_key_tenant_map:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "API key not bound to any tenant"},
                        headers={"X-Request-ID": request_id},
                    )
                bound_tenant = _api_key_tenant_map[provided]
                claimed_tenant = request.headers.get("X-Tenant-ID", "")
                if claimed_tenant and claimed_tenant != bound_tenant:
                    return JSONResponse(
                        status_code=403,
                        content={"detail": "API key not authorized for this tenant"},
                        headers={"X-Request-ID": request_id},
                    )
                request.state.tenant_id = bound_tenant
            else:
                request.state.tenant_id = request.headers.get("X-Tenant-ID", "")
        else:
            request.state.tenant_id = request.headers.get("X-Tenant-ID", "")

        request.state.api_key_hash = api_key_hash

        # Metrics
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        metrics.observe("http_request_duration_seconds", elapsed)
        metrics.inc_labeled(
            "http_requests_total",
            {
                "method": request.method,
                "endpoint": request.url.path,
                "status": str(response.status_code),
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Health Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.get("/v1/health", response_model=HealthResponse)
    async def health(request: Request):
        import director_ai

        lic = getattr(request.app.state, "_license", None)
        extra = {}
        if lic and lic.is_commercial:
            extra = {
                "license": "commercial",
                "tier": lic.tier,
                "licensee": lic.licensee,
            }
        elif lic and lic.is_trial:
            extra = {"license": "trial", "expires": lic.expires}
        else:
            extra = {"license": "agpl"}

        resp = HealthResponse(
            version=director_ai.__version__,
            mode=cfg.mode,
            profile=cfg.profile,
            nli_loaded=cfg.use_nli,
            uptime_seconds=time.monotonic() - _start_time,
        )
        return {**resp.model_dump(), **extra}

    @app.get("/v1/ready", response_model=ReadyResponse)
    async def readiness(request: Request):
        """Readiness probe: returns 200 only when scorer is operational."""
        scorer = request.app.state._state.get("scorer")
        if scorer is None:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "reason": "scorer not initialised"},
            )
        nli = getattr(scorer, "_nli", None)
        if cfg.use_nli and nli is not None and not nli.model_available:
            return JSONResponse(
                status_code=503,
                content={"ready": False, "reason": "NLI model not loaded"},
            )
        return {"ready": True}

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ AGPL Ă‚Â§13 source endpoint Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.get("/v1/source", response_model=SourceResponse)
    async def source(request: Request):
        import director_ai

        lic = getattr(request.app.state, "_license", None)
        if lic and lic.is_commercial:
            if not cfg.source_endpoint_enabled:
                raise HTTPException(
                    404, "Source endpoint disabled (commercial license)"
                )
            return {
                "license": "commercial",
                "licensee": lic.licensee,
                "tier": lic.tier,
                "version": director_ai.__version__,
                "agpl_obligation": "waived",
            }

        if not cfg.source_endpoint_enabled:
            raise HTTPException(404, "Source endpoint disabled")

        return {
            "license": "AGPL-3.0-or-later",
            "version": director_ai.__version__,
            "repository_url": cfg.source_repository_url,
            "instructions": f"git clone {cfg.source_repository_url}",
            "agpl_section": "13",
        }

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Review Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.post("/v1/review", response_model=ReviewResponse)
    async def review(
        req: ReviewRequest,
        request: Request,
    ) -> ReviewResponse:
        """Score an AI response against a given prompt using the active agent."""
        sanitizer = request.app.state._state.get("sanitizer")
        if sanitizer:
            check = sanitizer.check(req.prompt)
            if check.blocked:
                raise HTTPException(400, f"Prompt injection rejected: {check.reason}")

        redactor = request.app.state._state.get("redactor")
        if redactor and hasattr(redactor, "enabled") and redactor.enabled:
            req.prompt = redactor.redact(req.prompt)
            req.response = redactor.redact(req.response)

        scorer = request.app.state._state.get("scorer")
        if not scorer:  # pragma: no cover Ă˘â‚¬â€ť lifespan always sets scorer
            raise HTTPException(503, "Server not ready")

        # Tenant routing Ă˘â‚¬â€ť S-05: log tenant access for audit trail
        tenant_id = getattr(
            request.state,
            "tenant_id",
            request.headers.get("X-Tenant-ID", ""),
        )
        if tenant_id:
            logger.info(
                "Tenant access: tenant=%s src=%s path=%s",
                tenant_id,
                request.client.host if request.client else "unknown",
                request.url.path,
            )

        session = None
        if req.session_id:
            from .core.runtime.session import ConversationSession

            caller_hash = getattr(request.state, "api_key_hash", "")
            async with request.app.state._state["sessions_lock"]:
                sessions = request.app.state._state["sessions"]
                owners = request.app.state._state["session_owners"]
                if req.session_id not in sessions:
                    max_s = request.app.state._state.get("max_sessions", 10000)
                    if len(sessions) >= max_s:
                        oldest = next(iter(sessions))
                        del sessions[oldest]
                        owners.pop(oldest, None)
                    sessions[req.session_id] = ConversationSession(
                        session_id=req.session_id,
                    )
                    owners[req.session_id] = caller_hash
                else:
                    owner = owners.get(req.session_id, "")
                    if owner and owner != caller_hash:
                        raise HTTPException(
                            403, "Session belongs to a different API key"
                        )
                session = sessions[req.session_id]

        metrics.inc("reviews_total")
        start = time.monotonic()
        review_queue = request.app.state._state.get("review_queue")
        if review_queue and not session:
            with metrics.timer("review_duration_seconds"):
                approved, score = await review_queue.submit(
                    req.prompt,
                    req.response,
                    tenant_id=tenant_id,
                )
        else:
            loop = asyncio.get_running_loop()
            with metrics.timer("review_duration_seconds"):
                approved, score = await loop.run_in_executor(
                    None,
                    lambda: scorer.review(
                        req.prompt,
                        req.response,
                        session=session,
                        tenant_id=tenant_id,
                    ),
                )
        latency_ms = (time.monotonic() - start) * 1000

        if approved:
            metrics.inc("reviews_approved")
        else:
            metrics.inc("reviews_rejected")
        metrics.observe("coherence_score", score.score)

        stats_store = request.app.state._state.get("stats")
        if stats_store:
            stats_store.record_review(
                approved=approved,
                score=score.score,
                h_logical=score.h_logical,
                h_factual=score.h_factual,
            )

        audit = request.app.state._state.get("audit")
        if audit:
            audit.log_review(
                query=req.prompt,
                response=req.response,
                approved=approved,
                score=score.score,
                h_logical=score.h_logical,
                h_factual=score.h_factual,
                tenant_id=tenant_id,
                latency_ms=latency_ms,
            )

        c_log = request.app.state._state.get("compliance_log")
        if c_log:
            from .compliance.audit_log import AuditEntry as CAuditEntry

            c_log.log(
                CAuditEntry(
                    prompt=req.prompt,
                    response=req.response,
                    model=getattr(cfg, "llm_model", "server"),
                    provider="server",
                    score=score.score,
                    approved=approved,
                    verdict_confidence=getattr(score, "verdict_confidence", 0.0),
                    task_type="review",
                    domain="",
                    latency_ms=latency_ms,
                    timestamp=time.time(),
                    tenant_id=tenant_id,
                )
            )

        return ReviewResponse(
            approved=approved,
            coherence=score.score,
            h_logical=score.h_logical,
            h_factual=score.h_factual,
            warning=score.warning,
            evidence=_evidence_to_dict(score.evidence),
        )

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Verified Review (sentence-level multi-signal) Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.post("/v1/verify", response_model=VerifyResponse)
    async def verify_response(req: ReviewRequest, request: Request):
        """Sentence-level multi-signal fact verification.

        Decomposes the response into claims, matches each to the best
        source sentence from the KB, checks NLI + entity + number +
        negation signals. Returns per-claim verdicts with confidence.
        """
        import asyncio

        from .core.scoring.verified_scorer import VerifiedScorer

        sanitizer = request.app.state._state.get("sanitizer")
        if sanitizer:
            check = sanitizer.check(req.prompt)
            if check.blocked:
                raise HTTPException(400, f"Prompt rejected: {check.reason}")

        redactor = request.app.state._state.get("redactor")
        if redactor and hasattr(redactor, "enabled") and redactor.enabled:
            req.prompt = redactor.redact(req.prompt)
            req.response = redactor.redact(req.response)

        scorer = request.app.state._state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "Scorer not initialised")

        tenant_id = getattr(request.state, "tenant_id", "")
        store = getattr(scorer, "ground_truth_store", None)

        # Retrieve source context
        context = ""
        if store:
            ctx = store.retrieve_context(
                req.prompt,
                top_k=5,
                tenant_id=tenant_id,
            )
            if ctx:
                context = ctx

        if not context:
            return {
                "approved": False,
                "overall_score": 0.0,
                "confidence": "low",
                "reason": "No relevant context found in knowledge base",
                "claims": [],
            }

        nli = getattr(scorer, "_nli", None)
        vs = VerifiedScorer(nli_scorer=nli)

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            vs.verify,
            req.response,
            context,
        )
        return result.to_dict()

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Process Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.post("/v1/process", response_model=ProcessResponse)
    async def process(
        req: ProcessRequest,
        request: Request,
    ) -> ProcessResponse | PlainTextResponse:
        """Process a prompt through the Director AI pipeline."""
        sanitizer = request.app.state._state.get("sanitizer")
        if sanitizer:
            check = sanitizer.check(req.prompt)
            if check.blocked:
                raise HTTPException(400, f"Prompt injection rejected: {check.reason}")

        redactor = request.app.state._state.get("redactor")
        if redactor and redactor.enabled:
            req.prompt = redactor(req.prompt)

        agent = request.app.state._state.get("agent")
        if not agent:  # pragma: no cover Ă˘â‚¬â€ť lifespan always sets agent
            raise HTTPException(503, "Server not ready")
        metrics.inc("reviews_total")
        start = time.monotonic()

        tenant_id = getattr(
            request.state,
            "tenant_id",
            request.headers.get("X-Tenant-ID", ""),
        )
        if tenant_id:
            logger.info(
                "Tenant access: tenant=%s src=%s path=%s",
                tenant_id,
                request.client.host if request.client else "unknown",
                request.url.path,
            )

        try:
            with metrics.timer("review_duration_seconds"):
                result = await agent.aprocess(req.prompt, tenant_id=tenant_id)
        except Exception as e:
            logger.error("Review processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal processing error",
            ) from e
        latency_ms = (time.monotonic() - start) * 1000

        if result.halted:
            metrics.inc("reviews_rejected")
            metrics.inc("halts_total")
        else:
            metrics.inc("reviews_approved")
            if result.coherence:  # pragma: no branch
                metrics.observe("coherence_score", result.coherence.score)

        audit = request.app.state._state.get("audit")
        if audit:
            audit.log_review(
                query=req.prompt,
                response=result.output,
                approved=not result.halted,
                score=result.coherence.score if result.coherence else 0.0,
                h_logical=result.coherence.h_logical if result.coherence else 0.0,
                h_factual=result.coherence.h_factual if result.coherence else 0.0,
                halt_reason=(
                    result.halt_evidence.reason if result.halt_evidence else ""
                ),
                tenant_id=tenant_id,
                latency_ms=latency_ms,
            )

        output_text = result.output
        if redactor and hasattr(redactor, "enabled") and redactor.enabled:
            output_text = redactor.redact(output_text)

        return ProcessResponse(
            output=output_text,
            coherence=result.coherence.score if result.coherence else None,
            halted=result.halted,
            candidates_evaluated=result.candidates_evaluated,
            warning=result.coherence.warning if result.coherence else False,
            fallback_used=result.fallback_used,
            evidence=_evidence_to_dict(
                result.coherence.evidence if result.coherence else None,
            ),
            halt_evidence=_halt_evidence_to_dict(result.halt_evidence),
        )

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Batch Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.post("/v1/batch", response_model=BatchResponse)
    async def batch(
        req: BatchRequest,
        request: Request,
    ) -> BatchResponse:
        """Process a batch of prompts through the active pipeline."""
        # Per-item size limits (same as single-item endpoints)
        for i, p in enumerate(req.prompts):
            if len(p) > _MAX_PROMPT_CHARS:
                raise HTTPException(
                    422,
                    f"prompts[{i}] exceeds {_MAX_PROMPT_CHARS} char limit",
                )
        for i, r in enumerate(req.responses):
            if len(r) > _MAX_RESPONSE_CHARS:
                raise HTTPException(
                    422,
                    f"responses[{i}] exceeds {_MAX_RESPONSE_CHARS} char limit",
                )

        sanitizer = request.app.state._state.get("sanitizer")
        if sanitizer:
            for p in req.prompts:
                check = sanitizer.check(p)
                if check.blocked:
                    raise HTTPException(
                        400,
                        f"Prompt injection rejected: {check.reason}",
                    )

        batcher = request.app.state._state.get("batch")
        if not batcher:  # pragma: no cover Ă˘â‚¬â€ť lifespan always sets batch
            raise HTTPException(503, "Server not ready")

        redactor = request.app.state._state.get("redactor")
        if redactor and hasattr(redactor, "enabled") and redactor.enabled:
            req.prompts = [redactor.redact(p) for p in req.prompts]
            if req.responses:
                req.responses = [redactor.redact(r) for r in req.responses]

        tenant_id = getattr(
            request.state,
            "tenant_id",
            request.headers.get("X-Tenant-ID", ""),
        )
        if tenant_id:
            logger.info(
                "Tenant access: tenant=%s src=%s path=%s",
                tenant_id,
                request.client.host if request.client else "unknown",
                request.url.path,
            )

        results = []
        try:
            import time

            start_t = time.monotonic()
            if req.task == "review":
                if len(req.prompts) != len(req.responses):
                    raise HTTPException(
                        422,
                        f"review requires equal prompts ({len(req.prompts)}) "
                        f"and responses ({len(req.responses)})",
                    )
                pairs = [
                    (p, r) if r else (p, "")
                    for p, r in zip(req.prompts, req.responses, strict=True)
                ]
                batch_res = await batcher.review_batch_async(pairs, tenant_id=tenant_id)
            else:
                batch_res = await batcher.process_batch_async(
                    req.prompts,
                    tenant_id=tenant_id,
                )
            duration = time.monotonic() - start_t

            from director_ai.core.types import ReviewResult

            for idx, item in enumerate(batch_res.results):
                if isinstance(item, tuple):  # review
                    appr, sc = item
                    results.append(
                        {
                            "index": idx,
                            "approved": appr,
                            "score": sc.score,
                        },
                    )
                elif isinstance(item, ReviewResult):  # process
                    score_val = item.coherence.score if item.coherence else 0.0
                    output = item.output
                    if redactor and hasattr(redactor, "enabled") and redactor.enabled:
                        output = redactor.redact(output)
                    results.append(
                        {
                            "index": idx,
                            "output": output,
                            "approved": not item.halted,
                            "score": score_val,
                        },
                    )

            return BatchResponse(
                results=results,  # type: ignore
                total=batch_res.total,
                succeeded=batch_res.succeeded,
                failed=batch_res.failed,
                duration_seconds=duration,
                errors=[{"index": e[0], "error": e[1]} for e in batch_res.errors],
            )
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.error("Batch processing failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal processing error",
            ) from e

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Tenants Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.get("/v1/tenants", response_model=TenantListResponse)
    async def list_tenants(request: Request):
        router = request.app.state._state.get("tenant_router")
        if not router:
            raise HTTPException(404, "Tenant routing not enabled")
        bound = getattr(request.state, "tenant_id", "")
        visible = [bound] if bound else router.tenant_ids
        return {
            "tenants": [
                {"id": tid, "fact_count": router.fact_count(tid)}
                for tid in visible
                if tid in router.tenant_ids
            ],
        }

    def _enforce_tenant_binding(request: Request, tenant_id: str) -> None:
        bound = getattr(request.state, "tenant_id", "")
        if bound and bound != tenant_id:
            raise HTTPException(403, "API key not authorized for this tenant")

    @app.post("/v1/tenants/{tenant_id}/facts", response_model=StatusResponse)
    async def add_tenant_fact(request: Request, tenant_id: str, req: TenantFactRequest):
        router = request.app.state._state.get("tenant_router")
        if not router:
            raise HTTPException(404, "Tenant routing not enabled")
        _enforce_tenant_binding(request, tenant_id)
        router.add_fact(tenant_id, req.key, req.value)
        return {"status": "ok", "tenant_id": tenant_id, "key": req.key}

    @app.post("/v1/tenants/{tenant_id}/vector-facts", response_model=StatusResponse)
    async def add_tenant_vector_fact(
        request: Request,
        tenant_id: str,
        req: TenantVectorFactRequest,
    ):
        router = request.app.state._state.get("tenant_router")
        if not router:
            raise HTTPException(404, "Tenant routing not enabled")
        _enforce_tenant_binding(request, tenant_id)
        try:
            store = router.get_vector_store(tenant_id, backend_type=req.backend_type)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, f"Invalid backend_type: {exc}") from exc
        store.add_fact(req.key, req.value)
        return {
            "status": "ok",
            "tenant_id": tenant_id,
            "key": req.key,
            "backend_type": req.backend_type,
            "count": store.backend.count(),
        }

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Sessions Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    async def get_session(request: Request, session_id: str):
        caller_hash = getattr(request.state, "api_key_hash", "")
        async with request.app.state._state["sessions_lock"]:
            sessions = request.app.state._state["sessions"]
            owners = request.app.state._state["session_owners"]
            if session_id not in sessions:
                raise HTTPException(404, "Session not found")
            owner = owners.get(session_id, "")
            if owner and owner != caller_hash:
                raise HTTPException(404, "Session not found")
            s = sessions[session_id]
        return {
            "session_id": s.session_id,
            "turn_count": len(s),
            "turns": [
                {
                    "prompt": t.prompt,
                    "response": t.response,
                    "score": t.score,
                    "turn_index": t.turn_index,
                }
                for t in s.turns
            ],
        }

    @app.delete("/v1/sessions/{session_id}", response_model=DeletedResponse)
    async def delete_session(request: Request, session_id: str):
        caller_hash = getattr(request.state, "api_key_hash", "")
        async with request.app.state._state["sessions_lock"]:
            sessions = request.app.state._state["sessions"]
            owners = request.app.state._state["session_owners"]
            if session_id not in sessions:
                raise HTTPException(404, "Session not found")
            owner = owners.get(session_id, "")
            if owner and owner != caller_hash:
                raise HTTPException(404, "Session not found")
            del sessions[session_id]
            owners.pop(session_id, None)
        return {"status": "deleted", "session_id": session_id}

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Metrics Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.get("/v1/metrics")
    async def get_metrics(request: Request):
        return metrics.get_metrics()

    @app.get("/v1/metrics/prometheus", response_class=PlainTextResponse)
    async def get_prometheus(request: Request):
        return metrics.prometheus_format()

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Config Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.get("/v1/config", response_model=ConfigResponse)
    async def get_config():
        return ConfigResponse(config=cfg.to_dict())

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Stats / Dashboard Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    def _prometheus_summary() -> dict:
        """Derive summary from MetricsCollector when stats_backend=prometheus."""
        m = metrics.get_metrics()
        counters = m.get("counters", {})
        hists = m.get("histograms", {})
        total = counters.get("reviews_total", {}).get("total", 0)
        approved = counters.get("reviews_approved", {}).get("total", 0)
        rejected = counters.get("reviews_rejected", {}).get("total", 0)
        halted = counters.get("halts_total", {}).get("total", 0)
        score_hist = hists.get("coherence_score", {})
        duration_hist = hists.get("review_duration_seconds", {})
        avg_score = round(score_hist["mean"], 4) if score_hist.get("count") else None
        avg_latency = (
            round(duration_hist["mean"] * 1000, 1)
            if duration_hist.get("count")
            else None
        )
        return {
            "total": int(total),
            "approved": int(approved),
            "rejected": int(rejected),
            "halted": int(halted),
            "avg_score": avg_score,
            "avg_latency_ms": avg_latency,
        }

    @app.get("/v1/stats", response_model=StatsResponse)
    async def get_stats(request: Request):
        stats_store = request.app.state._state.get("stats")
        if stats_store:
            return stats_store.summary()
        return _prometheus_summary()

    @app.get("/v1/stats/hourly")
    async def get_stats_hourly(request: Request, days: int = 7):
        stats_store = request.app.state._state.get("stats")
        if stats_store:
            result = stats_store.hourly_breakdown(days=days)
            if isinstance(result, list):
                return {"data": result}
            return result
        return {
            "data": [],
            "note": "hourly breakdown requires stats_backend=sqlite",
        }

    @app.get("/v1/dashboard", response_class=PlainTextResponse)
    async def dashboard(request: Request):
        stats_store = request.app.state._state.get("stats")
        s = stats_store.summary() if stats_store else _prometheus_summary()
        approval_rate = (
            f"{s['approved'] / s['total'] * 100:.1f}%" if s["total"] else "N/A"
        )
        return (
            "<!DOCTYPE html><html><head><title>Director-AI Dashboard</title>"
            "<style>body{font-family:monospace;max-width:600px;margin:40px auto;}"
            "table{border-collapse:collapse;width:100%;}td,th{border:1px solid #ccc;"
            "padding:8px;text-align:left;}</style></head><body>"
            "<h1>Director-AI Dashboard</h1>"
            "<table>"
            f"<tr><th>Total Reviews</th><td>{s['total']}</td></tr>"
            f"<tr><th>Approved</th><td>{s['approved']}</td></tr>"
            f"<tr><th>Rejected</th><td>{s['rejected']}</td></tr>"
            f"<tr><th>Halted</th><td>{s['halted']}</td></tr>"
            f"<tr><th>Approval Rate</th><td>{approval_rate}</td></tr>"
            f"<tr><th>Avg Score</th><td>{s['avg_score'] or 'N/A'}</td></tr>"
            f"<tr><th>Avg Latency</th><td>{s['avg_latency_ms'] or 'N/A'} ms</td></tr>"
            "</table></body></html>"
        )

    # -- Compliance endpoints (EU AI Act Article 15) --------------------

    @app.get("/v1/compliance/report", response_model=ComplianceReportResponse)
    async def compliance_report(
        request: Request,
        since: float | None = None,
        until: float | None = None,
        model: str | None = None,
        domain: str | None = None,
        fmt: str = "json",
    ):
        reporter = request.app.state._state.get("compliance_reporter")
        if reporter is None:
            raise HTTPException(
                503,
                "Compliance reporting not configured. Set DIRECTOR_COMPLIANCE_DB_PATH.",
            )
        report = reporter.generate_report(
            since=since, until=until, model=model, domain=domain
        )
        if fmt == "md":
            return PlainTextResponse(report.to_markdown(), media_type="text/markdown")
        return {
            "report_timestamp": report.report_timestamp,
            "period_start": report.period_start,
            "period_end": report.period_end,
            "total_interactions": report.total_interactions,
            "overall_hallucination_rate": report.overall_hallucination_rate,
            "overall_hallucination_rate_ci": report.overall_hallucination_rate_ci,
            "avg_score": report.avg_score,
            "avg_verdict_confidence": report.avg_verdict_confidence,
            "avg_latency_ms": report.avg_latency_ms,
            "human_override_count": report.human_override_count,
            "human_override_rate": report.human_override_rate,
            "model_metrics": [
                {
                    "model": m.model,
                    "total_requests": m.total_requests,
                    "hallucination_rate": m.hallucination_rate,
                    "hallucination_rate_ci": m.hallucination_rate_ci,
                    "avg_score": m.avg_score,
                    "avg_confidence": m.avg_confidence,
                    "avg_latency_ms": m.avg_latency_ms,
                }
                for m in report.model_metrics
            ],
            "drift_detected": report.drift_detected,
            "drift_severity": report.drift_severity,
            "incident_count": report.incident_count,
        }

    @app.get("/v1/compliance/drift", response_model=DriftResponse)
    async def compliance_drift(request: Request):
        detector = request.app.state._state.get("compliance_drift")
        if detector is None:
            raise HTTPException(
                503,
                "Compliance reporting not configured. Set DIRECTOR_COMPLIANCE_DB_PATH.",
            )
        result = detector.analyze()
        return {
            "detected": result.detected,
            "severity": result.severity,
            "z_score": result.z_score,
            "p_value": result.p_value,
            "rate_change": result.rate_change,
            "windows": [
                {
                    "start": w.start,
                    "end": w.end,
                    "total": w.total,
                    "rejected": w.rejected,
                    "hallucination_rate": w.hallucination_rate,
                }
                for w in result.windows
            ],
        }

    @app.get("/v1/compliance/dashboard")
    async def compliance_dashboard(request: Request):
        reporter = request.app.state._state.get("compliance_reporter")
        if reporter is None:
            raise HTTPException(
                503,
                "Compliance reporting not configured. Set DIRECTOR_COMPLIANCE_DB_PATH.",
            )
        now = time.time()
        r_24h = reporter.generate_report(since=now - 86400, until=now)
        r_7d = reporter.generate_report(since=now - 7 * 86400, until=now)
        r_30d = reporter.generate_report(since=now - 30 * 86400, until=now)
        return {
            "24h": {
                "total": r_24h.total_interactions,
                "hallucination_rate": r_24h.overall_hallucination_rate,
                "avg_score": r_24h.avg_score,
            },
            "7d": {
                "total": r_7d.total_interactions,
                "hallucination_rate": r_7d.overall_hallucination_rate,
                "avg_score": r_7d.avg_score,
            },
            "30d": {
                "total": r_30d.total_interactions,
                "hallucination_rate": r_30d.overall_hallucination_rate,
                "avg_score": r_30d.avg_score,
            },
        }

    # -- Gem endpoints (Phase 5 verification & analysis) -----------------

    @app.post("/v1/verify/numeric", response_model=NumericVerifyResponse)
    async def verify_numeric_endpoint(req: TextRequest):
        """Verify numeric consistency in text.

        Checks percentage arithmetic, date logic, probability bounds,
        order-of-magnitude sanity, and internal number consistency.
        """
        from .core.verification.numeric_verifier import verify_numeric

        result = verify_numeric(req.text)
        return NumericVerifyResponse(
            claims_found=result.claims_found,
            issues=[
                NumericIssueResponse(
                    issue_type=i.issue_type,
                    description=i.description,
                    severity=i.severity,
                    context=i.context,
                )
                for i in result.issues
            ],
            valid=result.valid,
            error_count=result.error_count,
            warning_count=result.warning_count,
        )

    @app.post("/v1/verify/reasoning", response_model=ReasoningVerifyResponse)
    async def verify_reasoning_endpoint(req: TextRequest):
        """Verify logical structure of a reasoning chain.

        Extracts reasoning steps and checks each follows from its
        premises. Detects non-sequiturs, circular reasoning, and
        unsupported leaps.
        """
        from .core.verification.reasoning_verifier import verify_reasoning_chain

        result = verify_reasoning_chain(req.text)
        return ReasoningVerifyResponse(
            steps_found=result.steps_found,
            verdicts=[
                ReasoningVerdictResponse(
                    step_index=v.step_index,
                    step_text=v.step_text,
                    verdict=v.verdict,
                    confidence=v.confidence,
                    reason=v.reason,
                    premise_text=v.premise_text,
                )
                for v in result.verdicts
            ],
            chain_valid=result.chain_valid,
            issues_found=result.issues_found,
        )

    @app.post("/v1/temporal-freshness", response_model=FreshnessResponse)
    async def temporal_freshness_endpoint(req: TextRequest):
        """Score temporal freshness of claims in text.

        Detects date-sensitive entities (positions, prices, statistics)
        and assesses staleness risk based on entity type.
        """
        from .core.scoring.temporal_freshness import score_temporal_freshness

        result = score_temporal_freshness(req.text)
        return FreshnessResponse(
            claims=[
                FreshnessClaimResponse(
                    text=c.text,
                    claim_type=c.claim_type,
                    staleness_risk=c.staleness_risk,
                    reason=c.reason,
                )
                for c in result.claims
            ],
            overall_staleness_risk=result.overall_staleness_risk,
            has_temporal_claims=result.has_temporal_claims,
            stale_claim_count=len(result.stale_claims),
        )

    @app.post("/v1/consensus", response_model=ConsensusResponse)
    async def consensus_endpoint(req: ConsensusRequest):
        """Score factual agreement across pre-generated model responses.

        Accepts responses from multiple models and computes pairwise
        agreement using Jaccard word overlap.
        """
        from .core.scoring.consensus import ConsensusScorer, ModelResponse

        scorer = ConsensusScorer(
            models=[r.model for r in req.responses],
            generate_fn=None,
        )
        model_responses = [
            ModelResponse(model=r.model, response=r.response) for r in req.responses
        ]
        result = scorer.score_responses(model_responses)
        return ConsensusResponse(
            responses=[
                ConsensusResponseItem(model=r.model, response=r.response)
                for r in result.responses
            ],
            pairs=[
                PairwiseAgreementResponse(
                    model_a=p.model_a,
                    model_b=p.model_b,
                    divergence=p.divergence,
                    agreed=p.agreed,
                )
                for p in result.pairs
            ],
            agreement_score=result.agreement_score,
            lowest_pair_agreement=result.lowest_pair_agreement,
            has_consensus=result.has_consensus,
            num_models=result.num_models,
        )

    @app.post("/v1/adversarial/test", response_model=AdversarialResponse)
    async def adversarial_test_endpoint(req: ReviewRequest, request: Request):
        """Run adversarial robustness tests against the guardrail.

        Uses the prompt+response as a baseline, then tests adversarial
        transformations of the response against the scorer.
        """
        from .testing.adversarial_suite import AdversarialTester

        app_scorer = request.app.state._state.get("scorer")
        if app_scorer is None:
            raise HTTPException(503, "Scorer not initialised")

        def review_fn(prompt: str, response: str):
            approved, score = app_scorer.review(prompt, response)
            return approved, score.score

        tester = AdversarialTester(
            review_fn=review_fn,
            prompt=req.prompt,
        )
        report = tester.run()
        return AdversarialResponse(
            total_patterns=report.total_patterns,
            detected=report.detected,
            bypassed=report.bypassed,
            detection_rate=report.detection_rate,
            is_robust=report.is_robust,
            vulnerable_categories=report.vulnerable_categories,
            results=[
                AdversarialPatternResponse(
                    name=r.pattern.name,
                    category=r.pattern.category,
                    transform=r.pattern.transform,
                    detected=r.detected,
                    score=r.score,
                    original_score=r.original_score,
                )
                for r in report.results
            ],
        )

    @app.post("/v1/conformal/predict", response_model=ConformalResponse)
    async def conformal_predict_endpoint(req: ConformalRequest):
        """Compute conformal prediction interval for hallucination probability.

        Optionally calibrate from provided historical data first.
        """
        from .core.calibration.conformal import ConformalPredictor

        predictor = ConformalPredictor(coverage=req.coverage)
        if req.calibration_scores and req.calibration_labels:
            if len(req.calibration_scores) != len(req.calibration_labels):
                raise HTTPException(
                    status_code=422,
                    detail="calibration_scores and calibration_labels must have same length",
                )
            predictor.calibrate(req.calibration_scores, req.calibration_labels)
        interval = predictor.predict(req.score)
        return ConformalResponse(
            point_estimate=interval.point_estimate,
            lower=interval.lower,
            upper=interval.upper,
            coverage=interval.coverage,
            calibration_size=interval.calibration_size,
            is_reliable=interval.is_reliable,
        )

    @app.post("/v1/compliance/feedback-loops", response_model=FeedbackLoopResponse)
    async def feedback_loop_endpoint(req: FeedbackLoopCheckRequest):
        """Check if input text matches any previous AI output (feedback loop).

        Pass previous_outputs to seed the detector buffer, then checks
        the input_text against them.
        """
        from .compliance.feedback_loop_detector import FeedbackLoopDetector

        detector = FeedbackLoopDetector(
            similarity_threshold=req.similarity_threshold,
        )
        for i, output in enumerate(req.previous_outputs):
            detector.record_output(output, float(i))

        alert = detector.check_input(req.input_text)
        if alert is None:
            return FeedbackLoopResponse(
                loop_detected=False,
                similarity=0.0,
            )
        return FeedbackLoopResponse(
            loop_detected=True,
            similarity=alert.similarity,
            severity=alert.severity,
            matched_output=alert.matched_output,
        )

    @app.post("/v1/agentic/check-step", response_model=AgenticStepResponse)
    async def agentic_check_step_endpoint(req: AgenticStepRequest):
        """Evaluate a single agentic step for safety issues.

        Replays step_history to build monitor state, then evaluates
        the current step.
        """
        from .agentic.loop_monitor import LoopMonitor

        monitor = LoopMonitor(
            goal=req.goal,
            max_steps=req.max_steps,
        )
        for prev in req.step_history:
            monitor.check_step(
                action=prev.get("action", ""),
                args=prev.get("args", ""),
            )
        verdict = monitor.check_step(
            action=req.action,
            args=req.args,
            result=req.result,
            tokens=req.tokens,
        )
        return AgenticStepResponse(
            step_number=verdict.step_number,
            should_halt=verdict.should_halt,
            should_warn=verdict.should_warn,
            reasons=verdict.reasons,
            goal_drift_score=verdict.goal_drift_score,
            budget_remaining_pct=verdict.budget_remaining_pct,
        )

    # -- WebSocket streaming (multiplexed) ------------------------------

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        ws_tenant_id = ""
        if cfg.api_keys:
            provided = ws.headers.get("X-API-Key", "")
            if not any(hmac.compare_digest(provided, k) for k in cfg.api_keys):
                await ws.close(code=1008, reason="unauthorized")
                return
            if _api_key_tenant_map:
                ws_tenant_id = _api_key_tenant_map.get(provided, "")
        if not ws_tenant_id:
            ws_tenant_id = ws.headers.get("X-Tenant-ID", "")
        await ws.accept()

        send_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(_WS_MAX_CONCURRENT)
        active_tasks: dict[str, asyncio.Task] = {}

        async def _send(payload: dict) -> None:
            async with send_lock:
                await ws.send_json(payload)

        async def _handle_session(session_id: str, data: dict) -> None:
            prompt = data.get("prompt", "")

            sanitizer = ws.app.state._state.get("sanitizer")
            if sanitizer:
                check = sanitizer.check(prompt)
                if check.blocked:
                    await _send(
                        {
                            "session_id": session_id,
                            "error": f"injection rejected: {check.reason}",
                        },
                    )
                    return

            agent = ws.app.state._state.get("agent")
            if not agent:
                await _send({"session_id": session_id, "error": "server not ready"})
                return

            if data.get("streaming_oversight"):
                try:
                    from .core import StreamingKernel

                    kernel = StreamingKernel(
                        hard_limit=cfg.hard_limit,
                        window_size=getattr(cfg, "window_size", 5),
                        window_threshold=getattr(cfg, "window_threshold", 0.5),
                    )
                    result = await agent.aprocess(prompt, tenant_id=ws_tenant_id)
                    coherence = result.coherence.score if result.coherence else 0.0
                    halted = kernel.check_halt(coherence)
                    halt_reason = "hard_limit" if halted else None
                    msg = {
                        "session_id": session_id,
                        "type": "halt" if halted else "result",
                        "output": result.output,
                        "coherence": round(coherence, 4),
                        "halted": halted,
                    }
                    if halt_reason:
                        msg["reason"] = halt_reason
                    await _send(msg)
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                    OSError,
                ) as exc:  # pragma: no cover
                    logger.error("WebSocket streaming failed: %s", exc)
                    await _send(
                        {"session_id": session_id, "error": "streaming failed"},
                    )
                return

            try:
                result = await agent.aprocess(prompt, tenant_id=ws_tenant_id)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                logger.error("WebSocket agent.process() failed: %s", exc)
                await _send(
                    {"session_id": session_id, "error": "processing failed"},
                )
                return
            await _send(
                {
                    "session_id": session_id,
                    "type": "result",
                    "output": result.output,
                    "coherence": (result.coherence.score if result.coherence else None),
                    "halted": result.halted,
                    "warning": (
                        result.coherence.warning if result.coherence else False
                    ),
                    "fallback_used": result.fallback_used,
                    "evidence": _evidence_to_dict(
                        result.coherence.evidence if result.coherence else None,
                    ),
                    "halt_evidence": _halt_evidence_to_dict(result.halt_evidence),
                },
            )

        async def _run_session(session_id: str, data: dict) -> None:
            async with semaphore:
                try:
                    await _handle_session(session_id, data)
                finally:
                    active_tasks.pop(session_id, None)

        try:
            while True:
                try:
                    data = await ws.receive_json()
                except (ValueError, KeyError) as exc:
                    logger.warning("WebSocket bad JSON: %s", exc)
                    await _send({"error": "invalid JSON"})
                    continue

                if not isinstance(data, dict):
                    await _send({"error": "expected JSON object"})
                    continue

                # Cancel action
                action = data.get("action", "")
                if action == "cancel":
                    cancel_sid = data.get("session_id", "")
                    task = active_tasks.get(cancel_sid)
                    if task and not task.done():
                        task.cancel()
                        active_tasks.pop(cancel_sid, None)
                    await _send({"session_id": cancel_sid, "type": "cancelled"})
                    continue

                prompt = data.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    await _send({"error": "prompt must be a non-empty string"})
                    continue

                if len(prompt) > _WS_MAX_PROMPT_LENGTH:
                    await _send(
                        {"error": f"prompt exceeds {_WS_MAX_PROMPT_LENGTH} chars"},
                    )
                    continue

                session_id = data.get("session_id") or str(uuid.uuid4())
                task = asyncio.create_task(_run_session(session_id, data))
                active_tasks[session_id] = task

        except WebSocketDisconnect:
            for task in active_tasks.values():
                task.cancel()

    return app
