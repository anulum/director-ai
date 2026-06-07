# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FastAPI Server

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
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from .core.config import DirectorConfig
from .core.kb_write_security import (
    KBWriteAccessError,
    canonical_kb_payload,
    check_kb_write_access,
    parse_hmac_keys,
    verify_kb_payload_signature,
)
from .core.metrics import metrics

__all__ = [
    "AdversarialPatternResponse",
    "AdversarialResponse",
    "AgenticStepRequest",
    "AgenticStepResponse",
    "BatchRequest",
    "BatchResponse",
    "ComplianceDashboardResponse",
    "ComplianceReportResponse",
    "ConfigResponse",
    "ConformalRequest",
    "ConformalResponse",
    "ConsensusRequest",
    "ConsensusResponse",
    "ConsensusResponseItem",
    "DeletedResponse",
    "DriftResponse",
    "FeedbackCalibrationResponse",
    "FeedbackLoopCheckRequest",
    "FeedbackLoopResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "FreshnessClaimResponse",
    "FreshnessResponse",
    "FreshnessStatusResponse",
    "HealthResponse",
    "HourlyDataPoint",
    "HourlyResponse",
    "InjectionClaimResponse",
    "InjectionRequest",
    "InjectionResponse",
    "ModelMetricsResponse",
    "MultimodalDetectRequest",
    "MultimodalDetectResponse",
    "NumericIssueResponse",
    "NumericVerifyResponse",
    "PairwiseAgreementResponse",
    "PeriodMetrics",
    "ProcessRequest",
    "ProcessResponse",
    "ReadyResponse",
    "ReasoningVerdictResponse",
    "ReasoningVerifyResponse",
    "ReviewRequest",
    "ReviewResponse",
    "SessionResponse",
    "SourceResponse",
    "StatsResponse",
    "StatusResponse",
    "TenantFactRequest",
    "TenantInfo",
    "TenantListResponse",
    "TenantVectorFactRequest",
    "TextRequest",
    "TurnInfo",
    "VerifyResponse",
    "WindowStats",
    "create_app",
]

REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="",
)
_REQUEST_ID_MAX_LENGTH = 128
_REQUEST_ID_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
)

logger = logging.getLogger("DirectorAI.Server")

_WS_MAX_PROMPT_LENGTH = 100_000
_WS_MAX_CONCURRENT = 8
# Denial-of-service controls for the WebSocket streaming endpoint.
_WS_MAX_CONNECTIONS = 256  # global concurrent connections per process
_WS_MAX_CONNECTIONS_PER_IP = 16  # concurrent connections from one client IP
_WS_IDLE_TIMEOUT_S = 300.0  # close a connection idle this long between messages
_WS_MAX_LIFETIME_S = 3600.0  # close a connection older than this
_WS_RATE_WINDOW_S = 10.0  # sliding window for the per-connection message rate
_WS_MAX_MSGS_PER_WINDOW = 60  # messages allowed per window before rate limiting
_WS_CONN_CHAR_BUDGET = 5_000_000  # total prompt chars one connection may submit
_AUTH_EXEMPT_PATHS_BASE = frozenset(
    {"/v1/live", "/v1/health", "/v1/ready", "/v1/source"}
)

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

# slowapi is optional — declare as Any up front so the
# except-branch assignment does not conflict with mypy-inferred
# module / class types from the imports above.
slowapi: Any
Limiter: Any
try:
    import slowapi as _slowapi_mod
    from slowapi import Limiter as _Limiter
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    slowapi = _slowapi_mod
    Limiter = _Limiter
    _SLOWAPI_AVAILABLE = True
except ImportError:
    slowapi = None
    Limiter = None
    _SLOWAPI_AVAILABLE = False


def _check_fastapi() -> None:
    """Raise an install hint when FastAPI server extras are unavailable."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )


def _extract_request_api_key(request: Request) -> str:
    """Return the caller API key from supported production auth headers."""
    x_api_key = request.headers.get("X-API-Key", "").strip()
    if x_api_key:
        return x_api_key

    scheme, _, token = request.headers.get("Authorization", "").partition(" ")
    if scheme.lower() == "bearer":
        return token.strip()
    return ""


# Pydantic models extracted to _server_models.py (reduce module size)
_MAX_PROMPT_CHARS = 100_000
_MAX_RESPONSE_CHARS = 500_000

if _FASTAPI_AVAILABLE:  # pragma: no branch
    from ._server_helpers import (
        evidence_to_dict as _evidence_to_dict,
    )
    from ._server_helpers import (
        halt_evidence_to_dict as _halt_evidence_to_dict,
    )
    from ._server_models import (
        AdversarialPatternResponse,
        AdversarialResponse,
        AgenticStepRequest,
        AgenticStepResponse,
        BatchRequest,
        BatchResponse,
        ComplianceDashboardResponse,
        ComplianceReportResponse,
        ConfigResponse,
        ConformalRequest,
        ConformalResponse,
        ConsensusRequest,
        ConsensusResponse,
        ConsensusResponseItem,
        DeletedResponse,
        DriftResponse,
        FeedbackCalibrationResponse,
        FeedbackLoopCheckRequest,
        FeedbackLoopResponse,
        FeedbackRequest,
        FeedbackResponse,
        FreshnessClaimResponse,
        FreshnessResponse,
        FreshnessStatusResponse,
        HealthResponse,
        HourlyDataPoint,
        HourlyResponse,
        InjectionClaimResponse,
        InjectionRequest,
        InjectionResponse,
        ModelMetricsResponse,
        MultimodalDetectRequest,
        MultimodalDetectResponse,
        NumericIssueResponse,
        NumericVerifyResponse,
        PairwiseAgreementResponse,
        PeriodMetrics,
        ProcessRequest,
        ProcessResponse,
        ReadyResponse,
        ReasoningVerdictResponse,
        ReasoningVerifyResponse,
        ReviewRequest,
        ReviewResponse,
        SessionResponse,
        SourceResponse,
        StatsResponse,
        StatusResponse,
        TenantFactRequest,
        TenantInfo,
        TenantListResponse,
        TenantVectorFactRequest,
        TextRequest,
        TurnInfo,
        VerifyResponse,
        WindowStats,
    )


def _record_sector_policy_findings(
    *,
    policy: str,
    report: Any,
    source: str,
) -> None:
    """Record tenant-safe sector-policy finding metrics."""

    for finding in getattr(report, "findings", ()):
        metrics.inc_labeled(
            "sector_policy_findings_total",
            {
                "policy": policy,
                "source": source,
                "code": finding.code,
                "severity": finding.severity,
                "action": finding.action,
            },
        )


def _can_suppress_batcher_metrics(batcher: Any) -> bool:
    """Return true when the batcher supports endpoint-owned metrics."""

    from .core.runtime.batch import BatchProcessor

    return isinstance(batcher, BatchProcessor)


def _http_endpoint_label(request: Request) -> str:
    """Return a low-cardinality route label for HTTP metrics."""

    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path

    try:
        from starlette.routing import Match
    except ImportError:  # pragma: no cover - FastAPI depends on Starlette
        return "__unmatched__"

    partial_path = ""
    for candidate in request.app.routes:
        matches = getattr(candidate, "matches", None)
        if not callable(matches):
            continue
        match, _child_scope = matches(request.scope)
        candidate_path = getattr(candidate, "path", "")
        if not isinstance(candidate_path, str) or not candidate_path:
            continue
        if match is Match.FULL:
            return candidate_path
        if match is Match.PARTIAL and not partial_path:
            partial_path = candidate_path

    return partial_path or "__unmatched__"


def _record_http_metrics(
    request: Request,
    *,
    status_code: int,
    started_at: float,
) -> None:
    elapsed = time.monotonic() - started_at
    metrics.observe("http_request_duration_seconds", elapsed)
    metrics.inc_labeled(
        "http_requests_total",
        {
            "method": request.method,
            "endpoint": _http_endpoint_label(request),
            "status": str(status_code),
        },
    )


def _normalize_request_id(raw: str | None) -> str:
    if (
        raw
        and len(raw) <= _REQUEST_ID_MAX_LENGTH
        and all(char in _REQUEST_ID_ALLOWED_CHARS for char in raw)
    ):
        return raw
    return str(uuid.uuid4())


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
        agent_kwargs: dict = {
            "_scorer": scorer,
            "_store": store,
            "production_mode": cfg.production_mode,
            "llm_max_tokens": cfg.llm_max_tokens,
            "llm_temperature": cfg.llm_temperature,
        }
        if cfg.llm_provider == "local":
            agent_kwargs["llm_api_url"] = cfg.llm_api_url
        elif cfg.llm_provider in ("openai", "anthropic"):
            agent_kwargs["provider"] = cfg.llm_provider
            if cfg.llm_api_key:
                agent_kwargs["api_key"] = cfg.llm_api_key
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

        if cfg.feedback_db_path:
            from .core.calibration.feedback_store import FeedbackStore

            feedback_store = FeedbackStore(cfg.feedback_db_path)
            app.state._state["feedback_store"] = feedback_store
            logger.info("Feedback store: %s", cfg.feedback_db_path)

        if cfg.tenant_routing:
            app.state._state["tenant_router"] = TenantRouter()
            logger.info("Tenant routing enabled")

        from .core.retrieval.doc_registry import DocRegistry

        app.state._state["doc_registry"] = DocRegistry()

        # Multi-modal hallucination guard: opt-in and isolated. Only stood up
        # when the experimental hooks flag is set AND a modality is configured,
        # so the default safety posture is unchanged.
        if cfg.multimodal_enabled_modalities:
            from .experimental import experimental_hooks_enabled

            if experimental_hooks_enabled():
                from .core.multimodal_guard import build_hashbag_adapter

                app.state._state["multimodal_adapter"] = build_hashbag_adapter(
                    enabled_modalities=cfg.multimodal_enabled_modalities,
                    benchmarked_modalities=cfg.multimodal_benchmarked_modalities,
                    dim=cfg.multimodal_embedding_dim,
                    hallucination_threshold=cfg.multimodal_hallucination_threshold,
                    consistency_threshold=cfg.multimodal_consistency_threshold,
                    temporal_alpha=cfg.multimodal_temporal_alpha,
                    temporal_floor=cfg.multimodal_temporal_floor,
                    grounding_floor=cfg.multimodal_grounding_floor,
                    grounding_allow_threshold=cfg.multimodal_grounding_allow_threshold,
                )
                logger.info(
                    "Multimodal guard enabled (modalities=%s)",
                    cfg.multimodal_enabled_modalities,
                )

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
        feedback_shutdown = app.state._state.get("feedback_store")
        if feedback_shutdown is not None:
            feedback_shutdown.close()

    import director_ai

    app = FastAPI(
        title="Director-Class AI",
        version=director_ai.__version__,
        description="Real-time multi-agent orchestration and coherence scoring.",
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.router_mounts = {}

    # Fine-tuning API router (Phase C)
    try:
        from .finetune_api import create_finetune_router

        app.include_router(create_finetune_router(), prefix="/v1/finetune")
        app.state.router_mounts["finetune"] = "mounted"
    except ImportError as exc:
        app.state.router_mounts["finetune"] = f"unavailable:{exc}"
        if cfg.production_mode:
            raise RuntimeError(
                "production_mode requires the fine-tuning API router to load"
            ) from exc

    # Knowledge ingestion API
    try:
        from .knowledge_api import create_knowledge_router

        knowledge_router = create_knowledge_router()
        app.include_router(knowledge_router, prefix="/v1/knowledge")
        app.include_router(knowledge_router, prefix="/api/v1/knowledge")
        app.state.router_mounts["knowledge"] = "mounted"
    except ImportError as exc:
        app.state.router_mounts["knowledge"] = f"unavailable:{exc}"
        if cfg.production_mode:
            raise RuntimeError(
                "production_mode requires the knowledge API router to load"
            ) from exc

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
            "X-KB-Key-ID",
            "X-KB-Signature",
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
                """Render SlowAPI rate-limit failures as JSON responses."""
                return JSONResponse(  # pragma: no cover Ă˘â‚¬â€ť ASGI runtime handler
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )

    # Middleware: correlation IDs + API key auth + metrics

    _auth_exempt = (
        _AUTH_EXEMPT_PATHS_BASE
        if cfg.metrics_require_auth
        else _AUTH_EXEMPT_PATHS_BASE | {"/v1/metrics/prometheus"}
    )

    _api_key_tenant_map: dict[str, str] = {}
    if cfg.api_key_tenant_map:
        _api_key_tenant_map = _json_mod.loads(cfg.api_key_tenant_map)

    # Effective auth keys = explicit api_keys ∪ tenant-map keys. Enforcement
    # must consider both: a map-only config (e.g. the production profile with a
    # key→tenant binding but no separate api_keys list) still requires a valid
    # key. Keying enforcement on cfg.api_keys alone is fail-open.
    _valid_api_keys: list[str] = list(cfg.api_keys)
    for _bound_key in _api_key_tenant_map:
        if _bound_key not in _valid_api_keys:
            _valid_api_keys.append(_bound_key)
    if cfg.production_mode and not _valid_api_keys:
        raise RuntimeError(
            "production_mode requires at least one effective API key "
            "(set api_keys or api_key_tenant_map)"
        )

    def _request_authenticated(request: Request) -> bool:
        """Return True when the caller is authenticated for detail disclosure.

        Auth-exempt probes (`/v1/health`, `/v1/source`) still answer to
        unauthenticated callers, but the detailed payload (version, mode,
        profile, routers, revision health) is only returned to a valid key
        holder. When no API keys are configured there is no auth posture
        (dev server), so detail is returned to keep local debugging usable.
        """
        if not _valid_api_keys:
            return True
        provided = _extract_request_api_key(request)
        return any(hmac.compare_digest(provided, k) for k in _valid_api_keys)

    @app.middleware("http")
    async def _http_middleware(request: Request, call_next):
        """Apply request IDs, API-key auth, tenant binding, and metrics."""
        request_id = _normalize_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        REQUEST_ID_CTX.set(request_id)

        start = time.monotonic()
        api_key_hash = ""
        if _valid_api_keys and request.url.path not in _auth_exempt:
            provided = _extract_request_api_key(request)
            # Constant-time: always compare against ALL keys to prevent
            # timing side-channels that leak key position.
            key_valid = False
            for k in _valid_api_keys:
                if hmac.compare_digest(provided, k):
                    key_valid = True
            if not key_valid:
                logger.warning(
                    "Auth failed from %s on %s",
                    request.client.host if request.client else "unknown",
                    request.url.path,
                )
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                    headers={"X-Request-ID": request_id},
                )
                _record_http_metrics(
                    request,
                    status_code=response.status_code,
                    started_at=start,
                )
                return response
            import hashlib

            from .core.safety.audit_salt import get_audit_salt

            # Salted truncated SHA-256 fingerprint for audit logs only — NOT
            # used for authentication or password storage. The API key is
            # verified via constant-time HMAC comparison above. Salt is
            # per-installation (VULN-DAI-003) so a leaked log from one
            # deployment cannot be replayed against fingerprints from another.
            api_key_hash = hashlib.sha256(
                get_audit_salt() + provided.encode(),
            ).hexdigest()[:16]

            # Tenant binding: enforce API key Ă˘â€ â€™ tenant mapping if configured
            if _api_key_tenant_map:
                if provided not in _api_key_tenant_map:
                    response = JSONResponse(
                        status_code=403,
                        content={"detail": "API key not bound to any tenant"},
                        headers={"X-Request-ID": request_id},
                    )
                    _record_http_metrics(
                        request,
                        status_code=response.status_code,
                        started_at=start,
                    )
                    return response
                bound_tenant = _api_key_tenant_map[provided]
                claimed_tenant = request.headers.get("X-Tenant-ID", "")
                if claimed_tenant and claimed_tenant != bound_tenant:
                    response = JSONResponse(
                        status_code=403,
                        content={"detail": "API key not authorized for this tenant"},
                        headers={"X-Request-ID": request_id},
                    )
                    _record_http_metrics(
                        request,
                        status_code=response.status_code,
                        started_at=start,
                    )
                    return response
                request.state.tenant_id = bound_tenant
                request.state.kb_write_key_ok = True
                request.state.kb_tenant_binding_ok = True
            else:
                # No key→tenant map: accept header but log for audit.
                # Tenant isolation without key binding is advisory only.
                claimed = request.headers.get("X-Tenant-ID", "")
                if claimed:
                    logger.debug(
                        "Unbound tenant claim: %s (api_key=%s)", claimed, api_key_hash
                    )
                request.state.tenant_id = claimed
                request.state.kb_write_key_ok = True
                request.state.kb_tenant_binding_ok = False
        else:
            # No API keys configured — tenant from header is untrusted
            request.state.tenant_id = request.headers.get("X-Tenant-ID", "")
            request.state.kb_write_key_ok = False
            request.state.kb_tenant_binding_ok = False

        request.state.api_key_hash = api_key_hash

        # Metrics
        try:
            response = await call_next(request)
        except Exception:
            _record_http_metrics(
                request,
                status_code=500,
                started_at=start,
            )
            raise
        _record_http_metrics(
            request,
            status_code=response.status_code,
            started_at=start,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Health Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.get("/v1/live")
    async def liveness():
        """Minimal unauthenticated liveness probe — no version or config leak."""
        return {"ok": True}

    @app.get("/v1/health")
    async def health(request: Request):
        """Return liveness plus, for authenticated callers, build detail.

        Unauthenticated callers receive ``{status, license}`` only; a valid key
        unlocks version, mode, profile, router, and revision detail. The
        response is no longer a fixed ``HealthResponse`` schema because the
        payload shape now depends on authentication.
        """
        import director_ai

        lic = getattr(request.app.state, "_license", None)
        extra = {}
        if lic and lic.is_commercial:
            # Public, auth-exempt endpoint: expose the licence type only, never
            # the commercial licensee identity or tier.
            extra = {"license": "commercial"}
        elif lic and lic.is_trial:
            extra = {"license": "trial", "expires": lic.expires}
        else:
            extra = {"license": "agpl"}

        if not _request_authenticated(request):
            # Unauthenticated callers get liveness + licence type only; the
            # detailed build/router/revision payload is gated behind a valid
            # key. Pure liveness probes should use /v1/live.
            return {"status": "ok", **extra}

        resp = HealthResponse(
            version=director_ai.__version__,
            mode=cfg.mode,
            profile=cfg.profile,
            nli_loaded=cfg.use_nli,
            uptime_seconds=time.monotonic() - _start_time,
            routers=dict(request.app.state.router_mounts),
            model_revisions=cfg.model_revision_health(),
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
        """Return source-availability metadata for AGPL or commercial mode."""
        import director_ai

        lic = getattr(request.app.state, "_license", None)
        if lic and lic.is_commercial:
            if not cfg.source_endpoint_enabled:
                raise HTTPException(
                    404, "Source endpoint disabled (commercial license)"
                )
            # AGPL §13 source obligation is waived under a commercial licence,
            # so nothing needs publishing here. Do not leak the commercial tier
            # or the build version on this auth-exempt endpoint.
            return {
                "license": "commercial",
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
        sector_policy_report = None
        if req.sector_policy:
            from .core.financial_services import assess_banking_response

            sector_policy_report = assess_banking_response(
                req.prompt,
                req.response,
                evidence_refs=req.evidence_refs,
                numeric_evidence_refs=req.numeric_evidence_refs,
                policy_refs=req.policy_refs,
                jurisdiction=req.jurisdiction,
                product_line=req.product_line,
                human_review_acknowledged=req.human_review_acknowledged,
            )
            approved = approved and sector_policy_report.approved
            _record_sector_policy_findings(
                policy=req.sector_policy,
                report=sector_policy_report,
                source="review",
            )

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
            sector_policy=(
                sector_policy_report.to_dict() if sector_policy_report else None
            ),
        )

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    async def record_feedback(
        req: FeedbackRequest,
        request: Request,
    ) -> FeedbackResponse:
        """Record a human correction for online calibration."""
        feedback_store = request.app.state._state.get("feedback_store")
        if feedback_store is None:
            raise HTTPException(503, "Feedback store not configured")

        tenant_id = getattr(
            request.state,
            "tenant_id",
            request.headers.get("X-Tenant-ID", ""),
        )
        feedback_store.report(
            prompt=req.prompt,
            response=req.response,
            guardrail_approved=req.guardrail_approved,
            human_approved=req.human_approved,
            guardrail_score=req.guardrail_score,
            domain=req.domain,
            review_id=req.review_id,
            tenant_id=tenant_id,
        )
        metrics.inc("feedback_reports_total")
        if req.guardrail_approved != req.human_approved:
            metrics.inc("feedback_disagreements_total")
        return FeedbackResponse(
            accepted=True,
            correction_count=feedback_store.count(domain=req.domain or None),
            disagreement=req.guardrail_approved != req.human_approved,
            tenant_id=tenant_id,
            review_id=req.review_id,
        )

    @app.get("/v1/feedback/calibration", response_model=FeedbackCalibrationResponse)
    async def feedback_calibration(
        request: Request,
        domain: str = "",
        min_corrections: int = 20,
    ) -> FeedbackCalibrationResponse:
        """Return current online calibration metrics from human feedback."""
        if min_corrections < 1 or min_corrections > 100_000:
            raise HTTPException(400, "min_corrections must be between 1 and 100000")
        feedback_store = request.app.state._state.get("feedback_store")
        if feedback_store is None:
            raise HTTPException(503, "Feedback store not configured")

        from .core.calibration.online_calibrator import OnlineCalibrator

        calibrator = OnlineCalibrator(
            feedback_store,
            min_corrections=min_corrections,
        )
        report = calibrator.calibrate(domain=domain or None)
        return FeedbackCalibrationResponse(
            correction_count=report.correction_count,
            optimal_threshold=report.optimal_threshold,
            current_accuracy=report.current_accuracy,
            tpr=report.tpr,
            tnr=report.tnr,
            fpr=report.fpr,
            fnr=report.fnr,
            fpr_ci=report.fpr_ci,
            fnr_ci=report.fnr_ci,
        )

    # ── Verified Review (atomic multi-span signals) ──────────────────

    @app.post("/v1/verify", response_model=VerifyResponse)
    async def verify_response(req: ReviewRequest, request: Request):
        """Atomic multi-span fact verification.

        Decomposes the response into claims, ranks source spans from the
        KB, aggregates evidence, and checks NLI + entity + number +
        negation + traceability signals. Returns per-claim verdicts with
        confidence and provenance.
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

    # ── Injection Detection (output-side NLI) ─────────────────────

    @app.post("/v1/injection/detect", response_model=InjectionResponse)
    async def detect_injection(req: InjectionRequest, request: Request):
        """Detect prompt injection effects in LLM output via NLI divergence.

        Analyses whether the response diverges from the stated intent
        (system_prompt + user_query).  Returns per-claim attribution
        with grounded/drifted/injected verdicts.
        """
        import asyncio

        from .core.safety.injection import InjectionDetector

        scorer = request.app.state._state.get("scorer")
        nli = getattr(scorer, "_nli", None) if scorer else None

        cfg = request.app.state._state.get("config")

        sanitizer = request.app.state._state.get("sanitizer")

        detector = InjectionDetector(
            nli_scorer=nli,
            sanitizer=sanitizer,
            injection_threshold=getattr(cfg, "injection_threshold", 0.7),
            drift_threshold=getattr(cfg, "injection_drift_threshold", 0.6),
            injection_claim_threshold=getattr(cfg, "injection_claim_threshold", 0.75),
            baseline_divergence=getattr(cfg, "injection_baseline_divergence", 0.4),
            stage1_weight=getattr(cfg, "injection_stage1_weight", 0.3),
            require_model_backed_nli=getattr(
                cfg,
                "injection_require_model_backed_nli",
                False,
            ),
        )

        intent = req.intent or ""
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: detector.detect(
                intent=intent,
                response=req.response,
                user_query=req.user_query,
                system_prompt=req.system_prompt,
            ),
        )
        return result.to_dict()

    # ── Multimodal hallucination guard (opt-in, experimental) ─────

    @app.post("/v1/multimodal/check", response_model=MultimodalDetectResponse)
    async def multimodal_check(req: MultimodalDetectRequest, request: Request):
        """Check a text claim against paired image / audio / video evidence.

        Opt-in and isolated: returns 404 unless the experimental hooks flag is
        set and at least one modality is configured. The response is
        tenant-safe — no raw media, transcript, or claim text is echoed back.
        """
        import asyncio
        import base64
        import binascii

        adapter = request.app.state._state.get("multimodal_adapter")
        if adapter is None:
            raise HTTPException(
                404,
                "multimodal guard is disabled; enable experimental hooks and "
                "configure multimodal_enabled_modalities",
            )

        from .core.guard_control import RiskEnvelope
        from .core.multimodal_guard import MultimodalCheckRequest

        cfg = request.app.state._state.get("config")
        image_bytes = b""
        if req.image_base64:
            try:
                image_bytes = base64.b64decode(req.image_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(400, "image_base64 is not valid base64") from exc

        try:
            check_request = MultimodalCheckRequest(
                modality=req.modality,
                claim_text=req.claim_text,
                media_ref=req.media_ref,
                image_bytes=image_bytes,
                transcript_text=req.transcript_text,
                frame_similarities=req.frame_similarities,
                caption_text=req.caption_text,
                metadata=req.metadata,
            )
            risk_envelope = RiskEnvelope(
                action_category="multimodal",
                reversibility="reversible",
                domain="general",
                calibrated_threshold=getattr(
                    cfg, "multimodal_calibrated_threshold", 0.5
                ),
                no_go_threshold=getattr(cfg, "multimodal_no_go_threshold", 0.9),
            )
            policy_id = getattr(cfg, "multimodal_policy_id", "multimodal-default")
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: adapter.check(
                    check_request,
                    risk_envelope=risk_envelope,
                    policy_id=policy_id,
                ),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return result.to_dict()

    # ── Process ───────────────────────────────────────────────────

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
        if req.sector_policy and req.task != "review":
            raise HTTPException(
                422,
                "sector_policy is only supported for review batches",
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

        results: list[dict[str, Any]] = []
        try:
            import time

            start_t = time.monotonic()
            pairs: list[tuple[str, str]] = []
            suppress_batcher_metrics = _can_suppress_batcher_metrics(batcher)
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
                if suppress_batcher_metrics:
                    batch_res = await batcher.review_batch_async(
                        pairs,
                        tenant_id=tenant_id,
                        record_metrics=False,
                    )
                else:
                    batch_res = await batcher.review_batch_async(
                        pairs,
                        tenant_id=tenant_id,
                    )
            else:
                if suppress_batcher_metrics:
                    batch_res = await batcher.process_batch_async(
                        req.prompts,
                        tenant_id=tenant_id,
                        record_metrics=False,
                    )
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
                    approved = appr
                    result: dict[str, Any] = {
                        "index": idx,
                        "approved": approved,
                        "score": sc.score,
                    }
                    if req.sector_policy and idx < len(pairs):
                        from .core.financial_services import assess_banking_response

                        sector_policy_report = assess_banking_response(
                            pairs[idx][0],
                            pairs[idx][1],
                            evidence_refs=req.evidence_refs,
                            numeric_evidence_refs=req.numeric_evidence_refs,
                            policy_refs=req.policy_refs,
                            jurisdiction=req.jurisdiction,
                            product_line=req.product_line,
                            human_review_acknowledged=req.human_review_acknowledged,
                        )
                        approved = approved and sector_policy_report.approved
                        result["approved"] = approved
                        result["sector_policy"] = sector_policy_report.to_dict()
                        _record_sector_policy_findings(
                            policy=req.sector_policy,
                            report=sector_policy_report,
                            source="batch_review",
                        )
                    results.append(result)
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

            metrics.observe("batch_size", float(batch_res.total))
            approved_count = sum(1 for item in results if item.get("approved"))
            rejected_count = max(batch_res.total - approved_count, 0)
            metrics.inc("reviews_total", float(batch_res.total))
            metrics.inc("reviews_approved", float(approved_count))
            metrics.inc("reviews_rejected", float(rejected_count))
            if req.task == "review":
                for item in results:
                    metrics.observe("coherence_score", float(item["score"]))
            else:
                for item in results:
                    if item.get("approved"):
                        metrics.observe("coherence_score", float(item["score"]))

            return BatchResponse(
                results=results,
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
        """List tenants visible to the authenticated caller."""
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
        """Reject cross-tenant writes for tenant-bound API keys."""
        bound = getattr(request.state, "tenant_id", "")
        if bound and bound != tenant_id:
            raise HTTPException(403, "API key not authorized for this tenant")

    def _enforce_kb_write_access(request: Request, tenant_id: str) -> None:
        """Enforce configured knowledge-base write access controls."""
        try:
            check_kb_write_access(
                require_auth=cfg.knowledge_write_require_auth,
                require_tenant_binding=cfg.knowledge_write_require_tenant_binding,
                authenticated=bool(getattr(request.state, "kb_write_key_ok", False)),
                tenant_binding_enforced=bool(
                    getattr(request.state, "kb_tenant_binding_ok", False)
                ),
                bound_tenant=getattr(request.state, "tenant_id", ""),
                requested_tenant=tenant_id,
            )
        except KBWriteAccessError as exc:
            raise HTTPException(exc.status_code, exc.detail) from exc

    def _kb_signature_metadata(
        request: Request,
        canonical_payload: str,
        signature: str,
        key_id: str,
    ) -> dict[str, object]:
        """Verify tenant knowledge writes and return signature metadata."""
        clean_signature = signature.strip()
        clean_key_id = key_id.strip()
        if not clean_signature:
            if cfg.knowledge_write_require_signature:
                raise HTTPException(403, "Knowledge-base write signature required")
            return {}
        if not verify_kb_payload_signature(
            canonical_payload,
            clean_signature,
            parse_hmac_keys(cfg.knowledge_write_hmac_keys),
            clean_key_id,
        ):
            raise HTTPException(403, "Invalid knowledge-base write signature")
        return {
            "kb_signature": clean_signature,
            "kb_signature_key_id": clean_key_id,
            "kb_signature_verified": True,
        }

    @app.post("/v1/tenants/{tenant_id}/facts", response_model=StatusResponse)
    async def add_tenant_fact(request: Request, tenant_id: str, req: TenantFactRequest):
        """Add a scalar tenant fact after tenant and write checks."""
        router = request.app.state._state.get("tenant_router")
        if not router:
            raise HTTPException(404, "Tenant routing not enabled")
        _enforce_tenant_binding(request, tenant_id)
        _enforce_kb_write_access(request, tenant_id)
        _kb_signature_metadata(
            request,
            canonical_kb_payload(
                kind="tenant_fact",
                tenant_id=tenant_id,
                key=req.key,
                value=req.value,
            ),
            req.signature,
            req.signature_key_id,
        )
        router.add_fact(tenant_id, req.key, req.value)
        return {"status": "ok", "tenant_id": tenant_id, "key": req.key}

    @app.post("/v1/tenants/{tenant_id}/vector-facts", response_model=StatusResponse)
    async def add_tenant_vector_fact(
        request: Request,
        tenant_id: str,
        req: TenantVectorFactRequest,
    ):
        """Add a tenant-scoped vector fact to the configured vector store."""
        router = request.app.state._state.get("tenant_router")
        if not router:
            raise HTTPException(404, "Tenant routing not enabled")
        _enforce_tenant_binding(request, tenant_id)
        _enforce_kb_write_access(request, tenant_id)
        sig_meta = _kb_signature_metadata(
            request,
            canonical_kb_payload(
                kind="tenant_vector_fact",
                tenant_id=tenant_id,
                key=req.key,
                value=req.value,
            ),
            req.signature,
            req.signature_key_id,
        )
        try:
            store = router.get_vector_store(tenant_id, backend_type=req.backend_type)
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, f"Invalid backend_type: {exc}") from exc
        store.add_fact(req.key, req.value, metadata=sig_meta)
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
        """Return a session only to its owning API-key identity."""
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
        """Delete a session owned by the authenticated API-key identity."""
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
        """Return structured in-process metrics."""
        return metrics.get_metrics()

    @app.get("/v1/metrics/prometheus", response_class=PlainTextResponse)
    async def get_prometheus(request: Request):
        """Return metrics in Prometheus text exposition format."""
        return metrics.prometheus_format()

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Config Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    @app.get("/v1/config", response_model=ConfigResponse)
    async def get_config():
        """Return the effective non-secret server configuration."""
        return ConfigResponse(config=cfg.to_dict())

    @app.get("/v1/scorer/models")
    async def list_scorer_models(include_domain_only: bool = False):
        """Return current scorer settings and available scorer choices."""
        from .core.scoring.model_choices import scorer_model_choices_to_dict

        return {
            "current": {
                "scorer_model": cfg.scorer_model,
                "nli_model": cfg.nli_model,
                "nli_model_artifact_uri": cfg.nli_model_artifact_uri,
            },
            "models": scorer_model_choices_to_dict(
                include_domain_only=include_domain_only,
            ),
        }

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
        """Return review statistics from SQLite or Prometheus counters."""
        stats_store = request.app.state._state.get("stats")
        if stats_store:
            return stats_store.summary()
        return _prometheus_summary()

    @app.get("/v1/stats/hourly")
    async def get_stats_hourly(request: Request, days: int = 7):
        """Return hourly review statistics when SQLite stats are enabled."""
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
        """Render the built-in operational statistics dashboard."""
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
        """Return an EU AI Act Article 15 compliance report."""
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
        """Return compliance drift analysis for recent review windows."""
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
        """Return 24-hour, 7-day, and 30-day compliance dashboard data."""
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
                    source_id=c.source_id,
                    external_status=c.external_status,
                )
                for c in result.claims
            ],
            citation_statuses=[
                FreshnessStatusResponse(
                    source_id=v.source_id,
                    status=v.status,
                    risk=v.risk,
                    reason=v.reason,
                    status_source=v.status_source,
                )
                for v in result.citation_status_verdicts
            ],
            overall_staleness_risk=result.overall_staleness_risk,
            external_status_risk=result.external_status_risk,
            has_temporal_claims=result.has_temporal_claims,
            stale_claim_count=len(result.stale_claims),
            risky_status_count=len(result.risky_statuses),
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
            """Adapt the configured scorer to the adversarial tester API."""
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

    # Per-process WebSocket connection accounting (DoS controls).
    ws_conn_lock = asyncio.Lock()
    ws_conn_state: dict[str, int] = {"total": 0}
    ws_per_ip: dict[str, int] = {}

    async def _ws_admit(client_ip: str) -> bool:
        """Reserve a connection slot under the global and per-IP caps."""
        async with ws_conn_lock:
            if ws_conn_state["total"] >= _WS_MAX_CONNECTIONS:
                metrics.inc_labeled("ws_rejections_total", {"reason": "global_cap"})
                return False
            if ws_per_ip.get(client_ip, 0) >= _WS_MAX_CONNECTIONS_PER_IP:
                metrics.inc_labeled("ws_rejections_total", {"reason": "per_ip_cap"})
                return False
            ws_conn_state["total"] += 1
            ws_per_ip[client_ip] = ws_per_ip.get(client_ip, 0) + 1
            metrics.gauge_set("ws_active_connections", float(ws_conn_state["total"]))
            return True

    async def _ws_release(client_ip: str) -> None:
        """Release a previously reserved connection slot."""
        async with ws_conn_lock:
            ws_conn_state["total"] = max(0, ws_conn_state["total"] - 1)
            remaining = ws_per_ip.get(client_ip, 0) - 1
            if remaining > 0:
                ws_per_ip[client_ip] = remaining
            else:
                ws_per_ip.pop(client_ip, None)
            metrics.gauge_set("ws_active_connections", float(ws_conn_state["total"]))

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        """Handle multiplexed WebSocket agent sessions."""
        ws_tenant_id = ""
        if _valid_api_keys:
            provided = ws.headers.get("X-API-Key", "")
            ws_key_valid = False
            for k in _valid_api_keys:
                if hmac.compare_digest(provided, k):
                    ws_key_valid = True
            if not ws_key_valid:
                await ws.close(code=1008, reason="unauthorized")
                return
            if _api_key_tenant_map:
                if provided not in _api_key_tenant_map:
                    await ws.close(code=1008, reason="API key not bound to any tenant")
                    return
                ws_tenant_id = _api_key_tenant_map.get(provided, "")
                claimed = ws.headers.get("X-Tenant-ID", "")
                if claimed and ws_tenant_id and claimed != ws_tenant_id:
                    await ws.close(code=1008, reason="tenant mismatch")
                    return
        if not ws_tenant_id:
            ws_tenant_id = ws.headers.get("X-Tenant-ID", "")

        client_ip = ws.client.host if ws.client else ""
        if not await _ws_admit(client_ip):
            await ws.close(code=1013, reason="server at capacity")
            return
        await ws.accept()

        send_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(_WS_MAX_CONCURRENT)
        active_tasks: dict[str, tuple[asyncio.Task, threading.Event]] = {}

        async def _send(payload: dict) -> None:
            """Serialise WebSocket writes through a per-connection lock."""
            async with send_lock:
                await ws.send_json(payload)

        async def _handle_session(session_id: str, data: dict) -> None:
            """Process one WebSocket session payload."""
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
                    cancel_event = data["_cancel_event"]
                    result = await agent.aprocess(
                        prompt,
                        tenant_id=ws_tenant_id,
                        cancel_event=cancel_event,
                    )
                    if cancel_event.is_set():
                        return
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
                cancel_event = data["_cancel_event"]
                result = await agent.aprocess(
                    prompt,
                    tenant_id=ws_tenant_id,
                    cancel_event=cancel_event,
                )
                if cancel_event.is_set():
                    return
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
            """Run a session under the per-connection concurrency limit."""
            async with semaphore:
                try:
                    await _handle_session(session_id, data)
                finally:
                    active_tasks.pop(session_id, None)

        conn_start = time.monotonic()
        msg_window: list[float] = []
        processed_chars = 0
        try:
            while True:
                if time.monotonic() - conn_start > _WS_MAX_LIFETIME_S:
                    metrics.inc_labeled(
                        "ws_rejections_total", {"reason": "lifetime_exceeded"}
                    )
                    await ws.close(code=1001, reason="session lifetime exceeded")
                    break

                try:
                    data = await asyncio.wait_for(
                        ws.receive_json(), timeout=_WS_IDLE_TIMEOUT_S
                    )
                except TimeoutError:
                    metrics.inc_labeled(
                        "ws_rejections_total", {"reason": "idle_timeout"}
                    )
                    await ws.close(code=1001, reason="idle timeout")
                    break
                except (ValueError, KeyError) as exc:
                    logger.warning("WebSocket bad JSON: %s", exc)
                    await _send({"error": "invalid JSON"})
                    continue

                now = time.monotonic()
                msg_window.append(now)
                msg_window[:] = [t for t in msg_window if now - t <= _WS_RATE_WINDOW_S]
                if len(msg_window) > _WS_MAX_MSGS_PER_WINDOW:
                    metrics.inc_labeled(
                        "ws_rejections_total", {"reason": "rate_limited"}
                    )
                    await _send({"error": "message rate limit exceeded"})
                    continue

                if not isinstance(data, dict):
                    await _send({"error": "expected JSON object"})
                    continue

                # Cancel action
                action = data.get("action", "")
                if action == "cancel":
                    cancel_sid = data.get("session_id", "")
                    active = active_tasks.get(cancel_sid)
                    if active:
                        task, cancel_event = active
                        cancel_event.set()
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

                processed_chars += len(prompt)
                if processed_chars > _WS_CONN_CHAR_BUDGET:
                    metrics.inc_labeled(
                        "ws_rejections_total", {"reason": "budget_exceeded"}
                    )
                    await ws.close(code=1009, reason="connection budget exceeded")
                    break

                session_id = data.get("session_id") or str(uuid.uuid4())
                if session_id in active_tasks:
                    await _send(
                        {
                            "session_id": session_id,
                            "error": "session already active",
                        },
                    )
                    continue
                if len(active_tasks) >= _WS_MAX_CONCURRENT:
                    await _send(
                        {
                            "session_id": session_id,
                            "error": "too many active sessions",
                        },
                    )
                    continue
                data["_cancel_event"] = threading.Event()
                task = asyncio.create_task(_run_session(session_id, data))
                active_tasks[session_id] = (task, data["_cancel_event"])

        except WebSocketDisconnect:
            pass
        finally:
            for task, cancel_event in active_tasks.values():
                cancel_event.set()
                task.cancel()
            await _ws_release(client_ip)

    return app
