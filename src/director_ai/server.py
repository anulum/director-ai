# SPDX-License-Identifier: Apache-2.0
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
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from .core.config import DirectorConfig
from .core.metrics import metrics
from .server_support import (
    _extract_request_api_key,
    _normalize_request_id,
    _record_http_metrics,
)
from .server_support import (
    _http_endpoint_label as _http_endpoint_label,  # re-exported for tests
)

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

logger = logging.getLogger("DirectorAI.Server")

_AUTH_EXEMPT_PATHS_BASE = frozenset(
    {"/v1/live", "/v1/health", "/v1/ready", "/v1/source"}
)

try:
    from fastapi import (
        FastAPI,
        Request,
        Response,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

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


# Pydantic models extracted to _server_models.py (reduce module size)
_MAX_PROMPT_CHARS = 100_000
_MAX_RESPONSE_CHARS = 500_000

if _FASTAPI_AVAILABLE:  # pragma: no branch
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


def create_app(config: DirectorConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    _check_fastapi()

    # Bridge managed secrets (Vault / AWS / Azure) into the environment before
    # any config, license, or audit-salt read, so existing os.environ.get(...)
    # call sites resolve from the backend. No-op for the default env backend.
    from .core.secrets import hydrate_managed_secrets

    loaded_secrets = hydrate_managed_secrets()
    if loaded_secrets:
        logger.info("Loaded %d managed secret(s) from backend", len(loaded_secrets))

    if config is None:
        import os

        profile = os.environ.get("DIRECTOR_PROFILE", "")
        if profile and profile != "default":
            cfg = DirectorConfig.from_profile(profile)
        else:
            cfg = DirectorConfig.from_env()
    else:
        cfg = config

    # Fail fast: a production deployment must carry a per-installation audit salt,
    # never the shared legacy default (cross-tenant fingerprint correlation).
    from .core.safety.audit_salt import get_audit_salt

    get_audit_salt(strict=cfg.production_mode)

    _start_time = time.monotonic()
    _state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
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
            logger.info("Director-AI Ă˘â‚¬â€ť open core: Apache-2.0 + BUSL-1.1")

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

        if cfg.prompt_guard_model_enabled:
            from .core.safety.prompt_guard import (
                LayeredPromptGuard,
                PromptInjectionModel,
            )

            base = app.state._state.get("sanitizer") or InputSanitizer(
                block_threshold=cfg.sanitizer_block_threshold,
            )
            try:
                model = PromptInjectionModel.from_pretrained(
                    cfg.prompt_guard_model_id,
                    revision=cfg.prompt_guard_model_revision or None,
                    threshold=cfg.prompt_guard_threshold,
                )
                app.state._state["sanitizer"] = LayeredPromptGuard(base, model)
                logger.info(
                    "Model-backed prompt-injection screen enabled: %s",
                    cfg.prompt_guard_model_id,
                )
            except Exception as exc:  # noqa: BLE001 — degrade, never crash startup
                app.state._state["sanitizer"] = base
                logger.warning(
                    "prompt_guard_model_enabled but the classifier could not "
                    "load (%s); falling back to the pattern sanitizer.",
                    exc,
                )

        from .core.redactor import PIIRedactor

        app.state._state["redactor"] = PIIRedactor(enabled=cfg.redact_pii)
        if cfg.redact_pii:
            logger.info("Enterprise PII Redaction enabled")

        store = cfg.build_store()
        scorer = cfg.build_scorer(store=store)
        agent_kwargs: dict[str, Any] = {
            "_scorer": scorer,
            "_store": store,
            "production_mode": cfg.production_mode,
            "llm_max_tokens": cfg.llm_max_tokens,
            "llm_temperature": cfg.llm_temperature,
            "max_candidates": cfg.max_candidates,
        }
        if cfg.llm_provider == "local":
            agent_kwargs["llm_api_url"] = cfg.llm_api_url
        elif cfg.llm_provider in ("openai", "anthropic"):
            agent_kwargs["provider"] = cfg.llm_provider
            if cfg.llm_api_key:
                agent_kwargs["api_key"] = cfg.llm_api_key
            logger.info("LLM provider: %s", cfg.llm_provider)
        contradiction_halt = cfg.build_contradiction_halt(store)
        if contradiction_halt is not None:
            agent_kwargs["contradiction_halt"] = contradiction_halt
            logger.info(
                "Contradiction-driven streaming halt enabled (threshold=%.2f)",
                cfg.streaming_contradiction_threshold,
            )
        correctness_feedback = cfg.build_correctness_feedback()
        if correctness_feedback is not None:
            agent_kwargs["correctness_feedback"] = correctness_feedback
            logger.info(
                "REMANENTIA recall-correctness feedback enabled (%s)",
                cfg.remanentia_base_url,
            )
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
    app.state.start_time = _start_time

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
            elif cfg.server_workers > 1:
                # In-memory storage is per worker process; the configured limit is
                # multiplied by the worker count and is not shared across
                # instances. Set redis_url for a single global limit.
                logger.warning(
                    "Rate limiting is in-memory but server_workers=%d: the limit "
                    "is enforced per worker process, not globally. Set redis_url "
                    "for a shared limit across workers and instances.",
                    cfg.server_workers,
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
            async def _rate_limit_handler(
                request: Request, exc: RateLimitExceeded
            ) -> JSONResponse:
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

    # Expose the effective auth state on app.state so request handlers (and the
    # routers split out of this factory) read it from the request instead of
    # closing over create_app locals. The list is shared by reference, so later
    # merges stay visible.
    app.state.valid_api_keys = _valid_api_keys
    app.state.api_key_tenant_map = _api_key_tenant_map

    from .core.runtime.ws_ticket import WebSocketTicketRegistry

    _ws_ticket_registry = WebSocketTicketRegistry(
        ttl_seconds=getattr(cfg, "ws_ticket_ttl_seconds", 30.0),
    )
    app.state.ws_ticket_registry = _ws_ticket_registry

    @app.middleware("http")
    async def _http_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
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
                # Declared as the base Response so the later call_next() result
                # (a Response) is assignable to the same name.
                response: Response = JSONResponse(
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
                    # api_key_hash is a SHA-256 digest (see above), not the key.
                    # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
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

    # Diagnostic routes — live, health, ready, source, metrics, config — live in
    # routers/health.py and read their state from app.state.
    from .routers.health import create_health_router

    app.include_router(create_health_router())

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Review Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    # Core scoring routes - review, verify, injection/detect, multimodal/check -
    # live in routers/scoring.py and read scorer/sanitizer/redactor/sessions/
    # stats/audit/config from app.state.
    from .routers.scoring import create_scoring_router

    app.include_router(create_scoring_router())

    # Human-feedback routes - report, calibration - live in routers/feedback.py
    # and read the feedback store from app.state.
    from .routers.feedback import create_feedback_router

    app.include_router(create_feedback_router())

    # ── Process ───────────────────────────────────────────────────

    # Pipeline routes - process, batch - live in routers/process.py and read
    # their state (agent, batcher, sanitizer, redactor, audit) from app.state.
    from .routers.process import create_process_router

    app.include_router(create_process_router())

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Tenants Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    # Tenant routes - list, add fact, add vector fact - live in routers/tenants.py
    # and read the tenant router + write-access config from app.state.
    from .routers.tenants import create_tenants_router

    app.include_router(create_tenants_router())

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Sessions Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    # Session routes - get, delete - live in routers/sessions.py and read the
    # session store + ownership map from app.state under the shared lock.
    from .routers.sessions import create_sessions_router

    app.include_router(create_sessions_router())

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Metrics Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Config Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    # Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬ Stats / Dashboard Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬Ă˘â€ťâ‚¬

    # Statistics routes - stats, hourly, dashboard - live in routers/stats.py
    # and read the stats store from app.state.
    from .routers.stats import create_stats_router

    app.include_router(create_stats_router())

    # -- Compliance endpoints (EU AI Act Article 15) --------------------

    # Compliance routes - report, drift, dashboard - live in routers/compliance.py
    # and read the reporter + drift detector from app.state.
    from .routers.compliance import create_compliance_router

    app.include_router(create_compliance_router())

    # -- Gem endpoints (Phase 5 verification & analysis) -----------------

    # Standalone verification routes - numeric, reasoning, temporal-freshness,
    # consensus, adversarial, conformal, feedback-loops, agentic check-step -
    # live in routers/verification.py.
    from .routers.verification import create_verification_router

    app.include_router(create_verification_router())

    # WebSocket streaming - /v1/stream + /v1/stream/ticket - lives in
    # routers/streaming.py (per-process connection accounting + admit/release are
    # factory locals there). Reads auth keys, tenant map, ticket registry, and
    # config from app.state.
    from .routers.streaming import create_streaming_router

    app.include_router(create_streaming_router())

    return app
