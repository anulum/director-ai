# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FastAPI Server

"""Production-ready FastAPI server for Director-AI.

Usage::

    # Programmatic
    from director_ai.server import create_app
    app = create_app()

    # CLI
    director-ai serve --port 8080
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ._server_auth_middleware import (
    # Re-exported: the request-ID context and exempt-path base historically
    # live on director_ai.server for tests and integrations.
    _AUTH_EXEMPT_PATHS_BASE as _AUTH_EXEMPT_PATHS_BASE,
)
from ._server_auth_middleware import (
    REQUEST_ID_CTX as REQUEST_ID_CTX,
)
from ._server_auth_middleware import (
    install_auth_middleware,
)
from ._server_lifecycle import (
    # Re-exported for the fine-tune hot-swap surface and its tests, which
    # historically reach these through ``director_ai.server``.
    _activate_scorer as _activate_scorer,
)
from ._server_lifecycle import (
    _build_coherence_agent as _build_coherence_agent,
)
from ._server_lifecycle import (
    _swap_scorer as _swap_scorer,
)
from ._server_lifecycle import server_lifespan
from .core.config import DirectorConfig
from .server_support import (
    _extract_request_api_key as _extract_request_api_key,  # re-exported for tests
)
from .server_support import (
    _http_endpoint_label as _http_endpoint_label,  # re-exported for tests
)
from .server_support import (
    _normalize_request_id as _normalize_request_id,  # re-exported for tests
)
from .server_support import (
    _record_http_metrics as _record_http_metrics,  # re-exported for tests
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

logger = logging.getLogger("DirectorAI.Server")

try:
    from fastapi import (
        FastAPI,
        Request,
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
    if (
        not _FASTAPI_AVAILABLE
    ):  # pragma: no cover — extras-gated: FastAPI import guard ([server] extra)
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

    import director_ai

    app = FastAPI(
        title="Director-AI",
        version=director_ai.__version__,
        description="Real-time multi-agent orchestration and coherence scoring.",
        lifespan=server_lifespan,
    )
    app.state.config = cfg
    app.state.router_mounts = {}
    app.state.start_time = _start_time

    # Fine-tuning API router (Phase C)
    try:
        from .finetune_api import create_finetune_router

        finetune_models_dir = (
            Path(cfg.finetune_models_dir) if cfg.finetune_models_dir else None
        )
        app.include_router(
            create_finetune_router(models_dir=finetune_models_dir),
            prefix="/v1/finetune",
        )
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

    # ── Rate limiting ─────────────────────────────────────────────────

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
                return JSONResponse(  # pragma: no cover — ASGI runtime handler
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )

    # Correlation IDs + API-key auth + tenant binding + metrics live in
    # _server_auth_middleware.py; this wires app.state auth surfaces and
    # installs the HTTP middleware.
    install_auth_middleware(app, cfg)

    # ── Health ────────────────────────────────────────────────────────

    # Diagnostic routes — live, health, ready, source, metrics, config — live in
    # routers/health.py and read their state from app.state.
    from .routers.health import create_health_router

    app.include_router(create_health_router())

    # ── Review ────────────────────────────────────────────────────────

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

    # ── Tenants ───────────────────────────────────────────────────────

    # Tenant routes - list, add fact, add vector fact - live in routers/tenants.py
    # and read the tenant router + write-access config from app.state.
    from .routers.tenants import create_tenants_router

    app.include_router(create_tenants_router())

    # ── Sessions ──────────────────────────────────────────────────────

    # Session routes - get, delete - live in routers/sessions.py and read the
    # session store + ownership map from app.state under the shared lock.
    from .routers.sessions import create_sessions_router

    app.include_router(create_sessions_router())

    # ── Metrics ───────────────────────────────────────────────────────

    # ── Config ────────────────────────────────────────────────────────

    # ── Stats / Dashboard ────────────────────────────────────────────

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

    # REST SSE streaming - /v1/stream/sse - lives in routers/streaming_sse.py.
    # Same session shapes as the WebSocket, delivered as text/event-stream;
    # auth and tenant binding ride the standard REST middleware.
    from .routers.streaming_sse import create_sse_router

    app.include_router(create_sse_router())

    return app
