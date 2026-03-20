# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” FastAPI Server

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


# â”€â”€ Pydantic request/response models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                "Director-AI v%s â€” Licensed to %s (%s tier)",
                __import__("director_ai").__version__,
                lic.licensee or lic.key[:20],
                lic.tier,
            )
        elif lic.is_trial:
            logger.info("Director-AI â€” Trial license (expires %s)", lic.expires)
        else:
            logger.info("Director-AI â€” AGPL-3.0-or-later (community)")

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

        if cfg.tenant_routing:
            app.state._state["tenant_router"] = TenantRouter()
            logger.info("Tenant routing enabled")

        from .core.retrieval.doc_registry import DocRegistry

        app.state._state["doc_registry"] = DocRegistry()

        cfg.configure_logging()

        if cfg.otel_enabled:
            from .core.otel import setup_otel

            setup_otel()

        if cfg.use_nli:  # pragma: no cover â€” lifespan only runs under ASGI
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
            except Exception:  # pragma: no cover â€” defensive
                logger.warning("Failed to close stats database")

    app = FastAPI(
        title="Director-Class AI",
        version="3.9.4",
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

    # â”€â”€ Rate limiting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                return JSONResponse(  # pragma: no cover â€” ASGI runtime handler
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )

    # â”€â”€ Middleware: correlation IDs + API key auth + metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
            api_key_hash = hmac.new(
                b"director-ai", provided.encode(), "sha256"
            ).hexdigest()[:16]

            # Tenant binding: enforce API key â†’ tenant mapping if configured
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

    # â”€â”€ Health â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    @app.get("/v1/ready")
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

    # â”€â”€ AGPL Â§13 source endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @app.get("/v1/source")
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

    # â”€â”€ Review â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        if not scorer:  # pragma: no cover â€” lifespan always sets scorer
            raise HTTPException(503, "Server not ready")

        # Tenant routing â€” S-05: log tenant access for audit trail
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

        return ReviewResponse(
            approved=approved,
            coherence=score.score,
            h_logical=score.h_logical,
            h_factual=score.h_factual,
            warning=score.warning,
            evidence=_evidence_to_dict(score.evidence),
        )

    # â”€â”€ Verified Review (sentence-level multi-signal) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @app.post("/v1/verify")
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

    # â”€â”€ Process â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        if not agent:  # pragma: no cover â€” lifespan always sets agent
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

    # â”€â”€ Batch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
        if not batcher:  # pragma: no cover â€” lifespan always sets batch
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

    # â”€â”€ Tenants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @app.get("/v1/tenants")
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

    @app.post("/v1/tenants/{tenant_id}/facts")
    async def add_tenant_fact(request: Request, tenant_id: str, req: TenantFactRequest):
        router = request.app.state._state.get("tenant_router")
        if not router:
            raise HTTPException(404, "Tenant routing not enabled")
        _enforce_tenant_binding(request, tenant_id)
        router.add_fact(tenant_id, req.key, req.value)
        return {"status": "ok", "tenant_id": tenant_id, "key": req.key}

    @app.post("/v1/tenants/{tenant_id}/vector-facts")
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

    # â”€â”€ Sessions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @app.get("/v1/sessions/{session_id}")
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

    @app.delete("/v1/sessions/{session_id}")
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

    # â”€â”€ Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @app.get("/v1/metrics")
    async def get_metrics(request: Request):
        return metrics.get_metrics()

    @app.get("/v1/metrics/prometheus", response_class=PlainTextResponse)
    async def get_prometheus(request: Request):
        return metrics.prometheus_format()

    # â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @app.get("/v1/config", response_model=ConfigResponse)
    async def get_config():
        return ConfigResponse(config=cfg.to_dict())

    # â”€â”€ Stats / Dashboard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    @app.get("/v1/stats")
    async def get_stats(request: Request):
        stats_store = request.app.state._state.get("stats")
        if stats_store:
            return stats_store.summary()
        return _prometheus_summary()

    @app.get("/v1/stats/hourly")
    async def get_stats_hourly(request: Request, days: int = 7):
        stats_store = request.app.state._state.get("stats")
        if stats_store:
            return stats_store.hourly_breakdown(days=days)
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

    # â”€â”€ WebSocket streaming (multiplexed) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
