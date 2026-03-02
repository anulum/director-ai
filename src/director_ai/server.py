# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — FastAPI Server
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Production-ready FastAPI server for Director-Class AI.

Usage::

    # Programmatic
    from director_ai.server import create_app
    app = create_app()

    # CLI
    director-ai serve --port 8080
"""

from __future__ import annotations

import contextvars
import logging
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager

from .core.config import DirectorConfig
from .core.metrics import metrics
from .core.stats import StatsStore

REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

logger = logging.getLogger("DirectorAI.Server")

_WS_MAX_PROMPT_LENGTH = 100_000
_AUTH_EXEMPT_PATHS = frozenset({"/v1/health", "/v1/metrics/prometheus"})

try:
    from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False


def _check_fastapi() -> None:
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]"
        )


# ── Pydantic request/response models ─────────────────────────────────

if _FASTAPI_AVAILABLE:

    class ReviewRequest(BaseModel):
        prompt: str = Field(..., min_length=1, description="Input prompt")
        response: str = Field(..., min_length=1, description="LLM response to review")

    class ProcessRequest(BaseModel):
        prompt: str = Field(..., min_length=1, description="Input prompt")

    class BatchRequest(BaseModel):
        prompts: list[str] = Field(
            ..., min_length=1, max_length=1000, description="List of prompts"
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
        results: list[ProcessResponse]
        errors: list[dict]
        total: int
        succeeded: int
        failed: int
        duration_seconds: float

    class HealthResponse(BaseModel):
        status: str = "ok"
        version: str
        profile: str
        nli_loaded: bool
        uptime_seconds: float

    class ConfigResponse(BaseModel):
        config: dict

    class TenantFactRequest(BaseModel):
        key: str = Field(..., min_length=1)
        value: str = Field(..., min_length=1)


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
    return {
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


def create_app(config: DirectorConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    _check_fastapi()

    cfg = config or DirectorConfig.from_env()
    _start_time = time.monotonic()

    _state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .core.agent import CoherenceAgent
        from .core.audit import AuditLogger
        from .core.batch import BatchProcessor
        from .core.scorer import CoherenceScorer
        from .core.tenant import TenantRouter

        scorer = CoherenceScorer(
            threshold=cfg.coherence_threshold,
            use_nli=cfg.use_nli,
        )
        agent = CoherenceAgent(
            llm_api_url=cfg.llm_api_url if cfg.llm_provider == "local" else None,
        )
        batch_proc = BatchProcessor(agent, max_concurrency=cfg.batch_max_concurrency)
        stats = StatsStore()

        _state["agent"] = agent
        _state["scorer"] = scorer
        _state["batch"] = batch_proc
        _state["config"] = cfg
        _state["stats"] = stats

        if cfg.audit_log_path:
            _state["audit"] = AuditLogger(path=cfg.audit_log_path)
            logger.info("Audit logging enabled: %s", cfg.audit_log_path)

        if cfg.tenant_routing:
            _state["tenant_router"] = TenantRouter()
            logger.info("Tenant routing enabled")

        cfg.configure_logging()

        if cfg.otel_enabled:
            from .core.otel import setup_otel

            setup_otel()

        if cfg.use_nli:
            metrics.gauge_set("nli_model_loaded", 1.0)

        logger.info(
            "Director AI server started (profile=%s, nli=%s)",
            cfg.profile,
            cfg.use_nli,
        )
        yield
        logger.info("Director AI server shutting down")
        if stats:
            try:
                stats.close()
            except sqlite3.Error:
                logger.warning("Failed to close stats database")

    app = FastAPI(
        title="Director-Class AI",
        description="Coherence Engine — AI Output Verification & Safety Oversight",
        version=__import__("director_ai").__version__,
        lifespan=lifespan,
    )

    _origins = [o.strip() for o in cfg.cors_origins.split(",") if o.strip()]
    if len(_origins) > 100:
        raise ValueError(f"Too many CORS origins: {len(_origins)} (max 100)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ─────────────────────────────────────────────────

    limiter = None
    if cfg.rate_limit_rpm > 0:
        if not _SLOWAPI_AVAILABLE:
            logger.warning(
                "rate_limit_rpm=%d but slowapi not installed. "
                "Install with: pip install director-ai[ratelimit]",
                cfg.rate_limit_rpm,
            )
        else:
            limiter = Limiter(key_func=get_remote_address)
            app.state.limiter = limiter
            from slowapi.errors import RateLimitExceeded

            @app.exception_handler(RateLimitExceeded)
            async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )

    _rate_str = f"{cfg.rate_limit_rpm}/minute" if cfg.rate_limit_rpm > 0 else ""

    # ── Middleware: correlation IDs + API key auth + metrics ───────────

    @app.middleware("http")
    async def _http_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        REQUEST_ID_CTX.set(request_id)

        # API key auth
        if cfg.api_keys and request.url.path not in _AUTH_EXEMPT_PATHS:
            provided = request.headers.get("X-API-Key", "")
            if provided not in cfg.api_keys:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                    headers={"X-Request-ID": request_id},
                )

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

    # ── Health ────────────────────────────────────────────────────────

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        import director_ai

        return HealthResponse(
            version=director_ai.__version__,
            profile=cfg.profile,
            nli_loaded=cfg.use_nli,
            uptime_seconds=time.monotonic() - _start_time,
        )

    # ── Review ────────────────────────────────────────────────────────

    @app.post("/v1/review", response_model=ReviewResponse)
    async def review(req: ReviewRequest, request: Request):
        scorer = _state.get("scorer")
        if not scorer:
            raise HTTPException(503, "Server not ready")

        # Tenant routing
        tenant_id = request.headers.get("X-Tenant-ID", "")
        if tenant_id and "tenant_router" in _state:
            scorer = _state["tenant_router"].get_scorer(
                tenant_id,
                threshold=cfg.coherence_threshold,
                use_nli=cfg.use_nli,
            )

        metrics.inc("reviews_total")
        start = time.monotonic()
        with metrics.timer("review_duration_seconds"):
            approved, score = scorer.review(req.prompt, req.response)
        latency_ms = (time.monotonic() - start) * 1000

        if approved:
            metrics.inc("reviews_approved")
        else:
            metrics.inc("reviews_rejected")
        metrics.observe("coherence_score", score.score)

        stats_store = _state.get("stats")
        if stats_store:
            stats_store.record_review(
                approved=approved,
                score=score.score,
                h_logical=score.h_logical,
                h_factual=score.h_factual,
            )

        audit = _state.get("audit")
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

    # ── Process ───────────────────────────────────────────────────────

    @app.post("/v1/process", response_model=ProcessResponse)
    async def process(req: ProcessRequest, request: Request):
        agent = _state.get("agent")
        if not agent:
            raise HTTPException(503, "Server not ready")
        metrics.inc("reviews_total")
        start = time.monotonic()
        with metrics.timer("review_duration_seconds"):
            result = agent.process(req.prompt)
        latency_ms = (time.monotonic() - start) * 1000

        if result.halted:
            metrics.inc("reviews_rejected")
            metrics.inc("halts_total")
        else:
            metrics.inc("reviews_approved")
            if result.coherence:
                metrics.observe("coherence_score", result.coherence.score)

        audit = _state.get("audit")
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
                latency_ms=latency_ms,
            )

        return ProcessResponse(
            output=result.output,
            coherence=result.coherence.score if result.coherence else None,
            halted=result.halted,
            candidates_evaluated=result.candidates_evaluated,
            warning=result.coherence.warning if result.coherence else False,
            fallback_used=result.fallback_used,
            evidence=_evidence_to_dict(
                result.coherence.evidence if result.coherence else None
            ),
            halt_evidence=_halt_evidence_to_dict(result.halt_evidence),
        )

    # ── Batch ─────────────────────────────────────────────────────────

    @app.post("/v1/batch", response_model=BatchResponse)
    async def batch(req: BatchRequest):
        batch_proc = _state.get("batch")
        if not batch_proc:
            raise HTTPException(503, "Server not ready")
        batch_result = batch_proc.process_batch(req.prompts)
        results = []
        for r in batch_result.results:
            results.append(
                ProcessResponse(
                    output=r.output,
                    coherence=r.coherence.score if r.coherence else None,
                    halted=r.halted,
                    candidates_evaluated=r.candidates_evaluated,
                )
            )
        return BatchResponse(
            results=results,
            errors=[{"index": i, "error": e} for i, e in batch_result.errors],
            total=batch_result.total,
            succeeded=batch_result.succeeded,
            failed=batch_result.failed,
            duration_seconds=batch_result.duration_seconds,
        )

    # ── Tenants ───────────────────────────────────────────────────────

    @app.get("/v1/tenants")
    async def list_tenants():
        router = _state.get("tenant_router")
        if not router:
            raise HTTPException(404, "Tenant routing not enabled")
        return {
            "tenants": [
                {"id": tid, "fact_count": router.fact_count(tid)}
                for tid in router.tenant_ids
            ]
        }

    @app.post("/v1/tenants/{tenant_id}/facts")
    async def add_tenant_fact(tenant_id: str, req: TenantFactRequest):
        router = _state.get("tenant_router")
        if not router:
            raise HTTPException(404, "Tenant routing not enabled")
        router.add_fact(tenant_id, req.key, req.value)
        return {"status": "ok", "tenant_id": tenant_id, "key": req.key}

    # ── Metrics ───────────────────────────────────────────────────────

    @app.get("/v1/metrics")
    async def get_metrics():
        return metrics.get_metrics()

    @app.get("/v1/metrics/prometheus", response_class=PlainTextResponse)
    async def get_prometheus():
        return metrics.prometheus_format()

    # ── Config ────────────────────────────────────────────────────────

    @app.get("/v1/config", response_model=ConfigResponse)
    async def get_config():
        return ConfigResponse(config=cfg.to_dict())

    # ── Stats / Dashboard ────────────────────────────────────────────

    @app.get("/v1/stats")
    async def get_stats():
        stats_store = _state.get("stats")
        if not stats_store:
            raise HTTPException(503, "Stats not available")
        return stats_store.summary()

    @app.get("/v1/stats/hourly")
    async def get_stats_hourly(days: int = 7):
        stats_store = _state.get("stats")
        if not stats_store:
            raise HTTPException(503, "Stats not available")
        return stats_store.hourly_breakdown(days=days)

    @app.get("/v1/dashboard", response_class=PlainTextResponse)
    async def dashboard():
        stats_store = _state.get("stats")
        if not stats_store:
            raise HTTPException(503, "Stats not available")
        s = stats_store.summary()
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

    # ── WebSocket streaming ───────────────────────────────────────────

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                try:
                    data = await ws.receive_json()
                except (ValueError, KeyError) as exc:
                    logger.warning("WebSocket bad JSON: %s", exc)
                    await ws.send_json({"error": "invalid JSON"})
                    continue

                if not isinstance(data, dict):
                    await ws.send_json({"error": "expected JSON object"})
                    continue

                prompt = data.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    await ws.send_json({"error": "prompt must be a non-empty string"})
                    continue

                if len(prompt) > _WS_MAX_PROMPT_LENGTH:
                    await ws.send_json(
                        {"error": f"prompt exceeds {_WS_MAX_PROMPT_LENGTH} chars"}
                    )
                    continue

                agent = _state.get("agent")
                if not agent:
                    await ws.send_json({"error": "server not ready"})
                    continue

                # Streaming oversight mode
                if data.get("streaming_oversight"):
                    from .core.streaming import StreamingKernel

                    scorer = _state.get("scorer")
                    if not scorer:
                        await ws.send_json({"error": "scorer not ready"})
                        continue

                    kernel = StreamingKernel(
                        hard_limit=cfg.hard_limit,
                        soft_limit=cfg.soft_limit,
                    )

                    try:
                        result = agent.process(prompt)
                    except (RuntimeError, ValueError, TypeError, OSError) as exc:
                        await ws.send_json({"error": f"processing failed: {exc}"})
                        continue

                    tokens = result.output.split()

                    def _make_cb(sc, pr):
                        acc = []

                        def cb(token):
                            acc.append(token)
                            text = " ".join(acc)
                            _, s = sc.review(pr, text)
                            return s.score

                        return cb

                    session = kernel.stream_tokens(
                        iter(tokens),
                        _make_cb(scorer, prompt),
                    )

                    for event in session.events:
                        msg = {
                            "type": "token",
                            "token": event.token,
                            "coherence": round(event.coherence, 4),
                            "index": event.index,
                        }
                        if event.halted:
                            msg["type"] = "halt"
                            msg["reason"] = session.halt_reason
                        await ws.send_json(msg)

                    if not session.halted:
                        await ws.send_json(
                            {
                                "type": "result",
                                "output": session.output,
                                "halted": False,
                            }
                        )

                    continue

                # Standard (non-streaming) mode
                try:
                    result = agent.process(prompt)
                except (RuntimeError, ValueError, TypeError, OSError) as exc:
                    logger.error("WebSocket agent.process() failed: %s", exc)
                    await ws.send_json({"error": f"processing failed: {exc}"})
                    continue
                await ws.send_json(
                    {
                        "type": "result",
                        "output": result.output,
                        "coherence": (
                            result.coherence.score if result.coherence else None
                        ),
                        "halted": result.halted,
                        "warning": (
                            result.coherence.warning if result.coherence else False
                        ),
                        "fallback_used": result.fallback_used,
                        "evidence": _evidence_to_dict(
                            result.coherence.evidence if result.coherence else None
                        ),
                        "halt_evidence": _halt_evidence_to_dict(result.halt_evidence),
                    }
                )
        except WebSocketDisconnect:
            pass

    return app
