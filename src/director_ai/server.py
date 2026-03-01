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

import logging
import sqlite3
import time
from contextlib import asynccontextmanager

from .core.config import DirectorConfig
from .core.metrics import metrics
from .core.stats import StatsStore

logger = logging.getLogger("DirectorAI.Server")

_WS_MAX_PROMPT_LENGTH = 100_000

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


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


def _evidence_to_dict(evidence) -> dict | None:
    """Serialize ScoringEvidence to a JSON-safe dict."""
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
    """Create and configure the FastAPI application.

    Parameters
    ----------
    config : DirectorConfig — server configuration (default: from env).
    """
    _check_fastapi()

    cfg = config or DirectorConfig.from_env()
    _start_time = time.monotonic()

    # Lazy-init shared state
    _state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from .core.agent import CoherenceAgent
        from .core.batch import BatchProcessor
        from .core.scorer import CoherenceScorer

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

    @app.middleware("http")
    async def _http_metrics(request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        metrics.observe("http_request_duration_seconds", elapsed)
        metrics.inc_labeled("http_requests_total", {
            "method": request.method,
            "endpoint": request.url.path,
            "status": str(response.status_code),
        })
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
    async def review(req: ReviewRequest):
        scorer = _state.get("scorer")
        if not scorer:
            raise HTTPException(503, "Server not ready")
        metrics.inc("reviews_total")
        with metrics.timer("review_duration_seconds"):
            approved, score = scorer.review(req.prompt, req.response)
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
    async def process(req: ProcessRequest):
        agent = _state.get("agent")
        if not agent:
            raise HTTPException(503, "Server not ready")
        metrics.inc("reviews_total")
        with metrics.timer("review_duration_seconds"):
            result = agent.process(req.prompt)
        if result.halted:
            metrics.inc("reviews_rejected")
            metrics.inc("halts_total")
        else:
            metrics.inc("reviews_approved")
            if result.coherence:
                metrics.observe("coherence_score", result.coherence.score)
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
                    }
                )
        except WebSocketDisconnect:
            pass

    return app
