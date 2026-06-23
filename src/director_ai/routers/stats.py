# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — statistics and dashboard routes

"""The /v1/stats, /v1/stats/hourly, and /v1/dashboard routes.

Split out of the ``create_app`` factory. Each handler reads the stats store from
``request.app.state`` and falls back to the in-process metrics collector, so
``create_stats_router`` needs no construction-time dependencies.
"""

from __future__ import annotations

from typing import Any

from ..core.metrics import metrics

try:
    from fastapi import APIRouter, Request
    from fastapi.responses import PlainTextResponse

    from .._server_models import StatsResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def _prometheus_summary() -> dict[str, Any]:
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
        round(duration_hist["mean"] * 1000, 1) if duration_hist.get("count") else None
    )
    return {
        "total": int(total),
        "approved": int(approved),
        "rejected": int(rejected),
        "halted": int(halted),
        "avg_score": avg_score,
        "avg_latency_ms": avg_latency,
    }


def create_stats_router() -> APIRouter:
    """Build the statistics route group (stats, hourly, dashboard)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.get("/v1/stats", response_model=StatsResponse)
    async def get_stats(request: Request) -> dict[str, Any]:
        """Return review statistics from SQLite or Prometheus counters."""
        stats_store = request.app.state._state.get("stats")
        if stats_store:
            summary: dict[str, Any] = stats_store.summary()
            return summary
        return _prometheus_summary()

    @router.get("/v1/stats/hourly")
    async def get_stats_hourly(request: Request, days: int = 7) -> dict[str, Any]:
        """Return hourly review statistics when SQLite stats are enabled."""
        stats_store = request.app.state._state.get("stats")
        if stats_store:
            result = stats_store.hourly_breakdown(days=days)
            if isinstance(result, list):
                return {"data": result}
            breakdown: dict[str, Any] = result
            return breakdown
        return {
            "data": [],
            "note": "hourly breakdown requires stats_backend=sqlite",
        }

    @router.get("/v1/dashboard", response_class=PlainTextResponse)
    async def dashboard(request: Request) -> str:
        """Render the built-in operational statistics dashboard."""
        stats_store = request.app.state._state.get("stats")
        s = stats_store.summary() if stats_store else _prometheus_summary()
        approval_rate = (
            f"{s['approved'] / s['total'] * 100:.1f}%" if s["total"] else "N/A"
        )
        rows = [
            ("Total Reviews", s["total"]),
            ("Approved", s["approved"]),
            ("Rejected", s["rejected"]),
            ("Halted", s["halted"]),
            ("Approval Rate", approval_rate),
            ("Avg Score", s["avg_score"] or "N/A"),
            ("Avg Latency", f"{s['avg_latency_ms'] or 'N/A'} ms"),
        ]
        # Internal aggregate metrics only (integers/floats from the stats store,
        # never request input), and the endpoint serves PlainTextResponse — there
        # is no user-controlled data and no HTML-rendering context to inject into.
        table_rows = "".join(
            # nosemgrep: python.django.security.injection.raw-html-format.raw-html-format
            f"<tr><th>{label}</th><td>{value}</td></tr>"
            for label, value in rows
        )
        return (
            "<!DOCTYPE html><html><head><title>Director-AI Dashboard</title>"
            "<style>body{font-family:monospace;max-width:600px;margin:40px auto;}"
            "table{border-collapse:collapse;width:100%;}td,th{border:1px solid #ccc;"
            "padding:8px;text-align:left;}</style></head><body>"
            "<h1>Director-AI Dashboard</h1>"
            f"<table>{table_rows}</table></body></html>"
        )

    return router
