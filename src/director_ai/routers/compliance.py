# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — EU AI Act compliance reporting routes

"""The /v1/compliance report, drift, and dashboard routes.

Split out of the ``create_app`` factory. Each handler reads the compliance
reporter / drift detector from ``request.app.state``, so
``create_compliance_router`` needs no construction-time dependencies.
"""

from __future__ import annotations

import time
from typing import Any

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import PlainTextResponse

    from .._server_models import ComplianceReportResponse, DriftResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def create_compliance_router() -> APIRouter:
    """Build the compliance route group (report, drift, dashboard)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.get("/v1/compliance/report", response_model=ComplianceReportResponse)
    async def compliance_report(
        request: Request,
        since: float | None = None,
        until: float | None = None,
        model: str | None = None,
        domain: str | None = None,
        fmt: str = "json",
    ) -> dict[str, Any] | PlainTextResponse:
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

    @router.get("/v1/compliance/drift", response_model=DriftResponse)
    async def compliance_drift(request: Request) -> dict[str, Any]:
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

    @router.get("/v1/compliance/dashboard")
    async def compliance_dashboard(request: Request) -> dict[str, Any]:
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

    return router
