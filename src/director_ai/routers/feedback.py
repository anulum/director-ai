# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — human-feedback and calibration routes

"""The /v1/feedback and /v1/feedback/calibration routes.

Split out of the ``create_app`` factory. Both handlers read the feedback store
from ``request.app.state``, so ``create_feedback_router`` needs no
construction-time dependencies.
"""

from __future__ import annotations

from ..core.metrics import metrics

try:
    from fastapi import APIRouter, HTTPException, Request

    from .._server_models import (
        FeedbackCalibrationResponse,
        FeedbackRequest,
        FeedbackResponse,
    )

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def create_feedback_router() -> APIRouter:
    """Build the human-feedback route group (report, calibration)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.post("/v1/feedback", response_model=FeedbackResponse)
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

    @router.get("/v1/feedback/calibration", response_model=FeedbackCalibrationResponse)
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

        from ..core.calibration.online_calibrator import OnlineCalibrator

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

    return router
