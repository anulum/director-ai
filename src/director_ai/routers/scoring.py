# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — core scoring and verification routes

"""The core scoring routes: review, verify, injection detect, multimodal check.

Split out of the ``create_app`` factory. Every handler reads the scorer,
sanitizer, redactor, sessions, stats, audit, and config from
``request.app.state``, so ``create_scoring_router`` needs no construction-time
dependencies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..core.metrics import metrics
from ..server_support import _record_sector_policy_findings

logger = logging.getLogger("DirectorAI.Server")

try:
    from fastapi import APIRouter, HTTPException, Request

    from .._server_helpers import evidence_to_dict as _evidence_to_dict
    from .._server_models import (
        InjectionRequest,
        InjectionResponse,
        MultimodalDetectRequest,
        MultimodalDetectResponse,
        ReviewRequest,
        ReviewResponse,
        VerifyResponse,
    )

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def create_scoring_router() -> APIRouter:
    """Build the core scoring route group (review, verify, injection, multimodal)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.post("/v1/review", response_model=ReviewResponse)
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
        if not scorer:  # pragma: no cover - lifespan always sets scorer
            raise HTTPException(503, "Server not ready")

        # Tenant routing - S-05: log tenant access for audit trail
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
            from ..core.runtime.session import ConversationSession

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
            from ..core.financial_services import assess_banking_response

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
            from ..compliance.audit_log import AuditEntry as CAuditEntry

            c_log.log(
                CAuditEntry(
                    prompt=req.prompt,
                    response=req.response,
                    model=getattr(request.app.state.config, "llm_model", "server"),
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

    @router.post("/v1/verify", response_model=VerifyResponse)
    async def verify_response(req: ReviewRequest, request: Request) -> dict[str, Any]:
        """Atomic multi-span fact verification.

        Decomposes the response into claims, ranks source spans from the
        KB, aggregates evidence, and checks NLI + entity + number +
        negation + traceability signals. Returns per-claim verdicts with
        confidence and provenance.
        """
        import asyncio

        from ..core.scoring.verified_scorer import VerifiedScorer

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

    @router.post("/v1/injection/detect", response_model=InjectionResponse)
    async def detect_injection(
        req: InjectionRequest, request: Request
    ) -> dict[str, Any]:
        """Detect prompt injection effects in LLM output via NLI divergence.

        Analyses whether the response diverges from the stated intent
        (system_prompt + user_query).  Returns per-claim attribution
        with grounded/drifted/injected verdicts.
        """
        import asyncio

        from ..core.safety.injection import InjectionDetector

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

    @router.post("/v1/multimodal/check", response_model=MultimodalDetectResponse)
    async def multimodal_check(
        req: MultimodalDetectRequest, request: Request
    ) -> dict[str, Any]:
        """Check a text claim against paired image / audio / video evidence.

        Opt-in and isolated: returns 404 unless the experimental hooks flag is
        set and at least one modality is configured. The response is
        tenant-safe - no raw media, transcript, or claim text is echoed back.
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

        from ..core.guard_control import RiskEnvelope
        from ..core.multimodal_guard import MultimodalCheckRequest

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
        payload: dict[str, Any] = result.to_dict()
        return payload

    return router
