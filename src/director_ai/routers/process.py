# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — pipeline process and batch routes

"""The /v1/process and /v1/batch pipeline routes.

Split out of the ``create_app`` factory. Both handlers read the agent, batcher,
sanitizer, redactor, and audit log from ``request.app.state``, so
``create_process_router`` needs no construction-time dependencies.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..core.metrics import metrics
from ..server_support import (
    _can_suppress_batcher_metrics,
    _record_sector_policy_findings,
)

logger = logging.getLogger("DirectorAI.Server")

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import PlainTextResponse

    from .._server_helpers import evidence_to_dict as _evidence_to_dict
    from .._server_helpers import halt_evidence_to_dict as _halt_evidence_to_dict
    from .._server_models import (
        _MAX_PROMPT_CHARS,
        _MAX_RESPONSE_CHARS,
        BatchRequest,
        BatchResponse,
        ProcessRequest,
        ProcessResponse,
    )

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def create_process_router() -> APIRouter:
    """Build the pipeline route group (process, batch)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.post("/v1/process", response_model=ProcessResponse)
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
        if not agent:  # pragma: no cover - lifespan always sets agent
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

    @router.post("/v1/batch", response_model=BatchResponse)
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
        if not batcher:  # pragma: no cover - lifespan always sets batch
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
                        from ..core.financial_services import assess_banking_response

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

    return router
