# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — health, readiness, source, metrics, and config routes

"""Diagnostic routes: liveness, health, readiness, source, metrics, config.

Split out of the ``create_app`` factory. Every handler reads its state from
``request.app.state`` (config, start time, licence, router mounts, scorer), so
``create_health_router`` needs no construction-time dependencies.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.metrics import metrics
from ..server_support import _request_authenticated

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse

    from .._server_models import (
        ConfigResponse,
        HealthResponse,
        ReadyResponse,
        SourceResponse,
    )

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def create_health_router() -> APIRouter:
    """Build the diagnostic route group (live, health, ready, source, …)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    @router.get("/v1/live")
    async def liveness() -> dict[str, Any]:
        """Minimal unauthenticated liveness probe — no version or config leak."""
        return {"ok": True}

    @router.get("/v1/health")
    async def health(request: Request) -> dict[str, Any]:
        """Return liveness plus, for authenticated callers, build detail.

        Unauthenticated callers receive ``{status, license}`` only; a valid key
        unlocks version, mode, profile, router, and revision detail. The
        response is no longer a fixed ``HealthResponse`` schema because the
        payload shape now depends on authentication.
        """
        import director_ai

        cfg = request.app.state.config
        lic = getattr(request.app.state, "_license", None)
        extra = {}
        if lic and lic.is_commercial:
            # Public, auth-exempt endpoint: expose the licence type only, never
            # the commercial licensee identity or tier.
            extra = {"license": "commercial"}
        elif lic and lic.is_trial:
            extra = {"license": "trial", "expires": lic.expires}
        else:
            extra = {"license": "open-core"}

        if not _request_authenticated(request):
            # Unauthenticated callers get liveness + licence type only; the
            # detailed build/router/revision payload is gated behind a valid
            # key. Pure liveness probes should use /v1/live.
            return {"status": "ok", **extra}

        resp = HealthResponse(
            version=director_ai.__version__,
            mode=cfg.mode,
            profile=cfg.profile,
            nli_loaded=cfg.use_nli,
            uptime_seconds=time.monotonic() - request.app.state.start_time,
            routers=dict(request.app.state.router_mounts),
            model_revisions=cfg.model_revision_health(),
        )
        return {**resp.model_dump(), **extra}

    @router.get("/v1/ready", response_model=ReadyResponse)
    async def readiness(request: Request) -> dict[str, Any] | JSONResponse:
        """Readiness probe: returns 200 only when scorer is operational."""
        cfg = request.app.state.config
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

    @router.get("/v1/source", response_model=SourceResponse)
    async def source(request: Request) -> dict[str, Any]:
        """Return source-availability metadata for the running deployment.

        Open core: the Apache-2.0 core and the source-available BUSL-1.1 advanced
        tier are both published upstream. This endpoint is a transparency
        convenience, not a licence obligation.
        """
        import director_ai

        cfg = request.app.state.config
        lic = getattr(request.app.state, "_license", None)
        if lic and lic.is_commercial:
            if not cfg.source_endpoint_enabled:
                raise HTTPException(
                    404, "Source endpoint disabled (commercial license)"
                )
            # Do not leak the commercial tier or the build version on this
            # auth-exempt endpoint.
            return {"license": "commercial"}

        if not cfg.source_endpoint_enabled:
            raise HTTPException(404, "Source endpoint disabled")

        return {
            "license": "Apache-2.0 AND BUSL-1.1",
            "version": director_ai.__version__,
            "repository_url": cfg.source_repository_url,
            "instructions": f"git clone {cfg.source_repository_url}",
        }

    @router.get("/v1/metrics")
    async def get_metrics(request: Request) -> dict[str, Any]:
        """Return structured in-process metrics."""
        return metrics.get_metrics()

    @router.get("/v1/metrics/prometheus", response_class=PlainTextResponse)
    async def get_prometheus(request: Request) -> str:
        """Return metrics in Prometheus text exposition format."""
        return metrics.prometheus_format()

    @router.get("/v1/config", response_model=ConfigResponse)
    async def get_config(request: Request) -> ConfigResponse:
        """Return the effective non-secret server configuration."""
        return ConfigResponse(config=request.app.state.config.to_dict())

    @router.get("/v1/scorer/models")
    async def list_scorer_models(
        request: Request,
        include_domain_only: bool = False,
    ) -> dict[str, Any]:
        """Return current scorer settings and available scorer choices."""
        from ..core.scoring.model_choices import scorer_model_choices_to_dict

        cfg = request.app.state.config
        return {
            "current": {
                "scorer_model": cfg.scorer_model,
                "nli_model": cfg.nli_model,
                "nli_model_artifact_uri": cfg.nli_model_artifact_uri,
            },
            "models": scorer_model_choices_to_dict(
                include_domain_only=include_domain_only,
            ),
        }

    return router
