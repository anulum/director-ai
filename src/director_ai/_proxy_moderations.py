# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — proxy /v1/moderations route

"""OpenAI-compatible ``/v1/moderations`` route for the guardrail proxy.

Two modes, chosen by ``create_proxy_app(moderations=...)``:

* ``"local"`` (default) — every input is analysed by the shipped
  dependency-free detectors (:class:`KeywordToxicityDetector` and
  :class:`RegexPIIDetector`); the response follows the OpenAI
  moderations shape (``flagged`` / ``categories`` /
  ``category_scores``) with Director's own category names (``keyword``,
  ``threat``, ``self_harm_encouragement``, ``email``, ``phone``, …).
  Works against upstreams that have no moderations endpoint at all
  (vLLM, llama.cpp, most self-hosted gateways).
* ``"upstream"`` — the request body is forwarded verbatim to the
  upstream ``/v1/moderations`` and the upstream verdict is returned
  unchanged.

The route module is private; its public surface is the
``moderations`` parameter of :func:`director_ai.proxy.create_proxy_app`.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any

# FastAPI resolves route-handler annotations at runtime, so the names used
# in handler signatures must exist in module globals — mirror the graceful
# degradation used by ``director_ai.proxy`` (see the comment there).
try:
    from fastapi import Request
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover — exercised only without the server extra
    pass

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx
    from fastapi import FastAPI

    from director_ai.core.safety.moderation import ModerationDetector

_log = logging.getLogger("DirectorAI.Proxy.Moderations")

MODERATION_MODES = ("local", "upstream")
LOCAL_MODERATION_MODEL = "director-ai-local-moderation"


def build_default_detectors() -> list[ModerationDetector]:
    """Return the dependency-free detector set used by local mode."""
    from director_ai.core.safety.moderation import (
        KeywordToxicityDetector,
        RegexPIIDetector,
    )

    return [KeywordToxicityDetector(), RegexPIIDetector()]


def parse_moderation_input(body: Any) -> list[str] | None:
    """Extract the OpenAI ``input`` field as a non-empty list of strings.

    Returns ``None`` when the body is not a dict, the field is missing,
    an entry is not a string, or the list is empty — the caller maps
    that to a 400 in the OpenAI error shape.
    """
    if not isinstance(body, dict):
        return None
    raw = body.get("input")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and raw and all(isinstance(i, str) for i in raw):
        return list(raw)
    return None


def moderate_inputs(
    inputs: list[str],
    detectors: list[ModerationDetector],
) -> list[dict[str, Any]]:
    """Run every detector over every input, OpenAI-result-shaped."""
    results: list[dict[str, Any]] = []
    for text in inputs:
        categories: dict[str, bool] = {}
        category_scores: dict[str, float] = {}
        for detector in detectors:
            for match in detector.analyse(text).matches:
                categories[match.category] = True
                category_scores[match.category] = max(
                    category_scores.get(match.category, 0.0),
                    match.score,
                )
        results.append(
            {
                "flagged": bool(categories),
                "categories": categories,
                "category_scores": category_scores,
            },
        )
    return results


def register_moderations_route(
    app: FastAPI,
    *,
    mode: str,
    upstream: str,
    client_factory: Callable[..., httpx.AsyncClient],
    forward_headers: Callable[[Request], dict[str, str]],
) -> None:
    """Attach ``POST /v1/moderations`` to the proxy app.

    Parameters
    ----------
    app : FastAPI
        The proxy application being assembled.
    mode : str
        ``"local"`` or ``"upstream"`` (validated by the caller).
    upstream : str
        Upstream base URL without a trailing slash.
    client_factory : Callable[..., httpx.AsyncClient]
        The proxy's client builder (carries the test transport).
    forward_headers : Callable[[Request], dict[str, str]]
        Extracts the auth headers to forward upstream.
    """
    # Detectors are built lazily on the first local-mode request so an
    # upstream-mode proxy never pays for them.
    local_detectors: list[ModerationDetector] | None = None

    @app.post("/v1/moderations")
    async def proxy_moderations(request: Request) -> JSONResponse:
        nonlocal local_detectors
        if mode == "upstream":
            body_bytes = await request.body()
            async with client_factory(timeout=120.0) as client:
                resp = await client.post(
                    f"{upstream}/v1/moderations",
                    content=body_bytes,
                    headers={
                        **forward_headers(request),
                        "Content-Type": "application/json",
                    },
                )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

        try:
            body = await request.json()
        except ValueError:
            body = None
        inputs = parse_moderation_input(body)
        if inputs is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": (
                            "'input' must be a string or a non-empty list of strings"
                        ),
                        "type": "invalid_request_error",
                    },
                },
            )
        if local_detectors is None:
            local_detectors = build_default_detectors()
        results = moderate_inputs(inputs, local_detectors)
        flagged_count = sum(r["flagged"] for r in results)
        if flagged_count:
            _log.info(
                "moderations: %d/%d inputs flagged",
                flagged_count,
                len(results),
            )
        return JSONResponse(
            content={
                "id": f"modr-{uuid.uuid4().hex}",
                "model": LOCAL_MODERATION_MODEL,
                "results": results,
            },
        )
