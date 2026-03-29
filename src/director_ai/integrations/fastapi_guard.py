# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FastAPI Middleware

"""ASGI middleware that scores JSON responses for hallucination.

Usage::

    from director_ai.integrations.fastapi_guard import DirectorGuard
    app.add_middleware(DirectorGuard, facts={"policy": "30-day refunds"})
"""

from __future__ import annotations

import json
import logging

from director_ai.core import CoherenceScorer, GroundTruthStore

_log = logging.getLogger("DirectorAI.FastAPIGuard")

_RESPONSE_TEXT_KEYS = ("choices", "response", "text", "output", "content")
_REQUEST_PROMPT_KEYS = ("prompt", "query", "question", "input", "messages")


class DirectorGuard:
    """ASGI middleware that adds hallucination scoring to JSON responses.

    Parameters
    ----------
    app : ASGI application
    threshold : float
        Coherence threshold.
    facts : dict | None
        Key-value facts for the ground truth store.
    store : GroundTruthStore | None
        Pre-built store (takes precedence over *facts*).
    use_nli : bool | None
        Enable NLI model.
    paths : list[str] | None
        URL paths to score. ``None`` scores all POST responses.
    on_fail : str
        ``"warn"`` adds headers only. ``"reject"`` returns 422.

    """

    def __init__(
        self,
        app,
        *,
        threshold: float = 0.6,
        facts: dict[str, str] | None = None,
        store: GroundTruthStore | None = None,
        use_nli: bool | None = None,
        paths: list[str] | None = None,
        on_fail: str = "warn",
    ):
        if on_fail not in ("warn", "reject"):
            raise ValueError(f"on_fail must be 'warn' or 'reject', got {on_fail!r}")
        self.app = app
        self.paths = set(paths) if paths else None
        self.on_fail = on_fail
        self.threshold = threshold

        gts = store or GroundTruthStore()
        if facts:
            for k, v in facts.items():
                gts.add(k, v)
        self.scorer = CoherenceScorer(
            threshold=threshold,
            ground_truth_store=gts,
            use_nli=use_nli,
        )

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "GET")
        path = scope.get("path", "")

        if method != "POST" or (self.paths and path not in self.paths):
            await self.app(scope, receive, send)
            return

        # Pre-read full request body, then replay it for the inner app
        request_chunks = []
        while True:
            msg = await receive()
            if msg["type"] == "http.request":
                request_chunks.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    break
            elif msg["type"] == "http.disconnect":
                break

        request_body = b"".join(request_chunks)
        body_sent = False

        async def replay_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": request_body,
                    "more_body": False,
                }
            return await receive()

        # Buffer response
        response_status = 200
        response_headers_raw: list = []
        response_body = bytearray()

        async def buffered_send(message):
            nonlocal response_status, response_headers_raw
            if message["type"] == "http.response.start":
                response_status = message.get("status", 200)
                response_headers_raw = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))

        await self.app(scope, replay_receive, buffered_send)

        # Try to extract and score
        prompt = _extract_request_prompt(bytes(request_body))
        response_text = _extract_response_text(bytes(response_body))

        extra_headers = []
        reject = False

        if prompt and response_text:
            approved, cs = self.scorer.review(prompt, response_text)
            extra_headers = [
                (b"x-director-score", f"{cs.score:.4f}".encode()),
                (b"x-director-approved", str(approved).lower().encode()),
            ]
            if not approved and self.on_fail == "reject":
                reject = True
                reject_body = json.dumps(
                    {
                        "error": {
                            "message": "Hallucination detected by Director-AI",
                            "type": "content_filter",
                            "score": cs.score,
                            "threshold": self.threshold,
                        },
                    },
                ).encode()

        if reject:
            await send(
                {
                    "type": "http.response.start",
                    "status": 422,
                    "headers": [
                        (b"content-type", b"application/json"),
                        *extra_headers,
                    ],
                },
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": reject_body,
                },
            )
        else:
            merged_headers = response_headers_raw + extra_headers
            await send(
                {
                    "type": "http.response.start",
                    "status": response_status,
                    "headers": merged_headers,
                },
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": bytes(response_body),
                },
            )


def _extract_request_prompt(body: bytes) -> str:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    # OpenAI messages format
    messages = data.get("messages")
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                c = msg.get("content", "")
                return str(c) if isinstance(c, str) else ""
    for key in _REQUEST_PROMPT_KEYS:
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _extract_response_text(body: bytes) -> str:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    if not isinstance(data, dict):
        return ""
    # OpenAI choices format
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0]
        if isinstance(msg, dict):
            message = msg.get("message")
            if isinstance(message, dict):
                c = message.get("content", "")
                if isinstance(c, str) and c:
                    return c
    for key in ("response", "text", "output", "content"):
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    return ""
