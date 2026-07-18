# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy Streaming Disclosure Handler

"""SSE streaming with periodic mid-stream reviews for the guardrail proxy.

Implements both disclosure modes (KIMI2-A): ``immediate`` forwards each
chunk as it arrives (a halt stops future tokens only), ``buffered``
withholds chunks until they pass a review and fails closed on overflow
(KIMI3-H2) or a scorer exception (KIMI3-H5).
"""

from __future__ import annotations

import json
import logging
import time as _time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from director_ai._proxy_audit import _audit_log_entry
from director_ai._proxy_content import _stream_chat_content, _stream_text_content

if TYPE_CHECKING:
    from fastapi.responses import StreamingResponse

_log = logging.getLogger("DirectorAI.Proxy")

STREAM_CHECK_INTERVAL = 8

# Buffered-mode safety cap on the withheld ``pending`` window (KIMI3-H2). A
# stream can withhold many lines without ever producing reviewable content — a
# flood of content-less or non-data lines — which would grow the process memory
# without bound and never trigger a review. When the withheld window exceeds
# either cap the stream fails closed with a halt instead of buffering
# indefinitely; nothing withheld is disclosed.
STREAM_MAX_PENDING_LINES = 512
STREAM_MAX_PENDING_CHARS = 256 * 1024

# Streaming disclosure modes (KIMI2-A). ``immediate`` forwards every chunk as
# it arrives and a mid-stream halt only stops FUTURE tokens — content emitted
# before the halt has already reached the client (early termination with
# partial disclosure). ``buffered`` withholds chunks until they pass a review,
# so a halted stream discloses nothing unreviewed, at a latency cost of up to
# ``STREAM_CHECK_INTERVAL`` chunks.
STREAM_DISCLOSURE_MODES = ("immediate", "buffered")


async def _handle_streaming(
    body: dict[str, Any],
    headers: dict[str, str],
    upstream: str,
    prompt: str,
    scorer: Any,
    on_fail: str,
    transport: Any = None,
    audit_log: Any = None,
    endpoint: str = "/v1/chat/completions",
    task_type: str = "chat",
    stream_disclosure: str = "immediate",
) -> StreamingResponse:
    """Proxy an SSE stream with periodic mid-stream reviews.

    ``stream_disclosure="immediate"`` forwards each line as it arrives; a
    halt stops future tokens only (partial disclosure — content already
    emitted has reached the client). ``"buffered"`` holds lines in a
    pending window and releases them only after the accumulated content
    passes a review; a halt discards the unreleased window, and the
    terminal ``[DONE]`` review gates the final release, so a rejected
    stream never discloses unreviewed content. If the upstream drops
    without ``[DONE]``, the unreviewed pending window is discarded
    (fail-closed).
    """
    import httpx
    from fastapi.responses import StreamingResponse

    # Legacy completions stream text deltas at ``choices[0].text`` and a
    # halt chunk must mirror that shape; chat streams use ``delta`` and may
    # carry tool-call deltas that must be reviewed alongside content.
    legacy = endpoint == "/v1/completions"
    extract_content = _stream_text_content if legacy else _stream_chat_content
    halt_choice: dict[str, Any] = {"finish_reason": "content_filter", "index": 0}
    if legacy:
        halt_choice["text"] = ""
    else:
        halt_choice["delta"] = {}
    buffered = stream_disclosure == "buffered"

    async def _stream() -> AsyncIterator[str]:
        buffer: list[str] = []
        pending: list[str] = []  # withheld output lines (buffered mode)
        chunk_count = 0
        model_name = body.get("model", "unknown")
        t0 = _time.monotonic()

        def _pending_overflow() -> bool:
            """Report whether the withheld window has exceeded either cap."""
            return (
                len(pending) > STREAM_MAX_PENDING_LINES
                or sum(len(entry) for entry in pending) > STREAM_MAX_PENDING_CHARS
            )

        def _withhold(text: str) -> bool:
            """Hold *text* in the buffered window; return True on overflow."""
            pending.append(text)
            return _pending_overflow()

        def _fail_closed_frames() -> list[str]:
            """Drop the withheld window and return the halt frames.

            The shared fail-closed action for a pending-window overflow
            (KIMI3-H2) or a scorer exception mid-stream (KIMI3-H5): nothing
            withheld is disclosed — the window is cleared and the client receives
            only the halt marker, with the abort recorded to the audit log like
            any other terminal halt.
            """
            reviewed = "".join(buffer)
            _audit_log_entry(
                audit_log,
                prompt,
                reviewed,
                model=model_name,
                score=0.0,
                approved=False,
                confidence=0.0,
                latency_ms=(_time.monotonic() - t0) * 1000,
                task_type=task_type,
            )
            pending.clear()
            halt = {"choices": [dict(halt_choice)]}
            return [f"data: {json.dumps(halt)}\n", "data: [DONE]\n"]

        async with (
            httpx.AsyncClient(timeout=120.0, transport=transport) as client,
            client.stream(
                "POST",
                f"{upstream}{endpoint}",
                json=body,
                headers=headers,
            ) as resp,
        ):
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    # Non-data lines are withheld too in buffered mode so
                    # the released stream preserves the upstream ordering.
                    if buffered:
                        if _withhold(line + "\n"):
                            for _frame in _fail_closed_frames():
                                yield _frame
                            return
                    else:
                        yield line + "\n"
                    continue

                payload = line[6:]
                if payload.strip() == "[DONE]":
                    text = "".join(buffer)
                    if text:
                        try:
                            approved, _cs = scorer.review(prompt, text)
                        except Exception:  # noqa: BLE001 - any scorer error fails closed
                            _log.exception(
                                "scorer.review failed at stream end; failing closed"
                            )
                            for _frame in _fail_closed_frames():
                                yield _frame
                            return
                        latency_ms = (_time.monotonic() - t0) * 1000
                        _audit_log_entry(
                            audit_log,
                            prompt,
                            text,
                            model=model_name,
                            score=_cs.score,
                            approved=approved,
                            confidence=getattr(_cs, "verdict_confidence", 0.0),
                            latency_ms=latency_ms,
                            task_type=task_type,
                        )
                        if not approved and on_fail == "reject":
                            # Buffered: the withheld window is discarded, so
                            # the client receives only the halt marker.
                            pending.clear()
                            halt = {"choices": [dict(halt_choice)]}
                            yield f"data: {json.dumps(halt)}\n"
                            yield "data: [DONE]\n"
                            return
                    # Final review passed (or nothing to review): release
                    # any withheld tail before the terminal marker. The
                    # window holds only SSE framing lines and content lines
                    # from the reviewed buffer — malformed data lines were
                    # dropped at parse time (KIMI3-H3), so nothing here
                    # carries unreviewed payload bytes.
                    for held in pending:
                        yield held
                    pending.clear()
                    yield line + "\n"
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    if buffered:
                        # KIMI3-H3: an unparseable data line can never pass
                        # review — its bytes never reach the scorer — so the
                        # buffered contract (nothing unreviewed reaches the
                        # client) means dropping it, not withholding it for
                        # a later release with the reviewed window.
                        _log.warning(
                            "buffered stream: dropping malformed upstream "
                            "data line (%d bytes)",
                            len(line),
                        )
                    else:
                        yield line + "\n"
                    continue

                delta = extract_content(chunk)

                if delta:
                    buffer.append(delta)
                    chunk_count += 1

                    if chunk_count % STREAM_CHECK_INTERVAL == 0:
                        text = "".join(buffer)
                        try:
                            approved, _cs = scorer.review(prompt, text)
                        except Exception:  # noqa: BLE001 - any scorer error fails closed
                            _log.exception(
                                "scorer.review failed mid-stream; failing closed"
                            )
                            for _frame in _fail_closed_frames():
                                yield _frame
                            return
                        if not approved and on_fail == "reject":
                            # Record the mid-stream halt: like the terminal
                            # [DONE] review, a rejection must leave an audit entry.
                            latency_ms = (_time.monotonic() - t0) * 1000
                            _audit_log_entry(
                                audit_log,
                                prompt,
                                text,
                                model=model_name,
                                score=_cs.score,
                                approved=approved,
                                confidence=getattr(_cs, "verdict_confidence", 0.0),
                                latency_ms=latency_ms,
                                task_type=task_type,
                            )
                            pending.clear()
                            halt = {"choices": [dict(halt_choice)]}
                            yield f"data: {json.dumps(halt)}\n"
                            yield "data: [DONE]\n"
                            return
                        if buffered:
                            # Reviewed clean: release the withheld window
                            # (including this line) up to the review point.
                            pending.append(line + "\n")
                            for held in pending:
                                yield held
                            pending.clear()
                            continue

                if buffered:
                    if _withhold(line + "\n"):
                        for _frame in _fail_closed_frames():
                            yield _frame
                        return
                else:
                    yield line + "\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
