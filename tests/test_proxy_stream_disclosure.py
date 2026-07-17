# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Disclosure Modes (KIMI2-A)
"""Behavioural contracts for the proxy ``stream_disclosure`` modes.

KIMI proved end-to-end that in the immediate mode a mid-stream halt leaks the
tokens emitted before the halt ("800 mg of ibuprofen is safe per" reached the
client). The ``buffered`` mode inverts that proof: chunks are withheld until
the accumulated content passes a review, a halt discards the unreleased
window, and the terminal ``[DONE]`` review gates the final release — so a
rejected stream never discloses unreviewed content. These tests pin both
modes, the fail-closed upstream-drop path, and the config validation.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from director_ai.proxy import create_proxy_app

pytestmark = pytest.mark.asyncio


class _FixedScorer:
    def __init__(self, approved: bool = True, score: float = 0.9):
        self.approved = approved
        self.score = score
        self.calls: list[tuple[str, str]] = []

    def review(self, prompt: str, content: str):
        self.calls.append((prompt, content))
        return self.approved, SimpleNamespace(
            score=self.score,
            verdict_confidence=0.75,
        )


def _streaming_transport(lines: list[str]) -> httpx.MockTransport:
    async def _handler(request: httpx.Request):
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="\n".join(lines) + "\n",
        )

    return httpx.MockTransport(_handler)


def _app(scorer, lines: list[str], monkeypatch, *, disclosure: str):
    import director_ai.proxy as proxy

    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    return create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        on_fail="reject",
        stream_disclosure=disclosure,
        _transport=_streaming_transport(lines),
    )


async def _post_stream(app, prompt: str = "ask") -> str:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-model",
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
    assert resp.status_code == 200
    return resp.text


def _content_lines(n: int) -> list[str]:
    return [
        f'data: {{"choices":[{{"delta":{{"content":"t{idx}"}}}}]}}' for idx in range(n)
    ]


async def test_buffered_halted_stream_discloses_no_content(monkeypatch) -> None:
    # The inverse of KIMI's proof: the periodic review rejects, and the client
    # receives NONE of the withheld tokens — only the halt marker.
    scorer = _FixedScorer(approved=False, score=0.1)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(9) + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    for idx in range(9):
        assert f"t{idx}" not in body
    assert '"finish_reason": "content_filter"' in body
    assert body.rstrip().endswith("data: [DONE]")


async def test_buffered_clean_stream_releases_everything_in_order(monkeypatch) -> None:
    scorer = _FixedScorer(approved=True, score=0.95)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(10) + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    positions = [body.index(f'"content":"t{idx}"') for idx in range(10)]
    assert positions == sorted(positions)
    assert body.rstrip().endswith("data: [DONE]")


async def test_buffered_final_review_rejection_discards_the_tail(monkeypatch) -> None:
    # Fewer chunks than the periodic interval: the only review is at [DONE],
    # and its rejection must withhold the entire stream.
    scorer = _FixedScorer(approved=False, score=0.2)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(3) + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    for idx in range(3):
        assert f"t{idx}" not in body
    assert '"finish_reason": "content_filter"' in body
    assert scorer.calls == [("ask", "t0t1t2")]


async def test_buffered_final_review_pass_releases_the_tail(monkeypatch) -> None:
    scorer = _FixedScorer(approved=True, score=0.9)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(3) + ["data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    for idx in range(3):
        assert f'"content":"t{idx}"' in body
    assert body.rstrip().endswith("data: [DONE]")


async def test_buffered_upstream_drop_without_done_is_fail_closed(monkeypatch) -> None:
    # No [DONE]: the unreviewed pending window is discarded rather than
    # released unreviewed.
    scorer = _FixedScorer(approved=True, score=0.9)
    body = await _post_stream(
        _app(scorer, _content_lines(3), monkeypatch, disclosure="buffered"),
    )

    for idx in range(3):
        assert f"t{idx}" not in body
    assert scorer.calls == []


async def test_buffered_withholds_non_data_lines_for_ordering(monkeypatch) -> None:
    scorer = _FixedScorer(approved=True, score=0.9)
    body = await _post_stream(
        _app(
            scorer,
            [": keepalive", "data: not-json", *_content_lines(2), "data: [DONE]"],
            monkeypatch,
            disclosure="buffered",
        ),
    )

    # Everything is released at the passing [DONE] review, upstream order kept.
    assert body.index(": keepalive") < body.index("data: not-json")
    assert body.index("data: not-json") < body.index('"content":"t0"')
    assert body.rstrip().endswith("data: [DONE]")


async def test_immediate_default_still_discloses_pre_halt_tokens(monkeypatch) -> None:
    # Pin today's default: the tokens streamed BEFORE the rejecting review
    # reach the client (the documented partial-disclosure behaviour), while
    # the line that triggered the review and everything after it do not —
    # proving the buffered fix did not change the default mode.
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=False, score=0.1)
    body = await _post_stream(
        _app(
            scorer,
            _content_lines(proxy.STREAM_CHECK_INTERVAL + 1) + ["data: [DONE]"],
            monkeypatch,
            disclosure="immediate",
        ),
    )

    for idx in range(proxy.STREAM_CHECK_INTERVAL - 1):
        assert f'"content":"t{idx}"' in body
    assert f"t{proxy.STREAM_CHECK_INTERVAL - 1}" not in body
    assert f"t{proxy.STREAM_CHECK_INTERVAL}" not in body
    assert '"finish_reason": "content_filter"' in body


async def test_create_proxy_app_rejects_unknown_disclosure_mode() -> None:
    with pytest.raises(ValueError, match="stream_disclosure"):
        create_proxy_app(stream_disclosure="bogus")
