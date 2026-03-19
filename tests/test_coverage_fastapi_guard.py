# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from director_ai.integrations.fastapi_guard import (
    DirectorGuard,
    _extract_request_prompt,
    _extract_response_text,
)

# ---------------------------------------------------------------------------
# DirectorGuard.__init__ branches
# ---------------------------------------------------------------------------


def test_invalid_on_fail_raises():
    app = MagicMock()
    with pytest.raises(ValueError, match="on_fail must be"):
        DirectorGuard(app, on_fail="block")


def test_init_with_store_skips_fact_loop():
    from director_ai.core import GroundTruthStore

    app = MagicMock()
    store = GroundTruthStore()
    guard = DirectorGuard(app, store=store)
    assert guard.paths is None
    assert guard.on_fail == "warn"


def test_init_with_facts_adds_entries():
    app = MagicMock()
    guard = DirectorGuard(app, facts={"k": "v"}, use_nli=False)
    assert guard.scorer is not None


def test_init_paths_set_when_provided():
    app = MagicMock()
    guard = DirectorGuard(app, paths=["/a", "/b"])
    assert guard.paths == {"/a", "/b"}


def test_init_paths_none_when_not_provided():
    app = MagicMock()
    guard = DirectorGuard(app)
    assert guard.paths is None


# ---------------------------------------------------------------------------
# __call__: non-http scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_http_scope_passes_through():
    inner_called = []

    async def inner_app(scope, receive, send):
        inner_called.append(scope["type"])

    guard = DirectorGuard(inner_app)
    await guard({"type": "websocket"}, None, None)
    assert inner_called == ["websocket"]


# ---------------------------------------------------------------------------
# __call__: GET request passes through without scoring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_request_passthrough():
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from httpx import ASGITransport

    app = FastAPI()

    @app.get("/data")
    async def _():
        return JSONResponse({"text": "hello"})

    guard = DirectorGuard(app, use_nli=False)
    transport = ASGITransport(app=guard)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/data")

    assert resp.status_code == 200
    assert "x-director-score" not in resp.headers


# ---------------------------------------------------------------------------
# __call__: path filter excludes non-listed path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_path_filter_excludes():
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from httpx import ASGITransport

    app = FastAPI()

    @app.post("/other")
    async def _():
        return JSONResponse({"text": "hello"})

    guard = DirectorGuard(app, use_nli=False, paths=["/only-this"])
    transport = ASGITransport(app=guard)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/other", json={"prompt": "test"})

    assert resp.status_code == 200
    assert "x-director-score" not in resp.headers


# ---------------------------------------------------------------------------
# __call__: http.disconnect during body read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_during_body_read():
    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"text":"ok"}'})

    disconnect_msg = {"type": "http.disconnect"}

    async def receive():
        return disconnect_msg

    sent = []

    async def send(msg):
        sent.append(msg)

    guard = DirectorGuard(inner_app, use_nli=False)
    scope = {"type": "http", "method": "POST", "path": "/chat"}
    await guard(scope, receive, send)
    assert any(m["type"] == "http.response.start" for m in sent)


# ---------------------------------------------------------------------------
# __call__: replay_receive falls back to original receive after first call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_receive_second_call_hits_original():
    second_calls = []

    async def inner_app(scope, receive, send):
        await receive()
        msg = await receive()
        second_calls.append(msg)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"text":"ok"}'})

    request_calls = [0]

    async def receive():
        request_calls[0] += 1
        if request_calls[0] == 1:
            return {
                "type": "http.request",
                "body": b'{"prompt":"hi"}',
                "more_body": False,
            }
        return {"type": "http.disconnect"}

    sent = []

    async def send_outer(msg):
        sent.append(msg)

    guard = DirectorGuard(inner_app, use_nli=False)
    scope = {"type": "http", "method": "POST", "path": "/chat"}
    await guard(scope, receive, send_outer)
    assert second_calls[0] == {"type": "http.disconnect"}


# ---------------------------------------------------------------------------
# __call__: response scoring with approved result (warn mode)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scoring_adds_headers_on_approved():
    from director_ai.core.types import CoherenceScore

    async def inner_app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {"type": "http.response.body", "body": b'{"text":"Earth orbits the sun."}'}
        )

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"prompt":"Where does Earth orbit?"}',
            "more_body": False,
        }

    sent = []

    async def send_outer(msg):
        sent.append(msg)

    cs = CoherenceScore(score=0.9, approved=True, h_logical=0.1, h_factual=0.1)
    guard = DirectorGuard(inner_app, use_nli=False)

    with patch.object(guard.scorer, "review", return_value=(True, cs)):
        scope = {"type": "http", "method": "POST", "path": "/chat"}
        await guard(scope, receive, send_outer)

    start_msg = next(m for m in sent if m["type"] == "http.response.start")
    header_names = [k for k, _ in start_msg["headers"]]
    assert b"x-director-score" in header_names
    assert b"x-director-approved" in header_names
    assert start_msg["status"] == 200


# ---------------------------------------------------------------------------
# __call__: reject mode returns 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reject_mode_returns_422():
    from director_ai.core.types import CoherenceScore

    async def inner_app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"text":"wrong answer"}'})

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"prompt":"question"}',
            "more_body": False,
        }

    sent = []

    async def send_outer(msg):
        sent.append(msg)

    cs = CoherenceScore(score=0.3, approved=False, h_logical=0.8, h_factual=0.8)
    guard = DirectorGuard(inner_app, use_nli=False, on_fail="reject")

    with patch.object(guard.scorer, "review", return_value=(False, cs)):
        scope = {"type": "http", "method": "POST", "path": "/chat"}
        await guard(scope, receive, send_outer)

    start_msg = next(m for m in sent if m["type"] == "http.response.start")
    assert start_msg["status"] == 422

    body_msg = next(m for m in sent if m["type"] == "http.response.body")
    payload = json.loads(body_msg["body"])
    assert payload["error"]["type"] == "content_filter"
    assert payload["error"]["score"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# __call__: no prompt or no response text → no scoring headers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_prompt_skips_scoring():
    async def inner_app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b'{"text":"answer"}'})

    async def receive():
        return {"type": "http.request", "body": b"not json at all", "more_body": False}

    sent = []

    async def send_outer(msg):
        sent.append(msg)

    guard = DirectorGuard(inner_app, use_nli=False)
    scope = {"type": "http", "method": "POST", "path": "/chat"}
    await guard(scope, receive, send_outer)

    start_msg = next(m for m in sent if m["type"] == "http.response.start")
    header_names = [k for k, _ in start_msg["headers"]]
    assert b"x-director-score" not in header_names


@pytest.mark.asyncio
async def test_no_response_text_skips_scoring():
    async def inner_app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"not json"})

    async def receive():
        return {"type": "http.request", "body": b'{"prompt":"hi"}', "more_body": False}

    sent = []

    async def send_outer(msg):
        sent.append(msg)

    guard = DirectorGuard(inner_app, use_nli=False)
    scope = {"type": "http", "method": "POST", "path": "/chat"}
    await guard(scope, receive, send_outer)

    start_msg = next(m for m in sent if m["type"] == "http.response.start")
    header_names = [k for k, _ in start_msg["headers"]]
    assert b"x-director-score" not in header_names


# ---------------------------------------------------------------------------
# _extract_request_prompt
# ---------------------------------------------------------------------------


def test_extract_prompt_invalid_json():
    assert _extract_request_prompt(b"not json") == ""


def test_extract_prompt_not_dict():
    assert _extract_request_prompt(json.dumps([1, 2, 3]).encode()) == ""


def test_extract_prompt_openai_messages_user_last():
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        }
    ).encode()
    assert _extract_request_prompt(body) == "What is 2+2?"


def test_extract_prompt_openai_messages_no_user():
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Be helpful."},
            ]
        }
    ).encode()
    assert _extract_request_prompt(body) == ""


def test_extract_prompt_openai_messages_user_non_string_content():
    body = json.dumps(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    ).encode()
    assert _extract_request_prompt(body) == ""


def test_extract_prompt_messages_not_list():
    body = json.dumps({"messages": "not a list", "prompt": "fallback"}).encode()
    assert _extract_request_prompt(body) == "fallback"


def test_extract_prompt_standard_keys():
    for key in ("prompt", "query", "question", "input"):
        body = json.dumps({key: f"value for {key}"}).encode()
        assert _extract_request_prompt(body) == f"value for {key}"


def test_extract_prompt_empty_string_value_skipped():
    body = json.dumps({"prompt": "", "query": "real query"}).encode()
    assert _extract_request_prompt(body) == "real query"


def test_extract_prompt_no_matching_key():
    body = json.dumps({"data": "irrelevant"}).encode()
    assert _extract_request_prompt(body) == ""


def test_extract_prompt_unicode_decode_error():
    assert _extract_request_prompt(b"\xff\xfe") == ""


# ---------------------------------------------------------------------------
# _extract_response_text
# ---------------------------------------------------------------------------


def test_extract_response_invalid_json():
    assert _extract_response_text(b"not json") == ""


def test_extract_response_not_dict():
    assert _extract_response_text(json.dumps([1, 2]).encode()) == ""


def test_extract_response_openai_choices_with_message():
    body = json.dumps(
        {"choices": [{"message": {"content": "Paris is the capital of France."}}]}
    ).encode()
    assert _extract_response_text(body) == "Paris is the capital of France."


def test_extract_response_openai_choices_empty_content():
    body = json.dumps({"choices": [{"message": {"content": ""}}]}).encode()
    result = _extract_response_text(body)
    assert result == ""


def test_extract_response_openai_choices_missing_message():
    body = json.dumps({"choices": [{"text": "raw text"}]}).encode()
    assert _extract_response_text(body) == ""


def test_extract_response_openai_choices_empty_list():
    body = json.dumps({"choices": [], "text": "fallback"}).encode()
    assert _extract_response_text(body) == "fallback"


def test_extract_response_standard_keys():
    for key in ("response", "text", "output", "content"):
        body = json.dumps({key: f"value {key}"}).encode()
        assert _extract_response_text(body) == f"value {key}"


def test_extract_response_empty_string_value_skipped():
    body = json.dumps({"response": "", "text": "real"}).encode()
    assert _extract_response_text(body) == "real"


def test_extract_response_no_matching_key():
    body = json.dumps({"data": "irrelevant"}).encode()
    assert _extract_response_text(body) == ""


def test_extract_response_unicode_decode_error():
    assert _extract_response_text(b"\xff\xfe") == ""


def test_extract_response_choices_not_list():
    body = json.dumps({"choices": "not a list", "text": "fallback"}).encode()
    assert _extract_response_text(body) == "fallback"


def test_extract_response_message_content_not_string():
    body = json.dumps(
        {
            "choices": [{"message": {"content": 42}}],
            "text": "fallback",
        }
    ).encode()
    assert _extract_response_text(body) == "fallback"
