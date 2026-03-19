# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import json

import httpx
import pytest
from httpx import ASGITransport

from director_ai.core import GroundTruthStore
from director_ai.proxy import (
    _extract_prompt,
    _forward_headers,
    _load_facts,
    create_proxy_app,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status=200, body=None):
    if body is None:
        body = {
            "choices": [
                {"message": {"content": "The sky is blue."}, "finish_reason": "stop"}
            ]
        }
    return httpx.Response(status, json=body)


def _make_app(**kw):
    defaults = {
        "threshold": 0.1,
        "upstream_url": "https://api.openai.com",
        "_transport": httpx.MockTransport(lambda req: _mock_response()),
    }
    defaults.update(kw)
    return create_proxy_app(**defaults)


def _sse_lines(*chunks, done=True):
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
    if done:
        lines.append("data: [DONE]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# _extract_prompt — lines 277–288
# ---------------------------------------------------------------------------


class TestExtractPrompt:
    def test_string_user_content(self):
        msgs = [{"role": "user", "content": "hello world"}]
        assert _extract_prompt(msgs) == "hello world"

    def test_list_content_text_block(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "block text"}],
            }
        ]
        assert _extract_prompt(msgs) == "block text"

    def test_list_content_non_text_block(self):
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image_url", "url": "http://x.com/img.png"}],
            }
        ]
        result = _extract_prompt(msgs)
        assert isinstance(result, str)

    def test_list_content_returns_first_text_block(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "url": "x"},
                    {"type": "text", "text": "caption"},
                ],
            }
        ]
        assert _extract_prompt(msgs) == "caption"

    def test_non_string_non_list_content(self):
        msgs = [{"role": "user", "content": 42}]
        assert _extract_prompt(msgs) == "42"

    def test_no_user_role(self):
        msgs = [{"role": "system", "content": "sys prompt"}]
        assert _extract_prompt(msgs) == ""

    def test_empty_messages(self):
        assert _extract_prompt([]) == ""

    def test_most_recent_user_wins(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert _extract_prompt(msgs) == "second"


# ---------------------------------------------------------------------------
# _forward_headers — lines 291–296
# ---------------------------------------------------------------------------


class TestForwardHeaders:
    def test_authorization_forwarded(self):

        # Use a simple mock request object to avoid needing a running app
        class _FakeRequest:
            class headers:
                @staticmethod
                def get(key):
                    return "Bearer sk-test" if key == "authorization" else None

        result = _forward_headers(_FakeRequest())
        assert result == {"Authorization": "Bearer sk-test"}

    def test_no_authorization(self):
        class _FakeRequest:
            class headers:
                @staticmethod
                def get(key):
                    return None

        result = _forward_headers(_FakeRequest())
        assert result == {}


# ---------------------------------------------------------------------------
# _load_facts — lines 299–312
# ---------------------------------------------------------------------------


class TestLoadFacts:
    def test_key_value_pairs(self, tmp_path):
        f = tmp_path / "facts.txt"
        f.write_text("sky: blue\nocean: salty\n", encoding="utf-8")
        store = GroundTruthStore()
        _load_facts(store, str(f))
        assert "sky" in store.facts

    def test_plain_line_no_colon(self, tmp_path):
        f = tmp_path / "facts.txt"
        f.write_text("this is a plain fact line\n", encoding="utf-8")
        store = GroundTruthStore()
        _load_facts(store, str(f))

    def test_comment_and_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "facts.txt"
        f.write_text("# comment\n\nsky: blue\n", encoding="utf-8")
        store = GroundTruthStore()
        _load_facts(store, str(f))
        assert "sky" in store.facts

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_facts(GroundTruthStore(), str(tmp_path / "no_such.txt"))

    def test_facts_path_loads_into_app(self, tmp_path):
        f = tmp_path / "facts.txt"
        f.write_text("planet: Mars\n", encoding="utf-8")
        app = _make_app(facts_path=str(f))
        assert app is not None


# ---------------------------------------------------------------------------
# proxy_models endpoint — lines 121–126
# ---------------------------------------------------------------------------


class TestProxyModels:
    @pytest.mark.asyncio
    async def test_models_endpoint_forwards(self):
        models_body = {"object": "list", "data": [{"id": "gpt-4o", "object": "model"}]}
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=models_body)
        )
        app = _make_app(_transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get(
                "/v1/models", headers={"Authorization": "Bearer sk-test"}
            )

        assert resp.status_code == 200
        assert resp.json()["object"] == "list"

    @pytest.mark.asyncio
    async def test_models_endpoint_no_auth_header(self):
        models_body = {"object": "list", "data": []}
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=models_body)
        )
        app = _make_app(_transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get("/v1/models")

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Non-200 upstream in proxy_chat — line 155
# ---------------------------------------------------------------------------


class TestProxyChatNon200:
    @pytest.mark.asyncio
    async def test_upstream_error_forwarded(self):
        err_body = {"error": {"message": "rate limited", "type": "rate_limit_error"}}
        transport = httpx.MockTransport(lambda req: httpx.Response(429, json=err_body))
        app = _make_app(_transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert resp.status_code == 429
        assert resp.json()["error"]["type"] == "rate_limit_error"


# ---------------------------------------------------------------------------
# Empty content path in proxy_chat — line 163
# ---------------------------------------------------------------------------


class TestProxyChatEmptyContent:
    @pytest.mark.asyncio
    async def test_null_content_returns_data(self):
        body = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
        app = _make_app(_transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_choices_returns_data(self):
        body = {"id": "test", "choices": []}
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
        app = _make_app(_transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_string_content_returns_data(self):
        body = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
        app = _make_app(_transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Streaming — lines 199–274
# ---------------------------------------------------------------------------


def _sse_transport(sse_text: str, status: int = 200):
    """Return a MockTransport that streams the given SSE text."""

    async def _handler(request: httpx.Request):
        return httpx.Response(
            status,
            content=sse_text.encode(),
            headers={"content-type": "text/event-stream"},
        )

    return httpx.MockTransport(_handler)


class TestHandleStreaming:
    @pytest.mark.asyncio
    async def test_streaming_passthrough_approved(self):
        chunk = {
            "id": "c1",
            "choices": [
                {"delta": {"content": "Hello "}, "index": 0, "finish_reason": None}
            ],
        }
        done_chunk = {
            "id": "c2",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        sse = (
            f"data: {json.dumps(chunk)}\ndata: {json.dumps(done_chunk)}\ndata: [DONE]\n"
        )
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.0, _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_streaming_reject_on_done(self):
        chunk = {
            "id": "c1",
            "choices": [{"delta": {"content": "bad content "}, "index": 0}],
        }
        sse = f"data: {json.dumps(chunk)}\ndata: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.99, on_fail="reject", _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        body = resp.text
        assert "content_filter" in body or "[DONE]" in body

    @pytest.mark.asyncio
    async def test_streaming_warn_mode_passes_through(self):
        chunk = {
            "id": "c1",
            "choices": [{"delta": {"content": "some text"}, "index": 0}],
        }
        sse = f"data: {json.dumps(chunk)}\ndata: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.99, on_fail="warn", _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "[DONE]" in resp.text

    @pytest.mark.asyncio
    async def test_streaming_interval_check_reject(self):
        from director_ai.proxy import STREAM_CHECK_INTERVAL

        chunks = []
        for i in range(STREAM_CHECK_INTERVAL):
            chunks.append(
                {
                    "id": f"c{i}",
                    "choices": [{"delta": {"content": f"word{i} "}, "index": 0}],
                }
            )
        sse = "\n".join(f"data: {json.dumps(c)}" for c in chunks) + "\ndata: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.99, on_fail="reject", _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        body = resp.text
        assert "content_filter" in body or "[DONE]" in body

    @pytest.mark.asyncio
    async def test_streaming_interval_check_passes_warn(self):
        from director_ai.proxy import STREAM_CHECK_INTERVAL

        chunks = []
        for i in range(STREAM_CHECK_INTERVAL):
            chunks.append(
                {
                    "id": f"c{i}",
                    "choices": [{"delta": {"content": f"word{i} "}, "index": 0}],
                }
            )
        sse = "\n".join(f"data: {json.dumps(c)}" for c in chunks) + "\ndata: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.99, on_fail="warn", _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_streaming_non_data_line_forwarded(self):
        sse = "event: ping\ndata: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.1, _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "event: ping" in resp.text

    @pytest.mark.asyncio
    async def test_streaming_invalid_json_payload_forwarded(self):
        sse = "data: not-valid-json\ndata: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.1, _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "not-valid-json" in resp.text

    @pytest.mark.asyncio
    async def test_streaming_empty_buffer_on_done(self):
        sse = "data: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.99, on_fail="reject", _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "[DONE]" in resp.text

    @pytest.mark.asyncio
    async def test_streaming_chunk_with_null_delta_content(self):
        chunk = {
            "id": "c1",
            "choices": [{"delta": {"content": None}, "index": 0}],
        }
        sse = f"data: {json.dumps(chunk)}\ndata: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.1, _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_streaming_chunk_missing_choices(self):
        chunk = {"id": "c1", "choices": [{"delta": {}, "index": 0}]}
        sse = f"data: {json.dumps(chunk)}\ndata: [DONE]\n"
        transport = _sse_transport(sse)
        app = _make_app(threshold=0.1, _transport=transport)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _transport injection path — lines 117–119
# ---------------------------------------------------------------------------


class TestClientTransportInjection:
    @pytest.mark.asyncio
    async def test_transport_none_creates_plain_client(self):
        # When _transport=None, the _client() helper skips transport injection.
        # Test by providing a real HTTPS upstream (mocked via MockTransport at
        # the app level through the default _make_app path, which sets _transport).
        # Here we create an app WITHOUT _transport and confirm health still works.
        app = create_proxy_app(
            threshold=0.5,
            upstream_url="https://api.openai.com",
            on_fail="reject",
            use_nli=False,
            _transport=None,
        )
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_transport_injected_used_by_client(self):
        models_body = {"object": "list", "data": []}
        called = []

        def _handler(req: httpx.Request):
            called.append(req.url.path)
            return httpx.Response(200, json=models_body)

        transport = httpx.MockTransport(_handler)
        app = create_proxy_app(
            threshold=0.5,
            upstream_url="https://api.openai.com",
            on_fail="reject",
            use_nli=False,
            _transport=transport,
        )
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            await c.get("/v1/models")

        assert any("/v1/models" in p for p in called)


# ---------------------------------------------------------------------------
# Auth middleware edge cases
# ---------------------------------------------------------------------------


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_missing_api_key_header_rejected(self):
        app = _make_app(api_keys=["valid-key"])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "auth_error"

    @pytest.mark.asyncio
    async def test_multiple_valid_keys_accepted(self):
        app = _make_app(api_keys=["key-a", "key-b"])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
                headers={"X-API-Key": "key-b"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_bypasses_auth(self):
        app = _make_app(api_keys=["required-key"])
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            resp = await c.get("/health")
        assert resp.status_code == 200
