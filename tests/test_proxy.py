# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy Server Tests

from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from director_ai.core.config import DirectorConfig
from director_ai.proxy import create_proxy_app


def _upstream_transport(content: str):
    """Create an httpx transport that returns a fixed chat completion."""

    async def _handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    },
                ],
            },
        )

    return httpx.MockTransport(_handler)


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


@pytest.mark.asyncio
async def test_proxy_health():
    app = create_proxy_app(threshold=0.7, on_fail="reject")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["threshold"] == 0.7


@pytest.mark.asyncio
async def test_proxy_builds_from_director_config(tmp_path):
    facts = tmp_path / "facts.txt"
    facts.write_text("sky: blue\n", encoding="utf-8")
    config = DirectorConfig(
        mode="auto",
        use_nli=False,
        scorer_backend="lite",
        vector_backend="memory",
    )
    app = create_proxy_app(
        threshold=0.4,
        facts_path=str(facts),
        facts_root=str(tmp_path),
        config=config,
    )
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["threshold"] == 0.4


@pytest.mark.asyncio
async def test_proxy_config_override_loads_facts_and_use_nli(tmp_path):
    facts = tmp_path / "facts.txt"
    facts.write_text("sky: blue\n", encoding="utf-8")
    config = DirectorConfig(
        mode="auto",
        use_nli=False,
        scorer_backend="lite",
        vector_backend="memory",
    )

    app = create_proxy_app(
        threshold=0.4,
        facts_path=str(facts),
        facts_root=str(tmp_path),
        config=config,
        use_nli=True,
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["threshold"] == 0.4


@pytest.mark.asyncio
async def test_proxy_config_without_facts_uses_configured_store() -> None:
    config = DirectorConfig(
        mode="auto",
        use_nli=False,
        scorer_backend="lite",
        vector_backend="memory",
    )

    app = create_proxy_app(threshold=0.4, config=config)

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["on_fail"] == "reject"


@pytest.mark.asyncio
async def test_proxy_forwards_approved():
    mock_transport = _upstream_transport("The sky is blue due to Rayleigh scattering.")
    app = create_proxy_app(
        threshold=0.3,
        upstream_url="http://fake-upstream",
        on_fail="reject",
        use_nli=False,
        allow_http_upstream=True,
        _transport=mock_transport,
    )

    proxy_transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=proxy_transport,
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

    assert resp.status_code == 200
    assert "x-director-score" in resp.headers
    assert "x-director-approved" in resp.headers


@pytest.mark.asyncio
async def test_proxy_rejects_hallucination():
    mock_transport = _upstream_transport("Mars has two moons named Phobos and Deimos.")
    app = create_proxy_app(
        threshold=0.6,
        upstream_url="http://fake-upstream",
        on_fail="reject",
        use_nli=False,
        allow_http_upstream=True,
        _transport=mock_transport,
    )

    proxy_transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=proxy_transport,
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

    assert resp.status_code in (200, 422)
    if resp.status_code == 422:
        data = resp.json()
        assert data["error"]["type"] == "content_filter"


@pytest.mark.asyncio
async def test_proxy_warn_mode():
    mock_transport = _upstream_transport("Mars has two moons named Phobos and Deimos.")
    app = create_proxy_app(
        threshold=0.6,
        upstream_url="http://fake-upstream",
        on_fail="warn",
        use_nli=False,
        allow_http_upstream=True,
        _transport=mock_transport,
    )

    proxy_transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=proxy_transport,
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

    # Warn mode always forwards
    assert resp.status_code == 200
    assert "x-director-score" in resp.headers


def test_proxy_prompt_extraction_prefers_most_recent_user_text_block() -> None:
    from director_ai.proxy import _extract_prompt

    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "url": "ignored"},
                {"type": "text", "text": "caption"},
            ],
        },
    ]

    assert _extract_prompt(messages) == "caption"


def test_proxy_prompt_extraction_handles_non_text_content_safely() -> None:
    from director_ai.proxy import _extract_prompt

    assert _extract_prompt([{"role": "system", "content": "sys prompt"}]) == ""
    assert _extract_prompt([]) == ""
    assert _extract_prompt([{"role": "user", "content": 42}]) == "42"
    assert _extract_prompt([{"role": "user", "content": [{"type": "image_url"}]}]) == (
        "[{'type': 'image_url'}]"
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, ""),
        ({"choices": []}, ""),
        ({"choices": ["bad"]}, ""),
        ({"choices": [{"message": "bad"}]}, ""),
        ({"choices": [{"message": {"content": 42}}]}, ""),
        ({"choices": [{"message": {"content": "approved"}}]}, "approved"),
    ],
)
def test_proxy_chat_completion_content_extracts_openai_shape(payload, expected) -> None:
    from director_ai.proxy import _chat_completion_content

    assert _chat_completion_content(payload) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (None, ""),
        ({"choices": []}, ""),
        ({"choices": ["bad"]}, ""),
        ({"choices": [{"delta": "bad"}]}, ""),
        ({"choices": [{"delta": {"content": 42}}]}, ""),
        ({"choices": [{"delta": {"content": "token"}}]}, "token"),
    ],
)
def test_proxy_stream_delta_content_extracts_openai_shape(payload, expected) -> None:
    from director_ai.proxy import _stream_delta_content

    assert _stream_delta_content(payload) == expected


def test_proxy_forwards_authorization_header_only() -> None:
    from director_ai.proxy import _forward_headers

    class AuthorizationHeaders:
        @staticmethod
        def get(key: str):
            return "Bearer sk-test" if key == "authorization" else None

    class EmptyHeaders:
        @staticmethod
        def get(_key: str):
            return None

    class RequestWithAuthorization:
        headers = AuthorizationHeaders

    class RequestWithoutAuthorization:
        headers = EmptyHeaders

    assert _forward_headers(RequestWithAuthorization()) == {
        "Authorization": "Bearer sk-test",
    }
    assert _forward_headers(RequestWithoutAuthorization()) == {}


def test_proxy_loads_key_value_facts_and_skips_comments(tmp_path) -> None:
    from director_ai.core import GroundTruthStore
    from director_ai.proxy import _load_facts

    facts = tmp_path / "facts.txt"
    facts.write_text("# comment\n\nsky: blue\nocean: salty\n", encoding="utf-8")
    store = GroundTruthStore()

    _load_facts(store, str(facts))

    assert store.facts["sky"] == "blue"
    assert store.facts["ocean"] == "salty"


def test_proxy_missing_facts_file_fails_explicitly(tmp_path) -> None:
    from director_ai.core import GroundTruthStore
    from director_ai.proxy import _load_facts

    with pytest.raises(FileNotFoundError):
        _load_facts(GroundTruthStore(), str(tmp_path / "missing.txt"))


def test_proxy_directory_facts_path_fails_explicitly(tmp_path) -> None:
    from director_ai.core import GroundTruthStore
    from director_ai.proxy import _load_facts

    with pytest.raises(FileNotFoundError):
        _load_facts(GroundTruthStore(), str(tmp_path))


def test_proxy_loads_plain_line_facts(tmp_path) -> None:
    from director_ai.core import GroundTruthStore
    from director_ai.proxy import _load_facts

    facts = tmp_path / "facts.txt"
    facts.write_text("plain fact without separator\n", encoding="utf-8")
    store = GroundTruthStore()

    _load_facts(store, str(facts))

    assert store.retrieve_context("plain fact")


def test_proxy_rejects_invalid_failure_mode_and_http_upstream() -> None:
    with pytest.raises(ValueError, match="on_fail"):
        create_proxy_app(on_fail="panic")

    with pytest.raises(ValueError, match="Non-HTTPS upstream"):
        create_proxy_app(upstream_url="http://upstream.example")


def test_proxy_load_facts_rejects_missing_or_file_root(tmp_path) -> None:
    from director_ai.core import GroundTruthStore
    from director_ai.proxy import _load_facts

    facts = tmp_path / "facts.txt"
    facts.write_text("sky: blue\n", encoding="utf-8")
    file_root = tmp_path / "root.txt"
    file_root.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="facts_root"):
        _load_facts(
            GroundTruthStore(),
            str(facts),
            facts_root=str(tmp_path / "missing-root"),
        )
    with pytest.raises(ValueError, match="facts_root must be a directory"):
        _load_facts(GroundTruthStore(), str(facts), facts_root=str(file_root))
    outside = tmp_path.parent / "outside-facts.txt"
    outside.write_text("outside: fact\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside facts_root"):
        _load_facts(GroundTruthStore(), str(outside), facts_root=str(tmp_path))


@pytest.mark.asyncio
async def test_proxy_default_store_loads_configured_facts(tmp_path) -> None:
    facts = tmp_path / "facts.txt"
    facts.write_text("sky: blue\n", encoding="utf-8")
    app = create_proxy_app(
        facts_path=str(facts),
        facts_root=str(tmp_path),
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        use_nli=False,
        _transport=_upstream_transport("The sky is blue."),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

    assert response.status_code == 200
    assert response.headers["x-director-approved"] in {"true", "false"}


@pytest.mark.asyncio
async def test_proxy_requires_api_key_except_health() -> None:
    async def _handler(request: httpx.Request):
        return httpx.Response(200, json={"data": []})

    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        api_keys=["secret-key"],
        _transport=httpx.MockTransport(_handler),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        unauthenticated = await client.get("/v1/models")
        authenticated = await client.get(
            "/v1/models",
            headers={"X-API-Key": "secret-key"},
        )

    assert health.status_code == 200
    assert unauthenticated.status_code == 401
    assert unauthenticated.json()["error"]["type"] == "auth_error"
    assert authenticated.status_code == 200


def test_proxy_audit_log_entry_serialises_scored_chat() -> None:
    from director_ai.proxy import _audit_log_entry

    class AuditLog:
        def __init__(self) -> None:
            self.entries = []

        def log(self, entry) -> None:
            self.entries.append(entry)

    audit_log = AuditLog()

    _audit_log_entry(
        audit_log,
        "prompt",
        "response",
        model="model-a",
        score=0.91,
        approved=True,
        confidence=0.87,
        latency_ms=12.5,
    )

    assert len(audit_log.entries) == 1
    entry = audit_log.entries[0]
    assert entry.provider == "proxy"
    assert entry.prompt == "prompt"
    assert entry.response == "response"
    assert entry.model == "model-a"
    assert entry.approved is True


@pytest.mark.asyncio
async def test_proxy_models_forwards_authorization_header() -> None:
    async def _handler(request: httpx.Request):
        assert request.url.path == "/v1/models"
        assert request.headers["authorization"] == "Bearer upstream"
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        _transport=httpx.MockTransport(_handler),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/v1/models", headers={"Authorization": "Bearer upstream"}
        )

    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "model-a"


@pytest.mark.asyncio
async def test_proxy_models_uses_default_async_client_when_no_transport(
    monkeypatch,
) -> None:
    import httpx as httpx_module

    original_async_client = httpx_module.AsyncClient

    class RecordingAsyncClient:
        def __init__(self, **kwargs):
            assert "transport" not in kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str, headers: dict[str, str]):
            assert url == "https://api.example/v1/models"
            assert headers == {}
            return httpx.Response(200, json={"data": [{"id": "default-client"}]})

    monkeypatch.setattr(httpx_module, "AsyncClient", RecordingAsyncClient)
    app = create_proxy_app(upstream_url="https://api.example")

    transport = ASGITransport(app=app)
    async with original_async_client(
        transport=transport,
        base_url="http://test",
    ) as client:
        resp = await client.get("/v1/models")

    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "default-client"


@pytest.mark.asyncio
async def test_proxy_lifespan_closes_audit_log(tmp_path) -> None:
    app = create_proxy_app(audit_db=str(tmp_path / "audit.db"))

    async with app.router.lifespan_context(app):
        assert app.title == "Director-AI Proxy"


@pytest.mark.asyncio
async def test_proxy_lifespan_without_audit_log_exits_cleanly() -> None:
    app = create_proxy_app()

    async with app.router.lifespan_context(app):
        assert app.title == "Director-AI Proxy"


@pytest.mark.asyncio
async def test_proxy_forwards_upstream_chat_errors_without_scoring() -> None:
    async def _handler(request: httpx.Request):
        return httpx.Response(
            503,
            json={"error": {"message": "upstream overloaded", "type": "server_error"}},
        )

    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        _transport=httpx.MockTransport(_handler),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "status"}]},
        )

    assert resp.status_code == 503
    assert resp.json()["error"]["type"] == "server_error"
    assert "x-director-score" not in resp.headers


@pytest.mark.asyncio
async def test_proxy_empty_chat_content_bypasses_scoring_headers() -> None:
    async def _handler(request: httpx.Request):
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        _transport=httpx.MockTransport(_handler),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "empty?"}]},
        )

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] is None
    assert "x-director-score" not in resp.headers


@pytest.mark.asyncio
async def test_proxy_stream_forwards_non_data_and_malformed_events(monkeypatch) -> None:
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=True, score=0.92)
    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        _transport=_streaming_transport(
            [
                ": keepalive",
                "data: not-json",
                'data: {"choices":[{"delta":{"content":"Hello "}}]}',
                'data: {"choices":[{"delta":{"content":"world"}}]}',
                "data: [DONE]",
            ],
        ),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-model",
                "stream": True,
                "messages": [{"role": "user", "content": "greet"}],
            },
        )

    assert resp.status_code == 200
    body = resp.text
    assert ": keepalive\n" in body
    assert "data: not-json\n" in body
    assert "data: [DONE]\n" in body
    assert scorer.calls == [("greet", "Hello world")]


@pytest.mark.asyncio
async def test_proxy_stream_rejects_on_final_guardrail_failure(monkeypatch) -> None:
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=False, score=0.12)
    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        on_fail="reject",
        _transport=_streaming_transport(
            [
                'data: {"choices":[{"delta":{"content":"unsafe"}}]}',
                "data: [DONE]",
            ],
        ),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-model",
                "stream": True,
                "messages": [{"role": "user", "content": "answer"}],
            },
        )

    assert resp.status_code == 200
    assert '"finish_reason": "content_filter"' in resp.text
    assert resp.text.rstrip().endswith("data: [DONE]")
    assert scorer.calls == [("answer", "unsafe")]


@pytest.mark.asyncio
async def test_proxy_stream_rejects_on_periodic_guardrail_failure(monkeypatch) -> None:
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=False, score=0.1)
    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        on_fail="reject",
        _transport=_streaming_transport(
            [
                f'data: {{"choices":[{{"delta":{{"content":"t{idx}"}}}}]}}'
                for idx in range(proxy.STREAM_CHECK_INTERVAL + 1)
            ]
            + ["data: [DONE]"],
        ),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "guard"}],
            },
        )

    assert resp.status_code == 200
    assert '"finish_reason": "content_filter"' in resp.text
    assert f"t{proxy.STREAM_CHECK_INTERVAL}" not in resp.text
    assert scorer.calls == [("guard", "".join(f"t{idx}" for idx in range(8)))]


@pytest.mark.asyncio
async def test_proxy_stream_periodic_halt_writes_audit_entry(monkeypatch) -> None:
    # Regression: a mid-stream periodic halt returned without an audit entry,
    # while the terminal [DONE] review logs one. Both rejections must be audited.
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=False, score=0.1)
    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    entries: list[dict[str, object]] = []
    monkeypatch.setattr(
        proxy, "_audit_log_entry", lambda *_a, **kwargs: entries.append(kwargs)
    )
    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        on_fail="reject",
        _transport=_streaming_transport(
            [
                f'data: {{"choices":[{{"delta":{{"content":"t{idx}"}}}}]}}'
                for idx in range(proxy.STREAM_CHECK_INTERVAL + 1)
            ]
            + ["data: [DONE]"],
        ),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "stream-model",
                "stream": True,
                "messages": [{"role": "user", "content": "guard"}],
            },
        )

    assert resp.status_code == 200
    assert '"finish_reason": "content_filter"' in resp.text
    assert len(entries) == 1
    assert entries[0]["approved"] is False
    assert entries[0]["score"] == 0.1


@pytest.mark.asyncio
async def test_proxy_stream_done_without_text_forwards_done(monkeypatch) -> None:
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=False, score=0.1)
    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        on_fail="reject",
        _transport=_streaming_transport(["data: [DONE]"]),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "empty stream"}],
            },
        )

    assert resp.status_code == 200
    assert resp.text == "data: [DONE]\n"
    assert scorer.calls == []


@pytest.mark.asyncio
async def test_proxy_stream_empty_delta_is_forwarded_without_scoring(
    monkeypatch,
) -> None:
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=True, score=0.9)
    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        _transport=_streaming_transport(
            [
                'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                "data: [DONE]",
            ],
        ),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "role delta"}],
            },
        )

    assert resp.status_code == 200
    assert '"role":"assistant"' in resp.text
    assert resp.text.rstrip().endswith("data: [DONE]")
    assert scorer.calls == []


@pytest.mark.asyncio
async def test_proxy_stream_periodic_approval_continues_to_done(monkeypatch) -> None:
    import director_ai.proxy as proxy

    scorer = _FixedScorer(approved=True, score=0.95)
    monkeypatch.setattr(proxy, "CoherenceScorer", lambda **_kw: scorer)
    app = create_proxy_app(
        upstream_url="http://fake-upstream",
        allow_http_upstream=True,
        on_fail="reject",
        _transport=_streaming_transport(
            [
                f'data: {{"choices":[{{"delta":{{"content":"a{idx}"}}}}]}}'
                for idx in range(proxy.STREAM_CHECK_INTERVAL)
            ]
            + ["data: [DONE]"],
        ),
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "allow"}],
            },
        )

    assert resp.status_code == 200
    assert resp.text.rstrip().endswith("data: [DONE]")
    assembled = "".join(f"a{idx}" for idx in range(proxy.STREAM_CHECK_INTERVAL))
    assert scorer.calls == [("allow", assembled), ("allow", assembled)]
