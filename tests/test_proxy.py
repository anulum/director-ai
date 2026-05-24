# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy Server Tests

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


def test_proxy_forwards_authorization_header_only() -> None:
    from director_ai.proxy import _forward_headers

    class RequestWithAuthorization:
        class headers:
            @staticmethod
            def get(key: str):
                return "Bearer sk-test" if key == "authorization" else None

    class RequestWithoutAuthorization:
        class headers:
            @staticmethod
            def get(_key: str):
                return None

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
