# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy /v1/moderations Tests

"""Contract tests for the proxy ``/v1/moderations`` route.

Everything runs through the public surface — ``create_proxy_app`` with
the ``moderations`` parameter — so the tests pin the behaviour clients
see, not module internals.
"""

import httpx
import pytest
from httpx import ASGITransport

from director_ai.proxy import create_proxy_app


def _local_app(**kwargs):
    return create_proxy_app(threshold=0.6, on_fail="reject", **kwargs)


async def _post_moderations(app, payload):
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post("/v1/moderations", json=payload)


class TestLocalModerations:
    @pytest.mark.asyncio
    async def test_clean_text_not_flagged(self):
        resp = await _post_moderations(
            _local_app(),
            {"input": "The sky is blue today."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"].startswith("modr-")
        assert data["model"] == "director-ai-local-moderation"
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["flagged"] is False
        assert result["categories"] == {}
        assert result["category_scores"] == {}

    @pytest.mark.asyncio
    async def test_toxic_text_flagged_with_category(self):
        resp = await _post_moderations(
            _local_app(),
            {"input": "You should kill yourself."},
        )
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["flagged"] is True
        assert result["categories"].get("self_harm_encouragement") is True
        assert result["category_scores"]["self_harm_encouragement"] == 1.0

    @pytest.mark.asyncio
    async def test_pii_text_flagged_with_category(self):
        resp = await _post_moderations(
            _local_app(),
            {"input": "Contact me at jane.doe@example.com please."},
        )
        result = resp.json()["results"][0]
        assert result["flagged"] is True
        assert result["categories"].get("email") is True

    @pytest.mark.asyncio
    async def test_list_input_returns_one_result_per_item(self):
        resp = await _post_moderations(
            _local_app(),
            {"input": ["A calm sentence.", "I will kill you tonight."]},
        )
        results = resp.json()["results"]
        assert len(results) == 2
        assert results[0]["flagged"] is False
        assert results[1]["flagged"] is True
        assert results[1]["categories"].get("threat") is True

    @pytest.mark.asyncio
    async def test_empty_string_input_is_valid_and_unflagged(self):
        resp = await _post_moderations(_local_app(), {"input": ""})
        assert resp.status_code == 200
        assert resp.json()["results"][0]["flagged"] is False


class TestModerationInputValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"input": 7},
            {"input": []},
            {"input": ["ok", 3]},
            ["not", "a", "dict"],
        ],
    )
    async def test_invalid_input_returns_400(self, payload):
        resp = await _post_moderations(_local_app(), payload)
        assert resp.status_code == 400
        error = resp.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "input" in error["message"]

    @pytest.mark.asyncio
    async def test_non_json_body_returns_400(self):
        app = _local_app()
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/moderations",
                content=b"not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"


class TestUpstreamModerations:
    @pytest.mark.asyncio
    async def test_upstream_mode_forwards_verbatim(self):
        seen: dict = {}

        async def _handler(request: httpx.Request):
            seen["path"] = request.url.path
            seen["body"] = request.content
            seen["auth"] = request.headers.get("Authorization", "")
            return httpx.Response(
                200,
                json={"id": "modr-upstream", "results": [{"flagged": False}]},
            )

        app = _local_app(
            moderations="upstream",
            _transport=httpx.MockTransport(_handler),
        )
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/moderations",
                json={"input": "check this", "model": "omni-moderation-latest"},
                headers={"Authorization": "Bearer sk-upstream"},
            )
        assert resp.status_code == 200
        assert resp.json()["id"] == "modr-upstream"
        assert seen["path"] == "/v1/moderations"
        assert b"omni-moderation-latest" in seen["body"]
        assert seen["auth"] == "Bearer sk-upstream"

    @pytest.mark.asyncio
    async def test_upstream_error_status_is_propagated(self):
        async def _handler(request: httpx.Request):
            return httpx.Response(429, json={"error": {"message": "rate limit"}})

        app = _local_app(
            moderations="upstream",
            _transport=httpx.MockTransport(_handler),
        )
        resp = await _post_moderations(app, {"input": "x"})
        assert resp.status_code == 429
        assert resp.json()["error"]["message"] == "rate limit"


class TestModerationsWiring:
    def test_invalid_mode_rejected_at_build_time(self):
        with pytest.raises(ValueError, match="moderations"):
            create_proxy_app(moderations="bogus")

    @pytest.mark.asyncio
    async def test_api_key_middleware_guards_moderations(self):
        app = _local_app(api_keys=["secret-key"])
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            denied = await client.post("/v1/moderations", json={"input": "hi"})
            allowed = await client.post(
                "/v1/moderations",
                json={"input": "hi"},
                headers={"X-API-Key": "secret-key"},
            )
        assert denied.status_code == 401
        assert allowed.status_code == 200

    @pytest.mark.asyncio
    async def test_detectors_are_reused_across_requests(self):
        app = _local_app()
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            first = await client.post("/v1/moderations", json={"input": "hello"})
            second = await client.post(
                "/v1/moderations",
                json={"input": "reach me on 555-123-4567 ok"},
            )
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["results"][0]["categories"].get("phone") is True
