# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SSE Streaming Route Tests

"""Multi-angle tests for the ``POST /v1/stream/sse`` route.

Mirrors the WebSocket multiplexing suite: the same two session shapes
(whole-answer result, token-level pre-egress oversight) delivered as
``text/event-stream``, plus the pre-stream HTTP validation surface the
WebSocket does not have.
"""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="server extras not installed")

from starlette.testclient import TestClient

import director_ai.routers.streaming_sse as sse_mod
from director_ai.core.config import DirectorConfig
from director_ai.server import create_app


def _app():
    return create_app(config=DirectorConfig(use_nli=False, llm_provider="mock"))


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Split an SSE body into (event, payload) pairs."""
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        event = next(ln[7:] for ln in lines if ln.startswith("event: "))
        data = json.loads(next(ln[6:] for ln in lines if ln.startswith("data: ")))
        events.append((event, data))
    return events


class _FakeStreamAgent:
    """Agent stub yielding scripted (token, coherence) pairs."""

    def __init__(self, tokens, fail_after=None):
        self._tokens = tokens
        self._fail_after = fail_after
        self.seen_tenants: list[str] = []

    async def stream(self, prompt, tenant_id=""):
        self.seen_tenants.append(tenant_id)
        for i, (tok, score) in enumerate(self._tokens):
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError("scripted stream failure")
            yield tok, score


class TestSSEResultSession:
    def test_result_event_with_mock_provider(self):
        with TestClient(_app()) as client:
            resp = client.post("/v1/stream/sse", json={"prompt": "Hello world"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(resp.text)
        assert len(events) == 1
        event, data = events[0]
        assert event == "result"
        assert "output" in data
        assert "halted" in data
        assert "evidence" in data

    def test_result_event_without_coherence_uses_nulls(self):
        fake = SimpleNamespace(
            aprocess=None,
        )

        async def _aprocess(prompt, tenant_id=""):
            return SimpleNamespace(
                output="bare",
                coherence=None,
                halted=False,
                fallback_used=False,
                halt_evidence=None,
            )

        fake.aprocess = _aprocess
        with TestClient(_app()) as client:
            client.app.state._state["agent"] = fake
            resp = client.post("/v1/stream/sse", json={"prompt": "hi"})
        event, data = _parse_sse(resp.text)[0]
        assert event == "result"
        assert data["output"] == "bare"
        assert data["coherence"] is None
        assert data["warning"] is False

    def test_result_error_becomes_error_event(self):
        async def _aprocess(prompt, tenant_id=""):
            raise RuntimeError("scripted process failure")

        with TestClient(_app()) as client:
            client.app.state._state["agent"] = SimpleNamespace(aprocess=_aprocess)
            resp = client.post("/v1/stream/sse", json={"prompt": "hi"})
        assert resp.status_code == 200
        assert _parse_sse(resp.text) == [("error", {"error": "streaming failed"})]


class TestSSEOversightSession:
    def test_tokens_then_complete(self):
        agent = _FakeStreamAgent([("alpha", 0.9), ("beta", 0.85), ("gamma", 0.8)])
        with TestClient(_app()) as client:
            client.app.state._state["agent"] = agent
            resp = client.post(
                "/v1/stream/sse",
                json={"prompt": "hi", "streaming_oversight": True},
                headers={"X-Tenant-ID": "tenant-7"},
            )
        events = _parse_sse(resp.text)
        assert [e for e, _ in events] == ["token", "token", "token", "complete"]
        assert [d["token"] for _, d in events[:3]] == ["alpha", "beta", "gamma"]
        terminal = events[-1][1]
        assert terminal["halted"] is False
        assert terminal["tokens_delivered"] == 3
        assert agent.seen_tenants == ["tenant-7"]

    def test_halt_suppresses_offending_token(self):
        agent = _FakeStreamAgent(
            [("good", 0.9), ("ok", 0.8), ("bad", 0.2), ("never", 0.1)],
        )
        with TestClient(_app()) as client:
            client.app.state._state["agent"] = agent
            resp = client.post(
                "/v1/stream/sse",
                json={"prompt": "hi", "streaming_oversight": True},
            )
        events = _parse_sse(resp.text)
        assert [e for e, _ in events] == ["token", "token", "halt"]
        delivered = [d["token"] for e, d in events if e == "token"]
        assert delivered == ["good", "ok"]
        assert "bad" not in resp.text and "never" not in resp.text
        halt = events[-1][1]
        assert halt["halted"] is True
        assert halt["tokens_delivered"] == 2
        assert halt["reason"] == "coherence_halt"

    def test_stream_error_becomes_error_event_after_tokens(self):
        agent = _FakeStreamAgent([("alpha", 0.9), ("beta", 0.9)], fail_after=1)
        with TestClient(_app()) as client:
            client.app.state._state["agent"] = agent
            resp = client.post(
                "/v1/stream/sse",
                json={"prompt": "hi", "streaming_oversight": True},
            )
        events = _parse_sse(resp.text)
        assert [e for e, _ in events] == ["token", "error"]
        assert events[-1][1] == {"error": "streaming failed"}


class TestSSEValidation:
    @pytest.mark.parametrize(
        ("payload", "expected_status"),
        [
            ({"prompt": ""}, 400),
            ({"prompt": "   "}, 400),
            ({"prompt": 5}, 400),
            ({}, 400),
        ],
    )
    def test_bad_prompt_rejected_before_stream(self, payload, expected_status):
        with TestClient(_app()) as client:
            resp = client.post("/v1/stream/sse", json=payload)
        assert resp.status_code == expected_status

    def test_non_object_body_rejected(self):
        with TestClient(_app()) as client:
            resp = client.post("/v1/stream/sse", json=["not", "a", "dict"])
        assert resp.status_code == 400

    def test_invalid_json_body_rejected(self):
        with TestClient(_app()) as client:
            resp = client.post(
                "/v1/stream/sse",
                content=b"not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 400

    def test_oversized_prompt_rejected_with_413(self):
        with TestClient(_app()) as client:
            resp = client.post(
                "/v1/stream/sse",
                json={"prompt": "x" * (sse_mod._SSE_MAX_PROMPT_LENGTH + 1)},
            )
        assert resp.status_code == 413

    def test_blocking_sanitizer_rejects_before_stream(self):
        blocked = SimpleNamespace(blocked=True, reason="test-block")
        with TestClient(_app()) as client:
            client.app.state._state["sanitizer"] = SimpleNamespace(
                check=lambda p: blocked,
            )
            resp = client.post("/v1/stream/sse", json={"prompt": "hi"})
        assert resp.status_code == 400
        assert "injection rejected" in resp.text

    def test_non_blocking_sanitizer_lets_session_through(self):
        clear = SimpleNamespace(blocked=False, reason="")
        with TestClient(_app()) as client:
            client.app.state._state["sanitizer"] = SimpleNamespace(
                check=lambda p: clear,
            )
            resp = client.post("/v1/stream/sse", json={"prompt": "hi"})
        assert resp.status_code == 200
        assert _parse_sse(resp.text)[0][0] == "result"

    def test_absent_sanitizer_lets_session_through(self):
        with TestClient(_app()) as client:
            client.app.state._state["sanitizer"] = None
            resp = client.post("/v1/stream/sse", json={"prompt": "hi"})
        assert resp.status_code == 200
        assert _parse_sse(resp.text)[0][0] == "result"


class TestSSECapacityAndReadiness:
    def test_missing_agent_returns_503(self):
        with TestClient(_app()) as client:
            client.app.state._state["agent"] = None
            resp = client.post("/v1/stream/sse", json={"prompt": "hi"})
        assert resp.status_code == 503
        assert "not ready" in resp.text

    def test_capacity_cap_returns_503(self, monkeypatch):
        monkeypatch.setattr(sse_mod, "_SSE_MAX_CONCURRENT_STREAMS", 0)
        with TestClient(_app()) as client:
            resp = client.post("/v1/stream/sse", json={"prompt": "hi"})
        assert resp.status_code == 503
        assert "capacity" in resp.text

    def test_slot_released_after_stream_completes(self):
        with TestClient(_app()) as client:
            for _ in range(3):
                resp = client.post("/v1/stream/sse", json={"prompt": "hi"})
                assert resp.status_code == 200
