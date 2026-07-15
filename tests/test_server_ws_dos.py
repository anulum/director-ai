# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WebSocket DoS-control tests
"""Tests for the WebSocket denial-of-service controls.

Drives small limits (via monkeypatched module constants) to exercise the global
and per-IP connection caps, the per-connection message rate limit, the
per-connection character budget, the idle timeout, and the session lifetime cap,
plus the backpressure metric.
"""

import pytest

pytest.importorskip("fastapi", reason="server extras not installed")

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import director_ai.routers.streaming as streaming_mod
from director_ai.core.config import DirectorConfig
from director_ai.core.metrics import metrics
from director_ai.server import create_app


def _app():
    cfg = DirectorConfig(use_nli=False, llm_provider="mock", tenant_routing=True)
    return create_app(config=cfg)


class TestConnectionCaps:
    def test_per_ip_cap_rejects_second_connection(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_WS_MAX_CONNECTIONS_PER_IP", 1)
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws,
        ):
            ws.send_json({"prompt": "hi", "session_id": "s1"})
            ws.receive_json()
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect("/v1/stream"),
            ):
                pass

    def test_global_cap_rejects_second_connection(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_WS_MAX_CONNECTIONS", 1)
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws,
        ):
            ws.send_json({"prompt": "hi", "session_id": "s1"})
            ws.receive_json()
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect("/v1/stream"),
            ):
                pass

    def test_slot_released_after_disconnect(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_WS_MAX_CONNECTIONS_PER_IP", 1)

        def _use_then_close(client):
            with client.websocket_connect("/v1/stream") as ws:
                ws.send_json({"prompt": "hi", "session_id": "s1"})
                ws.receive_json()

        with TestClient(_app()) as client:
            _use_then_close(client)
            # First connection closed -> slot freed -> a new one is admitted.
            with client.websocket_connect("/v1/stream") as ws2:
                ws2.send_json({"prompt": "hi again", "session_id": "s2"})
                assert ws2.receive_json()["session_id"] == "s2"

    def test_closing_one_of_multiple_connections_preserves_remaining_slot(self):
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws1,
        ):
            with client.websocket_connect("/v1/stream") as ws2:
                ws2.send_json({"prompt": "second connection", "session_id": "s2"})
                assert ws2.receive_json()["session_id"] == "s2"

            ws1.send_json({"prompt": "first still active", "session_id": "s1"})
            assert ws1.receive_json()["session_id"] == "s1"


class TestRateLimit:
    def test_message_rate_limit(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_WS_MAX_MSGS_PER_WINDOW", 2)
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws,
        ):
            seen_rate_error = False
            for n in range(6):
                ws.send_json({"prompt": f"msg {n}", "session_id": f"s{n}"})
                resp = ws.receive_json()
                if resp.get("error") == "message rate limit exceeded":
                    seen_rate_error = True
                    break
            assert seen_rate_error


class TestPromptLengthCap:
    def test_oversized_prompt_is_rejected_per_message(self, monkeypatch):
        # KIMI-E: the SSE path has capped prompts since its introduction;
        # a single oversized WebSocket message must be rejected the same
        # way instead of riding on the slower connection-level budget.
        monkeypatch.setattr(streaming_mod, "_WS_MAX_PROMPT_LENGTH", 100)
        monkeypatch.setattr(streaming_mod, "_WS_CONN_CHAR_BUDGET", 10_000)
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws,
        ):
            ws.send_json({"prompt": "x" * 101, "session_id": "s1"})
            resp = ws.receive_json()
            assert resp["error"] == "prompt exceeds 100 chars"
            # The connection survives; a bounded prompt still works.
            ws.send_json({"prompt": "short", "session_id": "s2"})
            resp2 = ws.receive_json()
            assert resp2.get("error") != "prompt exceeds 100 chars"


class TestCharBudget:
    def test_budget_closes_connection(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_WS_CONN_CHAR_BUDGET", 10)
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws,
        ):
            ws.send_json({"prompt": "x" * 50, "session_id": "s1"})
            with pytest.raises(WebSocketDisconnect):
                # The over-budget prompt triggers a server-side close.
                while True:
                    ws.receive_json()


class TestLifetimeAndIdle:
    def test_lifetime_cap_closes(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_WS_MAX_LIFETIME_S", 0.0)
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws,
            pytest.raises(WebSocketDisconnect),
        ):
            ws.receive_json()

    def test_idle_timeout_closes(self, monkeypatch):
        monkeypatch.setattr(streaming_mod, "_WS_IDLE_TIMEOUT_S", 0.05)
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws,
            pytest.raises(WebSocketDisconnect),
        ):
            ws.receive_json()


class TestMetrics:
    def test_rejection_metric_emitted(self, monkeypatch):
        metrics.reset()
        monkeypatch.setattr(streaming_mod, "_WS_MAX_CONNECTIONS_PER_IP", 1)
        with (
            TestClient(_app()) as client,
            client.websocket_connect("/v1/stream") as ws,
        ):
            ws.send_json({"prompt": "hi", "session_id": "s1"})
            ws.receive_json()
            with (
                pytest.raises(WebSocketDisconnect),
                client.websocket_connect("/v1/stream"),
            ):
                pass
        snapshot = metrics.get_metrics()
        counter = snapshot["counters"].get("ws_rejections_total", {})
        assert counter.get("multi_labels", {}).get('reason="per_ip_cap"') == 1.0
