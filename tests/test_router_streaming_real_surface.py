# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - streaming router real-surface tests
"""Real WebSocket tests for the streaming router edge branches."""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

pytest.importorskip("fastapi", reason="server extras not installed")

from starlette.testclient import TestClient

import director_ai.routers.streaming as streaming_mod
from director_ai.core.config import DirectorConfig
from director_ai.server import create_app


class _StreamingAgent:
    """Agent stub that emits one token for streaming-oversight sessions."""

    async def stream(
        self,
        prompt: str,
        tenant_id: str = "",
    ) -> AsyncIterator[tuple[str, float]]:
        """Yield one high-coherence token for the requested prompt."""
        del tenant_id
        yield f"{prompt}-token", 0.95


class _EventFactory:
    """Factory that pre-cancels only the first router session event."""

    def __init__(self) -> None:
        """Initialise the created event buffer."""
        self.events: list[threading.Event] = []
        self._original_event = threading.Event

    def __call__(self) -> threading.Event:
        """Return a new event, pre-set only for the first session."""
        event = self._original_event()
        if not self.events:
            event.set()
        self.events.append(event)
        return event


def test_streaming_oversight_pre_cancel_suppresses_token_and_keeps_socket_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-cancelled oversight session should return without sending a token."""
    app = create_app(config=DirectorConfig(use_nli=False, llm_provider="mock"))
    event_factory = _EventFactory()
    router_threading = cast(Any, streaming_mod).threading

    with TestClient(app) as client, client.websocket_connect("/v1/stream") as ws:
        app.state._state["agent"] = _StreamingAgent()
        monkeypatch.setattr(router_threading, "Event", event_factory)
        ws.send_json(
            {
                "prompt": "suppressed",
                "session_id": "pre-cancelled",
                "streaming_oversight": True,
            },
        )
        time.sleep(0.05)
        ws.send_json(
            {
                "prompt": "visible",
                "session_id": "visible-session",
                "streaming_oversight": True,
            },
        )
        token = ws.receive_json()
        complete = ws.receive_json()

    assert len(event_factory.events) >= 2
    assert token == {
        "session_id": "visible-session",
        "type": "token",
        "token": "visible-token",
        "coherence": 0.95,
    }
    assert complete["session_id"] == "visible-session"
    assert complete["type"] == "complete"
    assert complete["tokens_delivered"] == 1
