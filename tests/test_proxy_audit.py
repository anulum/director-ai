# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for proxy audit log integration."""

from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from director_ai.compliance.audit_log import AuditLog
from director_ai.proxy import create_proxy_app


def _upstream_transport(content: str):
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
async def test_proxy_audit_logging(tmp_path):
    db_path = str(tmp_path / "audit.db")
    mock_transport = _upstream_transport("The sky is blue due to Rayleigh scattering.")
    app = create_proxy_app(
        threshold=0.3,
        upstream_url="http://fake-upstream",
        on_fail="warn",
        use_nli=False,
        allow_http_upstream=True,
        audit_db=db_path,
        _transport=mock_transport,
    )

    proxy_transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=proxy_transport, base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

    assert resp.status_code == 200

    log = AuditLog(db_path)
    entries = log.query()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.model == "gpt-4o-mini"
    assert entry.provider == "proxy"
    assert entry.approved is True
    assert entry.score > 0
    log.close()


@pytest.mark.asyncio
async def test_proxy_no_audit_when_disabled(tmp_path):
    mock_transport = _upstream_transport("Some response.")
    app = create_proxy_app(
        threshold=0.3,
        upstream_url="http://fake-upstream",
        on_fail="warn",
        use_nli=False,
        allow_http_upstream=True,
        _transport=mock_transport,
    )

    proxy_transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=proxy_transport, base_url="http://test"
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "test"}],
            },
        )

    assert resp.status_code == 200
