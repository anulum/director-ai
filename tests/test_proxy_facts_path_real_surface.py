# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - proxy facts path real-surface tests
"""Real proxy-app coverage for facts-path root enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import httpx
import pytest
from httpx import ASGITransport

from director_ai.proxy import create_proxy_app


def _write_facts(path: Path, body: str = "sky: blue\n") -> None:
    """Write a small key-value facts file."""
    path.write_text(body, encoding="utf-8")


def _upstream_transport(content: str) -> httpx.MockTransport:
    """Return an OpenAI-compatible upstream protocol fixture."""

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-facts-root-real",
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


def _chat_request() -> dict[str, object]:
    """Return a minimal OpenAI-compatible chat completion request."""
    return {
        "model": "local-test-model",
        "messages": [
            {
                "role": "user",
                "content": "What colour is the sky?",
            },
        ],
    }


@pytest.mark.asyncio
async def test_proxy_loads_allowed_symlinked_facts_over_real_http_boundary(
    tmp_path: Path,
) -> None:
    """Allowed facts paths should feed the production proxy request path."""
    root = tmp_path / "facts-root"
    root.mkdir()
    real_facts = root / "facts.txt"
    _write_facts(real_facts)
    alias = root / "alias.txt"
    alias.symlink_to(real_facts)

    app = create_proxy_app(
        facts_path=str(alias),
        facts_root=str(root),
        upstream_url="http://upstream.local",
        on_fail="warn",
        use_nli=False,
        allow_http_upstream=True,
        _transport=_upstream_transport("The sky is blue."),
    )
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://proxy.local",
    ) as client:
        response = await client.post("/v1/chat/completions", json=_chat_request())

    assert response.status_code == 200
    payload = cast(dict[str, object], response.json())
    assert payload["id"] == "chatcmpl-facts-root-real"
    assert response.headers["x-director-approved"] in {"true", "false"}
    assert float(response.headers["x-director-score"]) >= 0.0


def test_proxy_rejects_traversal_facts_path_before_serving(tmp_path: Path) -> None:
    """Traversal paths should fail through the public app factory."""
    root = tmp_path / "facts-root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    _write_facts(outside, "admin: password\n")
    traversal = root / ".." / "outside.txt"

    with pytest.raises(ValueError, match="outside facts_root"):
        create_proxy_app(
            facts_path=str(traversal),
            facts_root=str(root),
            upstream_url="https://upstream.local",
            use_nli=False,
        )


def test_proxy_rejects_symlink_escape_before_serving(tmp_path: Path) -> None:
    """Symlinks escaping the root should fail through the public app factory."""
    root = tmp_path / "facts-root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    _write_facts(outside, "leak: yes\n")
    alias = root / "alias.txt"
    alias.symlink_to(outside)

    with pytest.raises(ValueError, match="outside facts_root"):
        create_proxy_app(
            facts_path=str(alias),
            facts_root=str(root),
            upstream_url="https://upstream.local",
            use_nli=False,
        )
