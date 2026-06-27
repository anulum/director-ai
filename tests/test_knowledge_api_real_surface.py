# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Knowledge API real-surface tests
"""Real ASGI coverage for the tenant-scoped knowledge API router."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import cast

import httpx
import pytest

pytest.importorskip("fastapi", reason="fastapi required for knowledge API tests")

from fastapi import FastAPI, Request
from httpx import ASGITransport

from director_ai.core.config import DirectorConfig
from director_ai.core.kb_write_security import canonical_kb_payload, sign_kb_payload
from director_ai.core.retrieval.doc_registry import DocRegistry
from director_ai.core.retrieval.vector_store import (
    InMemoryBackend,
    VectorGroundTruthStore,
)
from director_ai.knowledge_api import create_knowledge_router

_TENANT_ID = "tenant.alpha"
_API_KEY = "tenant-alpha-key"
_SIGNATURE_KEY_ID = "main"
_SIGNATURE_SECRET = "tenant-alpha-hmac-secret"


class _KnowledgeScorer:
    """Minimal scorer state carrying the production vector store."""

    def __init__(self, ground_truth_store: VectorGroundTruthStore) -> None:
        self.ground_truth_store = ground_truth_store


def _knowledge_app() -> FastAPI:
    """Return a FastAPI app with production knowledge state mounted."""
    app = FastAPI()
    config = DirectorConfig(
        knowledge_write_require_auth=True,
        knowledge_write_require_tenant_binding=True,
        knowledge_write_require_signature=True,
        knowledge_write_hmac_keys=json.dumps({_SIGNATURE_KEY_ID: _SIGNATURE_SECRET}),
    )
    app.state.config = config
    app.state._state["config"] = config
    app.state._state["doc_registry"] = DocRegistry()
    app.state._state["scorer"] = _KnowledgeScorer(
        VectorGroundTruthStore(backend=InMemoryBackend()),
    )

    @app.middleware("http")
    async def _bind_tenant(
        request: Request,
        call_next: Callable[[Request], Awaitable[object]],
    ) -> object:
        """Bind request tenant and KB write credentials as server middleware does."""
        tenant_id = request.headers.get("X-Tenant-ID", "")
        api_key = request.headers.get("X-API-Key", "")
        request.state.tenant_id = tenant_id
        request.state.kb_write_key_ok = api_key == _API_KEY
        request.state.kb_tenant_binding_ok = api_key == _API_KEY and (
            tenant_id == _TENANT_ID
        )
        return await call_next(request)

    app.include_router(create_knowledge_router(), prefix="/v1/knowledge")
    return app


def _auth_headers() -> dict[str, str]:
    """Return tenant-bound API headers for KB write requests."""
    return {"X-Tenant-ID": _TENANT_ID, "X-API-Key": _API_KEY}


def _signed_body(
    *,
    kind: str,
    doc_id: str,
    source: str,
    text: str,
) -> dict[str, object]:
    """Return an API body signed with the production KB HMAC helper."""
    canonical = canonical_kb_payload(
        kind=kind,
        tenant_id=_TENANT_ID,
        doc_id=doc_id,
        source=source,
        text=text,
    )
    return {
        "doc_id": doc_id,
        "source": source,
        "text": text,
        "chunk_size": 128,
        "overlap": 16,
        "signature_key_id": _SIGNATURE_KEY_ID,
        "signature": sign_kb_payload(canonical, _SIGNATURE_SECRET),
    }


@pytest.mark.asyncio
async def test_knowledge_router_signed_tenant_document_lifecycle() -> None:
    """The mounted router should enforce auth and round-trip real KB state."""
    app = _knowledge_app()
    transport = ASGITransport(app=app)
    source = "policy.md"
    original_text = (
        "Capital buffer policy requires board approval before exposure changes. "
        "The tenant alpha evidence pack records the same capital buffer rule. "
    ) * 3
    updated_text = (
        "Capital buffer policy now requires board and risk committee approval. "
        "The tenant alpha evidence pack records the updated approval route. "
    ) * 3

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied_response = await client.post(
            "/v1/knowledge/ingest",
            json=_signed_body(
                kind="ingest",
                doc_id="policy-1",
                source=source,
                text=original_text,
            ),
            headers={"X-Tenant-ID": _TENANT_ID},
        )
        ingest_response = await client.post(
            "/v1/knowledge/ingest",
            json=_signed_body(
                kind="ingest",
                doc_id="policy-1",
                source=source,
                text=original_text,
            ),
            headers=_auth_headers(),
        )
        list_response = await client.get(
            "/v1/knowledge/documents",
            headers={"X-Tenant-ID": _TENANT_ID},
        )
        search_response = await client.get(
            "/v1/knowledge/search",
            params={"query": "capital buffer approval", "top_k": "1"},
            headers={"X-Tenant-ID": _TENANT_ID},
        )
        update_response = await client.put(
            "/v1/knowledge/documents/policy-1",
            json=_signed_body(
                kind="update",
                doc_id="policy-1",
                source=source,
                text=updated_text,
            ),
            headers=_auth_headers(),
        )
        delete_response = await client.delete(
            "/v1/knowledge/documents/policy-1",
            headers=_auth_headers(),
        )
        missing_response = await client.get(
            "/v1/knowledge/documents/policy-1",
            headers={"X-Tenant-ID": _TENANT_ID},
        )

    assert denied_response.status_code == 403
    assert denied_response.json()["detail"] == (
        "Knowledge-base writes require authentication"
    )

    assert ingest_response.status_code == 201, ingest_response.text
    ingest_payload = cast(dict[str, object], ingest_response.json())
    assert ingest_payload["doc_id"] == "policy-1"
    assert ingest_payload["tenant_id"] == _TENANT_ID
    assert cast(int, ingest_payload["chunk_count"]) >= 2

    assert list_response.status_code == 200, list_response.text
    list_payload = cast(dict[str, object], list_response.json())
    assert list_payload["count"] == 1
    documents = cast(list[dict[str, object]], list_payload["documents"])
    assert documents[0]["source"] == source
    assert documents[0]["content_hash"]

    assert search_response.status_code == 200, search_response.text
    search_payload = cast(dict[str, object], search_response.json())
    results = cast(list[dict[str, object]], search_payload["results"])
    assert len(results) == 1
    assert "Capital buffer policy" in cast(str, results[0]["text"])
    assert cast(dict[str, object], results[0]["metadata"])["tenant_id"] == _TENANT_ID

    assert update_response.status_code == 200, update_response.text
    update_payload = cast(dict[str, object], update_response.json())
    assert update_payload["unchanged"] is False
    assert cast(int, update_payload["chunk_count"]) >= 2

    assert delete_response.status_code == 200, delete_response.text
    delete_payload = cast(dict[str, object], delete_response.json())
    assert delete_payload["deleted"] == "policy-1"
    assert cast(int, delete_payload["chunks_removed"]) >= 2
    assert missing_response.status_code == 404
