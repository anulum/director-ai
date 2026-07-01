# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for knowledge-write security routes."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

pytest.importorskip("fastapi", reason="fastapi required for knowledge write tests")

from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.core.kb_write_security import canonical_kb_payload, sign_kb_payload
from director_ai.server import create_app
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

_TENANT_ID = "acme"
_OTHER_TENANT_ID = "other"
_API_KEY = "writer-key"
_SIGNATURE_KEY_ID = "main"
_SIGNATURE_SECRET = "writer-hmac-secret-at-least-32-chars"


def _server_config() -> DirectorConfig:
    """Build a production server config with tenant-bound signed KB writes."""
    return DirectorConfig(
        tenant_routing=True,
        llm_provider="mock",
        use_nli=False,
        knowledge_write_require_auth=True,
        knowledge_write_require_tenant_binding=True,
        knowledge_write_require_signature=True,
        knowledge_write_hmac_keys=json.dumps({_SIGNATURE_KEY_ID: _SIGNATURE_SECRET}),
        api_keys=[_API_KEY],
        api_key_tenant_map=json.dumps({_API_KEY: _TENANT_ID}),
    )


def _headers() -> dict[str, str]:
    """Return the tenant-bound writer credential header."""
    return {"X-API-Key": _API_KEY}


def _signed_write_body(
    *,
    kind: str,
    tenant_id: str,
    key: str,
    value: str,
    signature: str | None = None,
) -> dict[str, str]:
    """Return a signed tenant write request body for server routes."""
    canonical = canonical_kb_payload(
        kind=kind,
        tenant_id=tenant_id,
        key=key,
        value=value,
    )
    return {
        "key": key,
        "value": value,
        "signature": (
            sign_kb_payload(canonical, _SIGNATURE_SECRET)
            if signature is None
            else signature
        ),
        "signature_key_id": _SIGNATURE_KEY_ID,
    }


def _json(response: Any) -> dict[str, Any]:
    """Return a typed JSON object from a TestClient response."""
    return cast(dict[str, Any], response.json())


def test_knowledge_write_unit_guard_declares_this_real_surface_companion() -> None:
    """The unit guard manifest must point at this production-surface companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_knowledge_write_security.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_knowledge_write_security_real_surface.py" in reason


def test_tenant_fact_write_security_runs_through_server_middleware() -> None:
    """Tenant fact writes require bound credentials and valid HMAC signatures."""
    app = create_app(_server_config())
    valid_body = _signed_write_body(
        kind="tenant_fact",
        tenant_id=_TENANT_ID,
        key="refund-policy",
        value="Refund approvals require signed operator evidence.",
    )

    with TestClient(app) as client:
        wrong_tenant = client.post(
            f"/v1/tenants/{_OTHER_TENANT_ID}/facts",
            json=_signed_write_body(
                kind="tenant_fact",
                tenant_id=_OTHER_TENANT_ID,
                key="refund-policy",
                value="Wrong tenant write should not land.",
            ),
            headers=_headers(),
        )
        missing_signature = client.post(
            f"/v1/tenants/{_TENANT_ID}/facts",
            json={"key": "refund-policy", "value": valid_body["value"]},
            headers=_headers(),
        )
        invalid_signature = client.post(
            f"/v1/tenants/{_TENANT_ID}/facts",
            json={**valid_body, "signature": "bad-signature"},
            headers=_headers(),
        )
        accepted = client.post(
            f"/v1/tenants/{_TENANT_ID}/facts",
            json=valid_body,
            headers=_headers(),
        )
        tenants = client.get("/v1/tenants", headers=_headers())

    assert wrong_tenant.status_code == 403
    assert _json(wrong_tenant)["detail"] == "API key not authorized for this tenant"
    assert missing_signature.status_code == 403
    assert _json(missing_signature)["detail"] == (
        "Knowledge-base write signature required"
    )
    assert invalid_signature.status_code == 403
    assert (
        _json(invalid_signature)["detail"] == "Invalid knowledge-base write signature"
    )
    assert accepted.status_code == 200
    accepted_payload = _json(accepted)
    assert accepted_payload["status"] == "ok"
    assert accepted_payload["tenant_id"] == _TENANT_ID
    assert accepted_payload["key"] == "refund-policy"
    assert tenants.status_code == 200
    assert _json(tenants)["tenants"] == [{"id": _TENANT_ID, "fact_count": 1}]


def test_tenant_vector_fact_write_updates_real_store_count() -> None:
    """Signed vector writes should reach the tenant vector store through HTTP."""
    app = create_app(_server_config())
    valid_body = _signed_write_body(
        kind="tenant_vector_fact",
        tenant_id=_TENANT_ID,
        key="rollback-policy",
        value="Rollback approval evidence must be retained for audit.",
    )

    with TestClient(app) as client:
        invalid_signature = client.post(
            f"/v1/tenants/{_TENANT_ID}/vector-facts",
            json={**valid_body, "signature": "bad-signature"},
            headers=_headers(),
        )
        accepted = client.post(
            f"/v1/tenants/{_TENANT_ID}/vector-facts",
            json=valid_body,
            headers=_headers(),
        )

    assert invalid_signature.status_code == 403
    assert _json(invalid_signature)["detail"] == (
        "Invalid knowledge-base write signature"
    )
    assert accepted.status_code == 200
    assert _json(accepted) == {
        "status": "ok",
        "tenant_id": _TENANT_ID,
        "key": "rollback-policy",
        "backend_type": "memory",
        "count": 1,
    }
