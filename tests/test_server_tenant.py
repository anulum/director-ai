# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Server Tenant Routing Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest

from director_ai.core.config import DirectorConfig

try:
    from fastapi.testclient import TestClient

    from director_ai.server import create_app

    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _SERVER_AVAILABLE, reason="fastapi not installed")


def test_list_tenants_empty():
    cfg = DirectorConfig(tenant_routing=True, llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/tenants")
    assert r.status_code == 200
    assert r.json()["tenants"] == []


def test_add_fact_creates_tenant():
    cfg = DirectorConfig(tenant_routing=True, llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post(
            "/v1/tenants/acme/facts",
            json={"key": "hq", "value": "Acme HQ is in Springfield."},
        )
        assert r.status_code == 200
        assert r.json()["tenant_id"] == "acme"

        r2 = client.get("/v1/tenants")
        tenants = r2.json()["tenants"]
    assert len(tenants) == 1
    assert tenants[0]["id"] == "acme"
    assert tenants[0]["fact_count"] == 1


def test_review_with_tenant_header():
    cfg = DirectorConfig(tenant_routing=True, llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post(
            "/v1/tenants/acme/facts",
            json={
                "key": "sky",
                "value": "The sky is blue due to Rayleigh scattering.",
            },
        )
        r = client.post(
            "/v1/review",
            json={"prompt": "sky color?", "response": "The sky is blue."},
            headers={"X-Tenant-ID": "acme"},
        )
    assert r.status_code == 200
    assert "coherence" in r.json()


def test_tenant_isolation():
    cfg = DirectorConfig(tenant_routing=True, llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        client.post(
            "/v1/tenants/acme/facts",
            json={
                "key": "product",
                "value": "Acme sells rocket-powered roller skates.",
            },
        )
        client.post(
            "/v1/tenants/globex/facts",
            json={
                "key": "product",
                "value": "Globex manufactures doomsday devices.",
            },
        )
        tenants = client.get("/v1/tenants").json()["tenants"]
    ids = {t["id"] for t in tenants}
    assert ids == {"acme", "globex"}


def test_tenants_404_when_disabled():
    cfg = DirectorConfig(tenant_routing=False, llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/tenants")
    assert r.status_code == 404
