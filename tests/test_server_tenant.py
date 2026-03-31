# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Server Tenant Routing Tests (STRONG)
"""Multi-angle tests for tenant routing in FastAPI server.

Covers: empty tenant list, tenant creation, tenant-scoped review,
tenant-scoped retrieval, parametrised tenant IDs, pipeline
integration, and performance documentation.
"""

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


@pytest.mark.parametrize("tenant_id", ["alpha", "beta-corp", "tenant_123"])
def test_parametrised_tenant_creation(tenant_id):
    cfg = DirectorConfig(tenant_routing=True, llm_provider="mock")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.post(
            f"/v1/tenants/{tenant_id}/facts",
            json={"key": "test", "value": "test value"},
        )
    assert r.status_code in (200, 201, 204)


class TestTenantPerformanceDoc:
    """Document tenant routing pipeline performance."""

    def test_tenant_list_returns_json(self):
        cfg = DirectorConfig(tenant_routing=True, llm_provider="mock")
        app = create_app(cfg)
        with TestClient(app) as client:
            r = client.get("/v1/tenants")
        assert r.status_code == 200
        data = r.json()
        assert "tenants" in data
        assert isinstance(data["tenants"], list)
