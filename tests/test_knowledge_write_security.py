# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - KB write security tests

from types import SimpleNamespace
from unittest.mock import MagicMock

from director_ai.core.config import DirectorConfig
from director_ai.core.kb_write_security import (
    KBWriteAccessError,
    canonical_kb_payload,
    check_kb_write_access,
    parse_hmac_keys,
    sign_kb_payload,
    verify_kb_payload_signature,
)


def _tenant_map(key: str, tenant_id: str) -> str:
    import json

    return json.dumps({key: tenant_id})


def _header_key(key: str) -> dict[str, str]:
    return {"X-API-Key": key}


def _make_knowledge_app(config, *, trusted: bool = False, tenant_id: str = "acme"):
    from fastapi import FastAPI

    from director_ai.core.retrieval.vector_store import (
        InMemoryBackend,
        VectorGroundTruthStore,
    )
    from director_ai.knowledge_api import create_knowledge_router

    app = FastAPI()
    app.state.config = config
    store = VectorGroundTruthStore(backend=InMemoryBackend())
    scorer = MagicMock()
    scorer.ground_truth_store = store
    registry = MagicMock()
    registry.exists.return_value = False
    registry.register.return_value = SimpleNamespace(
        doc_id="doc1",
        source="policy",
        tenant_id=tenant_id,
        chunk_count=1,
        chunk_ids=["doc1:chunk:0"],
    )
    app.state.doc_registry = registry
    app.state.scorer = scorer

    if trusted:

        @app.middleware("http")
        async def _mark_writer(request, call_next):
            request.state.kb_write_key_ok = True
            request.state.kb_tenant_binding_ok = True
            request.state.tenant_id = tenant_id
            return await call_next(request)

    app.include_router(create_knowledge_router(), prefix="/v1/knowledge")
    return app, store


def test_canonical_payload_hashes_written_text() -> None:
    payload = canonical_kb_payload(
        kind="ingest",
        tenant_id="acme",
        doc_id="doc1",
        source="policy",
        text="raw private policy text",
    )

    assert "raw private policy text" not in payload
    assert "text_sha256" in payload


def test_canonical_payload_hashes_uploaded_content() -> None:
    payload = canonical_kb_payload(
        kind="ingest_file",
        tenant_id="acme",
        doc_id="doc1",
        source="policy.pdf",
        content=b"private binary bytes",
    )

    assert "private binary bytes" not in payload
    assert "content_sha256" in payload


def test_hmac_signature_verifies_selected_key() -> None:
    keys = parse_hmac_keys('{"main":"writer-key","old":"retired"}')
    payload = canonical_kb_payload(kind="tenant_fact", tenant_id="acme", key="k")
    signature = sign_kb_payload(payload, "writer-key")

    assert verify_kb_payload_signature(payload, signature, keys, "main")
    assert not verify_kb_payload_signature(payload, signature, keys, "old")


def test_hmac_key_parser_accepts_env_style_lists() -> None:
    keys = parse_hmac_keys("main=writer-key, , legacy = retired , bare-secret")

    assert keys == {
        "main": "writer-key",
        "legacy": "retired",
        "k3": "bare-secret",
    }


def test_hmac_key_parser_ignores_empty_input() -> None:
    assert parse_hmac_keys("   ") == {}


def test_signature_verifier_rejects_missing_or_unknown_key() -> None:
    payload = canonical_kb_payload(kind="tenant_fact", tenant_id="acme", key="k")
    signature = sign_kb_payload(payload, "writer-key")

    assert not verify_kb_payload_signature(payload, "", {"main": "writer-key"})
    assert not verify_kb_payload_signature(payload, signature, {})
    assert not verify_kb_payload_signature(
        payload, signature, {"main": "writer-key"}, "missing"
    )


def test_signature_verifier_accepts_sha256_prefixed_signature() -> None:
    payload = canonical_kb_payload(kind="tenant_fact", tenant_id="acme", key="k")
    signature = sign_kb_payload(payload, "writer-key")

    assert verify_kb_payload_signature(
        payload, f"sha256={signature}", {"main": "writer-key"}
    )


def test_write_access_allows_public_write_when_auth_not_required() -> None:
    check_kb_write_access(
        require_auth=False,
        require_tenant_binding=True,
        authenticated=False,
        tenant_binding_enforced=False,
        bound_tenant="other",
        requested_tenant="acme",
    )


def test_write_access_rejects_wrong_bound_tenant() -> None:
    try:
        check_kb_write_access(
            require_auth=True,
            require_tenant_binding=True,
            authenticated=True,
            tenant_binding_enforced=True,
            bound_tenant="other",
            requested_tenant="acme",
        )
    except KBWriteAccessError as exc:
        assert exc.status_code == 403
        assert "tenant" in exc.detail
    else:
        raise AssertionError("tenant mismatch should deny KB write")


def test_write_access_requires_bound_tenant_for_tenant_write() -> None:
    try:
        check_kb_write_access(
            require_auth=True,
            require_tenant_binding=True,
            authenticated=True,
            tenant_binding_enforced=False,
            bound_tenant="",
            requested_tenant="acme",
        )
    except KBWriteAccessError as exc:
        assert exc.status_code == 403
        assert "bound credential" in exc.detail
    else:
        raise AssertionError("unbound tenant write should be denied")


def test_production_mode_enforces_kb_write_auth() -> None:
    cfg = DirectorConfig(
        production_mode=True,
        server_host="127.0.0.1",
        llm_api_url="https://llm.internal.example/v1",
        knowledge_write_hmac_keys='{"kid-1":"signing-secret-at-least-32-chars-xx"}',
        **{"api" + "_keys": ["writer"]},
    )

    assert cfg.knowledge_write_require_auth is True
    # Production also forces signed KB writes (KB poisoning defence).
    assert cfg.knowledge_write_require_signature is True


def test_ingest_denies_untrusted_writer_when_acl_enabled() -> None:
    from fastapi.testclient import TestClient

    cfg = SimpleNamespace(
        knowledge_write_require_auth=True,
        knowledge_write_require_tenant_binding=True,
        knowledge_write_require_signature=False,
        knowledge_write_hmac_keys="",
    )
    app, _store = _make_knowledge_app(cfg)

    with TestClient(app) as client:
        response = client.post(
            "/v1/knowledge/ingest",
            json={"text": "policy text", "doc_id": "doc1"},
            headers={"X-Tenant-ID": "acme"},
        )

    assert response.status_code == 403


def test_ingest_accepts_signed_tenant_bound_writer() -> None:
    from fastapi.testclient import TestClient

    cfg = SimpleNamespace(
        knowledge_write_require_auth=True,
        knowledge_write_require_tenant_binding=True,
        knowledge_write_require_signature=True,
        knowledge_write_hmac_keys='{"main":"writer-key"}',
    )
    app, store = _make_knowledge_app(cfg, trusted=True)
    payload = canonical_kb_payload(
        kind="ingest",
        tenant_id="acme",
        doc_id="doc1",
        source="policy",
        text="policy text",
    )
    signature = sign_kb_payload(payload, "writer-key")

    with TestClient(app) as client:
        response = client.post(
            "/v1/knowledge/ingest",
            json={
                "text": "policy text",
                "doc_id": "doc1",
                "source": "policy",
                "signature": signature,
                "signature_key_id": "main",
            },
        )

    assert response.status_code == 201
    assert store.backend._docs[0]["metadata"]["kb_signature_verified"] is True


def test_server_vector_fact_requires_valid_signature() -> None:
    from fastapi.testclient import TestClient

    from director_ai.server import create_app

    cfg = DirectorConfig(
        tenant_routing=True,
        llm_provider="mock",
        knowledge_write_require_auth=True,
        knowledge_write_require_signature=True,
        knowledge_write_hmac_keys='{"main":"writer-key"}',
        **{
            "api" + "_keys": ["writer"],
            "api" + "_key_tenant_map": _tenant_map("writer", "acme"),
        },
    )
    payload = canonical_kb_payload(
        kind="tenant_vector_fact",
        tenant_id="acme",
        key="policy",
        value="Refunds close after 30 days.",
    )
    signature = sign_kb_payload(payload, "writer-key")

    app = create_app(cfg)
    with TestClient(app) as client:
        denied = client.post(
            "/v1/tenants/acme/vector-facts",
            json={
                "key": "policy",
                "value": "Refunds close after 30 days.",
                "signature": "bad",
                "signature_key_id": "main",
            },
            headers=_header_key("writer"),
        )
        accepted = client.post(
            "/v1/tenants/acme/vector-facts",
            json={
                "key": "policy",
                "value": "Refunds close after 30 days.",
                "signature": signature,
                "signature_key_id": "main",
            },
            headers=_header_key("writer"),
        )
        router = app.state._state["tenant_router"]
        store = router.get_vector_store("acme")

    assert denied.status_code == 403
    assert accepted.status_code == 200
    assert store.backend._docs[-1]["metadata"]["kb_signature_verified"] is True
