# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    *,
    tenant_state: str = "",
    tenant_header: str = "",
    registry=None,
    scorer=None,
):
    """Build a minimal fake Request for unit-testing standalone helpers."""
    state = SimpleNamespace(tenant_id=tenant_state, _state={})
    if registry is not None:
        state._state["doc_registry"] = registry
    if scorer is not None:
        state._state["scorer"] = scorer
    app = SimpleNamespace(state=state)
    headers: dict[str, str] = {}
    if tenant_header:
        headers["X-Tenant-ID"] = tenant_header
    return SimpleNamespace(state=state, app=app, headers=headers)


def _fake_record(doc_id="doc1", source="test", tenant_id="t1", chunk_ids=None):
    return SimpleNamespace(
        doc_id=doc_id,
        source=source,
        tenant_id=tenant_id,
        chunk_count=len(chunk_ids or []),
        chunk_ids=list(chunk_ids or []),
        created_at=0.0,
        updated_at=0.0,
    )


def _make_vector_store(facts=None):
    from director_ai.core.retrieval.vector_store import (
        InMemoryBackend,
        VectorGroundTruthStore,
    )

    store = VectorGroundTruthStore(backend=InMemoryBackend())
    if facts:
        store.facts.update(facts)
    return store


def _make_scorer(store):
    scorer = MagicMock()
    scorer.ground_truth_store = store
    return scorer


def _make_app(*, registry=None, scorer=None):
    """Build a FastAPI app with the knowledge router attached.

    State is set via individual attribute assignment so that
    request.app.state._state.get(...) works correctly in endpoints.
    FastAPI's State.__setattr__ places keys into its internal _state dict,
    so app.state.foo = x means app.state._state['foo'] == x.
    """
    from fastapi import FastAPI

    from director_ai.knowledge_api import create_knowledge_router

    app = FastAPI()
    app.include_router(create_knowledge_router(), prefix="/v1/knowledge")
    if registry is not None:
        app.state.doc_registry = registry
    if scorer is not None:
        app.state.scorer = scorer
    return app


def _full_app():
    reg = MagicMock()
    store = _make_vector_store()
    scorer = _make_scorer(store)
    app = _make_app(registry=reg, scorer=scorer)
    return app, reg, store


# ---------------------------------------------------------------------------
# _get_tenant
# ---------------------------------------------------------------------------


class TestGetTenant:
    def test_from_state(self):
        from director_ai.knowledge_api import _get_tenant

        req = _make_request(tenant_state="abc")
        assert _get_tenant(req) == "abc"

    def test_from_header_when_state_empty(self):
        from director_ai.knowledge_api import _get_tenant

        req = _make_request(tenant_state="", tenant_header="hdr_tenant")
        assert _get_tenant(req) == "hdr_tenant"

    def test_empty_when_neither(self):
        from director_ai.knowledge_api import _get_tenant

        req = _make_request()
        assert _get_tenant(req) == ""


# ---------------------------------------------------------------------------
# _get_registry
# ---------------------------------------------------------------------------


class TestGetRegistry:
    def test_returns_registry_when_present(self):
        from director_ai.knowledge_api import _get_registry

        reg = MagicMock()
        req = _make_request(registry=reg)
        assert _get_registry(req) is reg

    def test_raises_503_when_absent(self):
        from fastapi import HTTPException

        from director_ai.knowledge_api import _get_registry

        req = _make_request()
        with pytest.raises(HTTPException) as exc_info:
            _get_registry(req)
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# _get_store
# ---------------------------------------------------------------------------


class TestGetStore:
    def test_raises_503_when_scorer_absent(self):
        from fastapi import HTTPException

        from director_ai.knowledge_api import _get_store

        req = _make_request()
        with pytest.raises(HTTPException) as exc_info:
            _get_store(req)
        assert exc_info.value.status_code == 503

    def test_raises_503_when_store_not_vector(self):
        from fastapi import HTTPException

        from director_ai.knowledge_api import _get_store

        scorer = MagicMock()
        scorer.ground_truth_store = MagicMock()
        req = _make_request(scorer=scorer)
        with pytest.raises(HTTPException) as exc_info:
            _get_store(req)
        assert exc_info.value.status_code == 503

    def test_raises_503_when_scorer_has_none_store(self):
        from fastapi import HTTPException

        from director_ai.knowledge_api import _get_store

        scorer = MagicMock()
        scorer.ground_truth_store = None
        req = _make_request(scorer=scorer)
        with pytest.raises(HTTPException) as exc_info:
            _get_store(req)
        assert exc_info.value.status_code == 503

    def test_returns_store_when_valid(self):
        from director_ai.knowledge_api import _get_store

        store = _make_vector_store()
        scorer = _make_scorer(store)
        req = _make_request(scorer=scorer)
        assert _get_store(req) is store


# ---------------------------------------------------------------------------
# _chunk_and_store
# ---------------------------------------------------------------------------


class TestChunkAndStore:
    def test_single_chunk(self):
        from director_ai.knowledge_api import _chunk_and_store

        store = _make_vector_store()
        ids = _chunk_and_store("hello world", "doc1", "t1", store, 512, 64)
        assert ids == ["doc1:chunk:0"]

    def test_chunk_id_format(self):
        from director_ai.knowledge_api import _chunk_and_store

        store = _make_vector_store()
        ids = _chunk_and_store("a", "mydoc", "t1", store, 512, 64)
        assert ids == ["mydoc:chunk:0"]

    def test_multi_chunk_ids_sequential(self):
        from director_ai.knowledge_api import _chunk_and_store

        store = _make_vector_store()
        long_text = "word " * 300
        ids = _chunk_and_store(long_text, "d", "t1", store, 64, 0)
        assert len(ids) > 1
        for i, cid in enumerate(ids):
            assert cid == f"d:chunk:{i}"

    def test_backend_add_called(self):
        from director_ai.knowledge_api import _chunk_and_store

        store = _make_vector_store()
        backend_mock = MagicMock()
        store.backend = backend_mock
        _chunk_and_store("text", "doc", "t1", store, 512, 64)
        backend_mock.add.assert_called_once()


# ---------------------------------------------------------------------------
# _delete_chunks
# ---------------------------------------------------------------------------


class TestDeleteChunks:
    def test_removes_from_facts_and_counts_backend_deletes(self):
        from director_ai.knowledge_api import _delete_chunks

        store = _make_vector_store(facts={"c0": "t0", "c1": "t1"})
        backend_mock = MagicMock()
        store.backend = backend_mock
        record = _fake_record(chunk_ids=["c0", "c1"])
        removed = _delete_chunks(record, store)
        assert removed == 2
        assert "c0" not in store.facts
        assert "c1" not in store.facts

    def test_tolerates_attribute_error(self):
        from director_ai.knowledge_api import _delete_chunks

        store = _make_vector_store()
        store.backend = MagicMock()
        store.backend.delete.side_effect = AttributeError
        record = _fake_record(chunk_ids=["x:chunk:0"])
        removed = _delete_chunks(record, store)
        assert removed == 0

    def test_tolerates_type_error(self):
        from director_ai.knowledge_api import _delete_chunks

        store = _make_vector_store()
        store.backend = MagicMock()
        store.backend.delete.side_effect = TypeError
        record = _fake_record(chunk_ids=["x:chunk:0"])
        removed = _delete_chunks(record, store)
        assert removed == 0

    def test_empty_chunk_ids(self):
        from director_ai.knowledge_api import _delete_chunks

        store = _make_vector_store()
        removed = _delete_chunks(_fake_record(chunk_ids=[]), store)
        assert removed == 0

    def test_facts_popped_even_when_backend_raises(self):
        from director_ai.knowledge_api import _delete_chunks

        store = _make_vector_store(facts={"z": "val"})
        store.backend = MagicMock()
        store.backend.delete.side_effect = AttributeError
        record = _fake_record(chunk_ids=["z"])
        _delete_chunks(record, store)
        assert "z" not in store.facts


# ---------------------------------------------------------------------------
# create_knowledge_router guard
# ---------------------------------------------------------------------------


class TestCreateRouterNoFastAPI:
    def test_raises_import_error_when_fastapi_absent(self):
        import director_ai.knowledge_api as mod

        orig = mod._FASTAPI_AVAILABLE
        mod._FASTAPI_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="FastAPI required"):
                mod.create_knowledge_router()
        finally:
            mod._FASTAPI_AVAILABLE = orig


# ---------------------------------------------------------------------------
# /upload endpoint
# ---------------------------------------------------------------------------


class TestUploadEndpoint:
    def test_413_from_content_length_header(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/knowledge/upload",
            files={"file": ("test.txt", b"hi", "text/plain")},
            headers={"content-length": str(60 * 1024 * 1024)},
        )
        assert resp.status_code == 413

    def test_503_no_registry(self):
        from fastapi.testclient import TestClient

        app = _make_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/knowledge/upload",
            files={"file": ("f.txt", b"hi", "text/plain")},
        )
        assert resp.status_code == 503

    def test_415_unsupported_extension(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/knowledge/upload",
            files={"file": ("bad.exe", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 415

    def test_415_no_extension_in_filename(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        client = TestClient(app)
        resp = client.post(
            "/v1/knowledge/upload",
            files={"file": ("no_ext", b"data", "text/plain")},
        )
        assert resp.status_code == 415

    def test_413_from_body_size(self):
        # Call the upload endpoint directly with a fake request + UploadFile that returns
        # oversized content, bypassing the content-length header check.
        import asyncio
        from unittest.mock import AsyncMock

        from fastapi import HTTPException

        store = _make_vector_store()
        scorer = _make_scorer(store)
        reg = MagicMock()

        req = _make_request(registry=reg, scorer=scorer)
        file_mock = MagicMock()
        file_mock.filename = "test.txt"
        big_content = b"x" * (50 * 1024 * 1024 + 1)
        file_mock.read = AsyncMock(return_value=big_content)

        # Import the router function via the router app
        from director_ai.knowledge_api import create_knowledge_router

        # Access the endpoint via the router's routes
        router = create_knowledge_router()
        upload_fn = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/upload":
                upload_fn = route.endpoint
                break

        assert upload_fn is not None, "upload endpoint not found in router"

        async def _run():
            return await upload_fn(req, file_mock)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 413

    def test_422_parse_raises_value_error(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        client = TestClient(app)
        with patch(
            "director_ai.core.retrieval.doc_parser.parse", side_effect=ValueError("bad")
        ):
            resp = client.post(
                "/v1/knowledge/upload",
                files={"file": ("test.txt", b"data", "text/plain")},
            )
        assert resp.status_code == 422

    def test_422_parse_raises_import_error(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        client = TestClient(app)
        with patch(
            "director_ai.core.retrieval.doc_parser.parse",
            side_effect=ImportError("missing dep"),
        ):
            resp = client.post(
                "/v1/knowledge/upload",
                files={"file": ("test.txt", b"data", "text/plain")},
            )
        assert resp.status_code == 422

    def test_422_empty_parsed_text(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        client = TestClient(app)
        with patch("director_ai.core.retrieval.doc_parser.parse", return_value="   "):
            resp = client.post(
                "/v1/knowledge/upload",
                files={"file": ("test.txt", b"   ", "text/plain")},
            )
        assert resp.status_code == 422

    def test_201_success(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        record = _fake_record(
            doc_id="abc", source="good.txt", chunk_ids=["abc:chunk:0"]
        )
        reg.register.return_value = record
        client = TestClient(app)
        with patch(
            "director_ai.core.retrieval.doc_parser.parse", return_value="real text"
        ):
            resp = client.post(
                "/v1/knowledge/upload",
                files={"file": ("good.txt", b"real text", "text/plain")},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert data["doc_id"] == "abc"
        assert data["chunk_count"] == 1


# ---------------------------------------------------------------------------
# /ingest endpoint
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    def test_201_success(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        record = _fake_record(doc_id="d1", source="api", chunk_ids=["d1:chunk:0"])
        reg.exists.return_value = False
        reg.register.return_value = record
        client = TestClient(app)
        resp = client.post(
            "/v1/knowledge/ingest", json={"text": "meaningful content here"}
        )
        assert resp.status_code == 201
        assert resp.json()["doc_id"] == "d1"

    def test_409_duplicate_doc_id(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        reg.exists.return_value = True
        client = TestClient(app)
        resp = client.post(
            "/v1/knowledge/ingest",
            json={"text": "hello", "doc_id": "existing_id"},
        )
        assert resp.status_code == 409

    def test_custom_doc_id_used(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        record = _fake_record(
            doc_id="custom", source="text", chunk_ids=["custom:chunk:0"]
        )
        reg.exists.return_value = False
        reg.register.return_value = record
        client = TestClient(app)
        resp = client.post(
            "/v1/knowledge/ingest",
            json={"text": "some content here", "doc_id": "custom"},
        )
        assert resp.status_code == 201
        assert resp.json()["doc_id"] == "custom"

    def test_503_no_scorer(self):
        from fastapi.testclient import TestClient

        app = _make_app(registry=MagicMock())
        client = TestClient(app)
        resp = client.post("/v1/knowledge/ingest", json={"text": "hello"})
        assert resp.status_code == 503

    def test_503_no_registry(self):
        from fastapi.testclient import TestClient

        app = _make_app()
        client = TestClient(app)
        resp = client.post("/v1/knowledge/ingest", json={"text": "hello"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------


class TestListDocuments:
    def test_empty_list(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        reg.list_for_tenant.return_value = []
        client = TestClient(app)
        resp = client.get("/v1/knowledge/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["documents"] == []

    def test_with_documents(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        rec = _fake_record(doc_id="d1", source="s1", chunk_ids=["d1:chunk:0"])
        reg.list_for_tenant.return_value = [rec]
        client = TestClient(app)
        resp = client.get("/v1/knowledge/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["documents"][0]["doc_id"] == "d1"

    def test_503_no_registry(self):
        from fastapi.testclient import TestClient

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/v1/knowledge/documents")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /documents/{doc_id}
# ---------------------------------------------------------------------------


class TestGetDocument:
    def test_404_not_found(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        reg.get.return_value = None
        client = TestClient(app)
        resp = client.get("/v1/knowledge/documents/missing")
        assert resp.status_code == 404

    def test_200_found(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        rec = _fake_record(doc_id="d2", tenant_id="", chunk_ids=["d2:chunk:0"])
        reg.get.return_value = rec
        client = TestClient(app)
        resp = client.get("/v1/knowledge/documents/d2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "d2"
        assert "chunk_count" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "tenant_id" in data


# ---------------------------------------------------------------------------
# DELETE /documents/{doc_id}
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    def test_404_not_found(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        reg.get.return_value = None
        client = TestClient(app)
        resp = client.delete("/v1/knowledge/documents/ghost")
        assert resp.status_code == 404

    def test_200_deleted(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        backend_mock = MagicMock()
        store.backend = backend_mock
        rec = _fake_record(doc_id="d3", chunk_ids=["d3:c0", "d3:c1"])
        reg.get.return_value = rec
        store.facts["d3:c0"] = "t0"
        store.facts["d3:c1"] = "t1"
        client = TestClient(app)
        resp = client.delete("/v1/knowledge/documents/d3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] == "d3"
        assert data["chunks_removed"] == 2


# ---------------------------------------------------------------------------
# PUT /documents/{doc_id}
# ---------------------------------------------------------------------------


class TestUpdateDocument:
    def test_404_not_found(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        reg.get.return_value = None
        client = TestClient(app)
        resp = client.put(
            "/v1/knowledge/documents/nope",
            json={"text": "new content"},
        )
        assert resp.status_code == 404

    def test_200_updated(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        backend_mock = MagicMock()
        store.backend = backend_mock
        rec = _fake_record(doc_id="d4", chunk_ids=["d4:chunk:0"])
        reg.get.return_value = rec
        store.facts["d4:chunk:0"] = "old text"
        client = TestClient(app)
        resp = client.put(
            "/v1/knowledge/documents/d4",
            json={"text": "updated content for document d4"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == "d4"
        assert "chunk_count" in data


# ---------------------------------------------------------------------------
# GET /search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def test_empty_results_from_backend(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        client = TestClient(app)
        resp = client.get("/v1/knowledge/search?query=nothing")
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_returns_results(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        store.backend = MagicMock()
        store.backend.query.return_value = [
            {"text": "chunk text", "distance": 0.1, "metadata": {}}
        ]
        client = TestClient(app)
        resp = client.get("/v1/knowledge/search?query=hello&top_k=3")
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1
        assert resp.json()["results"][0]["text"] == "chunk text"

    def test_text_truncated_to_500(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        store.backend = MagicMock()
        store.backend.query.return_value = [
            {"text": "x" * 1000, "distance": 0.0, "metadata": {}}
        ]
        client = TestClient(app)
        resp = client.get("/v1/knowledge/search?query=q")
        assert len(resp.json()["results"][0]["text"]) == 500

    def test_falls_back_on_type_error(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        store.backend = MagicMock()

        def _query_side_effect(query, **kwargs):
            if "tenant_id" in kwargs:
                raise TypeError("no tenant_id param")
            return [{"text": "fallback", "distance": 0.2, "metadata": {}}]

        store.backend.query.side_effect = _query_side_effect
        client = TestClient(app)
        resp = client.get("/v1/knowledge/search?query=test")
        assert resp.status_code == 200
        assert resp.json()["results"][0]["text"] == "fallback"

    def test_503_no_scorer(self):
        from fastapi.testclient import TestClient

        app = _make_app()
        client = TestClient(app)
        resp = client.get("/v1/knowledge/search?query=x")
        assert resp.status_code == 503

    def test_result_missing_fields_uses_defaults(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        store.backend = MagicMock()
        store.backend.query.return_value = [{}]
        client = TestClient(app)
        resp = client.get("/v1/knowledge/search?query=y")
        assert resp.status_code == 200
        result = resp.json()["results"][0]
        assert result["text"] == ""
        assert result["distance"] is None
        assert result["metadata"] == {}


# ---------------------------------------------------------------------------
# POST /tune-embeddings
# ---------------------------------------------------------------------------


class TestTuneEmbeddingsEndpoint:
    def test_422_fewer_than_2_docs(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        reg.list_for_tenant.return_value = [_fake_record()]
        client = TestClient(app)
        resp = client.post("/v1/knowledge/tune-embeddings")
        assert resp.status_code == 422
        assert "2 documents" in resp.json()["detail"]

    def test_422_zero_docs(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        reg.list_for_tenant.return_value = []
        client = TestClient(app)
        resp = client.post("/v1/knowledge/tune-embeddings")
        assert resp.status_code == 422

    def test_422_not_enough_docs_with_2plus_chunks(self):
        from fastapi.testclient import TestClient

        app, reg, store = _full_app()
        rec1 = _fake_record(doc_id="d1", chunk_ids=["d1:c0"])
        rec2 = _fake_record(doc_id="d2", chunk_ids=["d2:c0"])
        reg.list_for_tenant.return_value = [rec1, rec2]
        store.facts = {"d1:c0": "only chunk", "d2:c0": "also only chunk"}
        client = TestClient(app)
        resp = client.post("/v1/knowledge/tune-embeddings")
        assert resp.status_code == 422
        assert "2+ chunks" in resp.json()["detail"]

    def test_200_tune_success(self):
        from fastapi.testclient import TestClient

        from director_ai.core.retrieval.embedding_tuner import TuneResult

        app, reg, store = _full_app()
        rec1 = _fake_record(doc_id="d1", chunk_ids=["d1:c0", "d1:c1"])
        rec2 = _fake_record(doc_id="d2", chunk_ids=["d2:c0", "d2:c1"])
        reg.list_for_tenant.return_value = [rec1, rec2]
        store.facts = {
            "d1:c0": "first chunk doc one",
            "d1:c1": "second chunk doc one",
            "d2:c0": "first chunk doc two",
            "d2:c1": "second chunk doc two",
        }
        fake_result = TuneResult(
            model_path="models/tuned",
            train_samples=4,
            epochs=3,
            loss_start=1.0,
            loss_end=0.5,
        )
        client = TestClient(app)
        with patch(
            "director_ai.core.retrieval.embedding_tuner.tune_embeddings",
            return_value=fake_result,
        ):
            resp = client.post("/v1/knowledge/tune-embeddings")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_path"] == "models/tuned"
        assert data["train_samples"] == 4
        assert data["epochs"] == 3
        assert "Re-ingest" in data["message"]

    def test_docs_with_empty_chunk_text_excluded(self):
        from fastapi.testclient import TestClient

        from director_ai.core.retrieval.embedding_tuner import TuneResult

        app, reg, store = _full_app()
        rec1 = _fake_record(doc_id="d1", chunk_ids=["d1:c0", "d1:c1"])
        rec2 = _fake_record(doc_id="d2", chunk_ids=["d2:c0"])  # 1 chunk → excluded
        rec3 = _fake_record(doc_id="d3", chunk_ids=["d3:c0", "d3:c1"])
        reg.list_for_tenant.return_value = [rec1, rec2, rec3]
        store.facts = {
            "d1:c0": "text a",
            "d1:c1": "text b",
            "d3:c0": "text c",
            "d3:c1": "text d",
        }
        fake_result = TuneResult(
            model_path="m", train_samples=2, epochs=3, loss_start=0.9, loss_end=0.4
        )
        client = TestClient(app)
        with patch(
            "director_ai.core.retrieval.embedding_tuner.tune_embeddings",
            return_value=fake_result,
        ):
            resp = client.post("/v1/knowledge/tune-embeddings")
        assert resp.status_code == 200

    def test_503_no_registry(self):
        from fastapi.testclient import TestClient

        app = _make_app()
        client = TestClient(app)
        resp = client.post("/v1/knowledge/tune-embeddings")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# IngestRequest model validation
# ---------------------------------------------------------------------------


class TestIngestRequestModel:
    def test_defaults(self):
        from director_ai.knowledge_api import IngestRequest

        req = IngestRequest(text="hello")
        assert req.source == "text"
        assert req.doc_id is None
        assert req.chunk_size == 512
        assert req.overlap == 64

    def test_custom_fields(self):
        from director_ai.knowledge_api import IngestRequest

        req = IngestRequest(
            text="x", source="s", doc_id="d", chunk_size=128, overlap=32
        )
        assert req.source == "s"
        assert req.doc_id == "d"
        assert req.chunk_size == 128
        assert req.overlap == 32
