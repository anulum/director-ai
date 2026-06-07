# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — knowledge API tests

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from director_ai.core.kb_write_security import canonical_kb_payload, sign_kb_payload
from director_ai.knowledge_api import (
    _cleanup_chunk_ids,
    _content_hash,
    _delete_chunks,
    _file_extension,
    _get_config,
    _get_registry,
    _get_store,
    _get_tenant,
    _require_write_access,
    _signature_metadata,
    _validate_chunking,
    _validate_content_length,
    _validate_optional_identifier,
    _validate_source,
    _validate_upload_type,
)


class _Record(SimpleNamespace):
    @property
    def chunk_count(self) -> int:
        return len(self.chunk_ids)


class _Registry:
    def __init__(self, records: list[_Record] | None = None) -> None:
        self.records = {record.doc_id: record for record in records or []}
        self.deleted: list[str] = []

    def exists(self, doc_id: str) -> bool:
        return doc_id in self.records

    def register(
        self,
        doc_id: str,
        source: str,
        tenant_id: str,
        chunk_ids: list[str],
        *,
        content_hash: str,
    ) -> _Record:
        record = _Record(
            doc_id=doc_id,
            source=source,
            tenant_id=tenant_id,
            chunk_ids=chunk_ids,
            created_at="created",
            updated_at="updated",
            content_hash=content_hash,
        )
        self.records[doc_id] = record
        return record

    def list_for_tenant(self, tenant_id: str) -> list[_Record]:
        return [
            record for record in self.records.values() if record.tenant_id == tenant_id
        ]

    def get(self, doc_id: str, tenant_id: str) -> _Record | None:
        record = self.records.get(doc_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record

    def update(
        self,
        doc_id: str,
        chunk_ids: list[str],
        *,
        source: str,
        content_hash: str,
    ) -> _Record:
        record = self.records[doc_id]
        record.chunk_ids = chunk_ids
        record.source = source
        record.content_hash = content_hash
        record.updated_at = "updated-again"
        return record

    def delete(self, doc_id: str) -> None:
        self.deleted.append(doc_id)
        self.records.pop(doc_id, None)


class _EndpointBackend:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, object]] = {}
        self.delete_result: object = 1
        self.fail_delete = False
        self.raise_type_error_on_tenant_query = False

    def add(self, doc_id: str, text: str, metadata: dict[str, object]) -> None:
        self.docs[doc_id] = {"text": text, "metadata": metadata}

    def delete(self, doc_ids: list[str]) -> object:
        if self.fail_delete:
            raise RuntimeError("delete failed")
        for doc_id in doc_ids:
            self.docs.pop(doc_id, None)
        return self.delete_result

    def query(self, text: str, n_results: int = 3, tenant_id: str = ""):
        if tenant_id and self.raise_type_error_on_tenant_query:
            raise TypeError("tenant_id unsupported")
        return [
            {
                "text": f"{text} result with a long body" * 40,
                "distance": 0.2,
                "metadata": {"tenant_id": tenant_id},
            }
        ][:n_results]

    def count(self) -> int:
        return len(self.docs)


def _request(
    *,
    config: object | None = None,
    registry: object | None = None,
    scorer: object | None = None,
    tenant_id: str = "",
    headers: dict[str, str] | None = None,
    kb_write_key_ok: bool = False,
    kb_tenant_binding_ok: bool = False,
) -> SimpleNamespace:
    state_data: dict[str, object] = {}
    if config is not None:
        state_data["config"] = config
    if registry is not None:
        state_data["doc_registry"] = registry
    if scorer is not None:
        state_data["scorer"] = scorer
    app_state = SimpleNamespace(_state=state_data)
    if config is not None:
        app_state.config = config
    return SimpleNamespace(
        headers=headers or {},
        state=SimpleNamespace(
            tenant_id=tenant_id,
            kb_write_key_ok=kb_write_key_ok,
            kb_tenant_binding_ok=kb_tenant_binding_ok,
        ),
        app=SimpleNamespace(state=app_state),
    )


def _make_app(
    *,
    registry: _Registry | None = None,
    backend: _EndpointBackend | None = None,
    tenant_id: str = "tenant-a",
):
    from fastapi import FastAPI

    from director_ai.core.retrieval.vector_store import VectorGroundTruthStore
    from director_ai.knowledge_api import create_knowledge_router

    backend = backend or _EndpointBackend()
    store = VectorGroundTruthStore(backend=backend)
    scorer = SimpleNamespace(ground_truth_store=store)
    app = FastAPI()
    app.state.config = SimpleNamespace(
        knowledge_write_require_auth=False,
        knowledge_write_require_tenant_binding=True,
        knowledge_write_require_signature=False,
        knowledge_write_hmac_keys="",
    )
    app.state.doc_registry = registry or _Registry()
    app.state.scorer = scorer

    @app.middleware("http")
    async def _tenant(request, call_next):
        request.state.tenant_id = tenant_id
        return await call_next(request)

    app.include_router(create_knowledge_router(), prefix="/v1/knowledge")
    return app, app.state.doc_registry, store, backend


def _assert_http(excinfo, status_code: int, detail: str) -> None:
    assert excinfo.value.status_code == status_code
    assert detail in excinfo.value.detail


def test_tenant_and_identifier_validation_paths() -> None:
    request = _request(headers={"X-Tenant-ID": "tenant-a"})
    assert _get_tenant(request) == "tenant-a"

    request_with_state = _request(tenant_id="tenant-state")
    assert _get_tenant(request_with_state) == "tenant-state"
    assert _validate_optional_identifier("doc_id", None) == ""
    assert _validate_optional_identifier("doc_id", "") == ""
    assert _validate_optional_identifier("doc_id", "doc_1:rev-2") == "doc_1:rev-2"

    with pytest.raises(HTTPException) as excinfo:
        _validate_optional_identifier("doc_id", "bad/id")
    _assert_http(excinfo, 400, "doc_id must be")


def test_source_chunking_content_length_and_upload_type_validation() -> None:
    assert _validate_source("  policy.md  ") == "policy.md"
    with pytest.raises(HTTPException) as excinfo:
        _validate_source(" ")
    _assert_http(excinfo, 400, "source must be non-empty")
    with pytest.raises(HTTPException) as excinfo:
        _validate_source("bad\nsource")
    _assert_http(excinfo, 400, "invalid control")

    _validate_chunking(64, 63)
    with pytest.raises(HTTPException) as excinfo:
        _validate_chunking(64, 64)
    _assert_http(excinfo, 400, "overlap")

    _validate_content_length(None)
    _validate_content_length("")
    _validate_content_length("10")
    for raw in ("not-int", "-1"):
        with pytest.raises(HTTPException) as excinfo:
            _validate_content_length(raw)
        _assert_http(excinfo, 400, "content-length")
    with pytest.raises(HTTPException) as excinfo:
        _validate_content_length(str(51 * 1024 * 1024))
    _assert_http(excinfo, 413, "File exceeds")

    assert _file_extension("README") == ""
    assert _file_extension("Policy.PDF") == "pdf"
    assert (
        _validate_upload_type("policy.pdf", "application/pdf; charset=binary") == "pdf"
    )
    assert _validate_upload_type("policy.md", "application/octet-stream") == "md"
    with pytest.raises(HTTPException) as excinfo:
        _validate_upload_type("policy.exe", "application/octet-stream")
    _assert_http(excinfo, 415, "not supported")
    with pytest.raises(HTTPException) as excinfo:
        _validate_upload_type("policy.pdf", "text/plain")
    _assert_http(excinfo, 415, "does not match")


def test_registry_store_and_config_accessors() -> None:
    from director_ai.core.retrieval.vector_store import (
        InMemoryBackend,
        VectorGroundTruthStore,
    )

    registry = object()
    store = VectorGroundTruthStore(backend=InMemoryBackend())
    scorer = SimpleNamespace(ground_truth_store=store)
    config = SimpleNamespace(knowledge_write_require_auth=False)
    request = _request(config=config, registry=registry, scorer=scorer)

    assert _get_registry(request) is registry
    assert _get_store(request) is store
    assert _get_config(request) is config

    with pytest.raises(HTTPException) as excinfo:
        _get_registry(_request())
    _assert_http(excinfo, 503, "Document registry")

    with pytest.raises(HTTPException) as excinfo:
        _get_store(_request())
    _assert_http(excinfo, 503, "Scorer")

    bad_scorer = SimpleNamespace(ground_truth_store=object())
    with pytest.raises(HTTPException) as excinfo:
        _get_store(_request(scorer=bad_scorer))
    _assert_http(excinfo, 503, "Vector store")

    request_without_attr_config = _request(config=config)
    delattr(request_without_attr_config.app.state, "config")
    assert _get_config(request_without_attr_config) is config


def test_write_access_and_signature_metadata_paths() -> None:
    public_cfg = SimpleNamespace(
        knowledge_write_require_auth=False,
        knowledge_write_require_tenant_binding=True,
        knowledge_write_require_signature=False,
        knowledge_write_hmac_keys="",
    )
    _require_write_access(_request(config=public_cfg), "tenant-a")

    locked_cfg = SimpleNamespace(
        knowledge_write_require_auth=True,
        knowledge_write_require_tenant_binding=True,
        knowledge_write_require_signature=False,
        knowledge_write_hmac_keys="",
    )
    with pytest.raises(HTTPException) as excinfo:
        _require_write_access(_request(config=locked_cfg), "tenant-a")
    _assert_http(excinfo, 403, "Knowledge-base")

    signed_cfg = SimpleNamespace(
        knowledge_write_require_auth=False,
        knowledge_write_require_tenant_binding=True,
        knowledge_write_require_signature=True,
        knowledge_write_hmac_keys='{"main":"writer-key"}',
    )
    request = _request(config=signed_cfg)
    payload = canonical_kb_payload(kind="ingest", tenant_id="tenant-a", doc_id="doc")
    signature = sign_kb_payload(payload, "writer-key")
    meta = _signature_metadata(request, payload, signature, "main")
    assert meta == {
        "kb_signature": signature,
        "kb_signature_key_id": "main",
        "kb_signature_verified": True,
    }

    with pytest.raises(HTTPException) as excinfo:
        _signature_metadata(request, payload)
    _assert_http(excinfo, 403, "signature required")

    with pytest.raises(HTTPException) as excinfo:
        _signature_metadata(request, payload, "bad", "main")
    _assert_http(excinfo, 403, "Invalid")

    optional_cfg = SimpleNamespace(
        knowledge_write_require_signature=False,
        knowledge_write_hmac_keys="",
    )
    assert _signature_metadata(_request(config=optional_cfg), payload) == {}


def test_chunk_store_content_hash_cleanup_and_delete_paths(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    class _Backend:
        def __init__(self) -> None:
            self.added: list[str] = []
            self.deleted: list[str] = []
            self.fail_on_add = ""
            self.delete_count: object = 1

        def add(self, *, doc_id: str, text: str, metadata: dict[str, object]) -> None:
            if doc_id == self.fail_on_add:
                raise RuntimeError("add failed")
            self.added.append(doc_id)

        def delete(self, doc_ids: list[str]) -> object:
            self.deleted.extend(doc_ids)
            return self.delete_count

    store = SimpleNamespace(backend=_Backend(), facts={})
    monkeypatch.setattr(
        "director_ai.core.retrieval.doc_chunker.split",
        lambda text, config: ["chunk one", "chunk two"],
    )

    chunk_ids = knowledge_api._chunk_and_store(
        "text",
        "doc1",
        "tenant-a",
        store,
        64,
        8,
        {"sig": True},
    )

    assert chunk_ids == ["doc1:chunk:0", "doc1:chunk:1"]
    assert store.backend.added == chunk_ids
    assert store.facts == {
        "doc1:chunk:0": "chunk one",
        "doc1:chunk:1": "chunk two",
    }
    assert _content_hash("same") == _content_hash("same")

    _cleanup_chunk_ids(["doc1:chunk:0"], store)
    assert store.backend.deleted[-1] == "doc1:chunk:0"
    assert "doc1:chunk:0" not in store.facts

    store.facts["doc1:chunk:1"] = "chunk two"
    removed = _delete_chunks(SimpleNamespace(chunk_ids=["doc1:chunk:1"]), store)
    assert removed == 1
    assert "doc1:chunk:1" not in store.facts


def test_chunk_store_rolls_back_staged_chunks_on_add_failure(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    class _Backend:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def add(self, *, doc_id: str, text: str, metadata: dict[str, object]) -> None:
            if doc_id.endswith(":1"):
                raise RuntimeError("boom")

        def delete(self, doc_ids: list[str]) -> int:
            self.deleted.extend(doc_ids)
            return 1

    store = SimpleNamespace(backend=_Backend(), facts={})
    monkeypatch.setattr(
        "director_ai.core.retrieval.doc_chunker.split",
        lambda text, config: ["first", "second"],
    )

    with pytest.raises(RuntimeError, match="boom"):
        knowledge_api._chunk_and_store("text", "doc", "tenant", store, 64, 8)

    assert store.backend.deleted == ["doc:chunk:0"]
    assert store.facts == {}


def test_delete_chunks_reports_backend_contract_violations() -> None:
    class _Backend:
        def __init__(self, result):
            self.result = result

        def delete(self, doc_ids: list[str]):
            if self.result == "raise":
                raise RuntimeError("offline")
            return self.result

    store = SimpleNamespace(backend=_Backend(None), facts={"c1": "text"})
    assert _delete_chunks(SimpleNamespace(chunk_ids=["c1"]), store) == 1

    store = SimpleNamespace(backend=_Backend(0), facts={"c1": "text"})
    with pytest.raises(RuntimeError, match="reported 0 deletions"):
        _delete_chunks(SimpleNamespace(chunk_ids=["c1"]), store)

    store = SimpleNamespace(backend=_Backend("raise"), facts={"c1": "text"})
    with pytest.raises(RuntimeError, match="Unable to delete chunk"):
        _delete_chunks(SimpleNamespace(chunk_ids=["c1"]), store)


def test_create_router_requires_fastapi(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    monkeypatch.setattr(knowledge_api, "_FASTAPI_AVAILABLE", False)
    with pytest.raises(ImportError, match="FastAPI required"):
        knowledge_api.create_knowledge_router()


def test_module_import_without_fastapi_marks_router_unavailable() -> None:
    import director_ai.knowledge_api as knowledge_api

    spec = importlib.util.spec_from_file_location(
        "director_ai._knowledge_api_no_fastapi",
        knowledge_api.__file__,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"fastapi": None, "pydantic": None}):
        spec.loader.exec_module(module)

    assert module._FASTAPI_AVAILABLE is False
    with pytest.raises(ImportError, match="FastAPI required"):
        module.create_knowledge_router()


def test_document_list_get_delete_and_search_endpoints() -> None:
    record = _Record(
        doc_id="doc1",
        source="policy",
        tenant_id="tenant-a",
        chunk_ids=["doc1:chunk:0"],
        created_at="created",
        updated_at="updated",
        content_hash="hash",
    )
    registry = _Registry([record])
    backend = _EndpointBackend()
    backend.docs["doc1:chunk:0"] = {"text": "old", "metadata": {}}
    app, registry, store, backend = _make_app(registry=registry, backend=backend)
    store.facts["doc1:chunk:0"] = "old"

    with TestClient(app) as client:
        listing = client.get("/v1/knowledge/documents")
        found = client.get("/v1/knowledge/documents/doc1")
        missing = client.get("/v1/knowledge/documents/missing")
        bad_doc_id = client.get("/v1/knowledge/documents/bad/id")
        bad_query = client.get("/v1/knowledge/search", params={"query": " "})
        bad_top_k = client.get(
            "/v1/knowledge/search",
            params={"query": "policy", "top_k": 0},
        )
        search = client.get(
            "/v1/knowledge/search",
            params={"query": "policy", "top_k": 1},
        )
        deleted = client.delete("/v1/knowledge/documents/doc1")

    assert listing.json()["documents"][0]["content_hash"] == "hash"
    assert found.json()["doc_id"] == "doc1"
    assert missing.status_code == 404
    assert bad_doc_id.status_code == 404
    assert bad_query.status_code == 400
    assert bad_top_k.status_code == 400
    assert search.status_code == 200
    assert len(search.json()["results"][0]["text"]) == 500
    assert deleted.json() == {"deleted": "doc1", "chunks_removed": 1}
    assert registry.deleted == ["doc1"]
    assert "doc1:chunk:0" not in store.facts


def test_delete_missing_and_backend_failure_paths() -> None:
    missing_app, _registry, _store, _backend = _make_app()
    with TestClient(missing_app) as client:
        missing = client.delete("/v1/knowledge/documents/doc1")
    assert missing.status_code == 404

    record = _Record(
        doc_id="doc1",
        source="policy",
        tenant_id="tenant-a",
        chunk_ids=["doc1:chunk:0"],
        created_at="created",
        updated_at="updated",
        content_hash="hash",
    )
    backend = _EndpointBackend()
    backend.fail_delete = True
    app, _registry, store, _backend = _make_app(
        registry=_Registry([record]),
        backend=backend,
    )
    store.facts["doc1:chunk:0"] = "old"
    with TestClient(app) as client:
        failed = client.delete("/v1/knowledge/documents/doc1")

    assert failed.status_code == 503
    assert "delete document chunks" in failed.json()["detail"]


def test_search_falls_back_for_backends_without_tenant_query() -> None:
    backend = _EndpointBackend()
    backend.raise_type_error_on_tenant_query = True
    app, _registry, _store, _backend = _make_app(backend=backend)

    with TestClient(app) as client:
        response = client.get(
            "/v1/knowledge/search",
            params={"query": "policy", "top_k": 1},
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["metadata"]["tenant_id"] == ""


def test_upload_endpoint_success_and_parse_errors(monkeypatch) -> None:
    import director_ai.core.retrieval.doc_parser as doc_parser

    monkeypatch.setattr(doc_parser, "parse", lambda content, filename: "parsed text")
    monkeypatch.setattr(
        "director_ai.core.retrieval.doc_chunker.split",
        lambda text, config: ["parsed chunk"],
    )
    app, registry, store, _backend = _make_app()

    with TestClient(app) as client:
        uploaded = client.post(
            "/v1/knowledge/upload",
            files={"file": ("policy.txt", b"raw bytes", "text/plain")},
        )

    assert uploaded.status_code == 201
    body = uploaded.json()
    assert body["source"] == "policy.txt"
    assert body["chunk_count"] == 1
    assert body["tenant_id"] == "tenant-a"
    doc_id = body["doc_id"]
    assert doc_id in registry.records
    assert f"{doc_id}:chunk:0" in store.facts

    monkeypatch.setattr(
        doc_parser,
        "parse",
        lambda content, filename: (_ for _ in ()).throw(ValueError("bad document")),
    )
    with TestClient(app) as client:
        bad_parse = client.post(
            "/v1/knowledge/upload",
            files={"file": ("policy.txt", b"raw bytes", "text/plain")},
        )
    assert bad_parse.status_code == 422
    assert "bad document" in bad_parse.json()["detail"]

    monkeypatch.setattr(doc_parser, "parse", lambda content, filename: "   ")
    with TestClient(app) as client:
        empty_parse = client.post(
            "/v1/knowledge/upload",
            files={"file": ("policy.txt", b"raw bytes", "text/plain")},
        )
    assert empty_parse.status_code == 422
    assert "no text" in empty_parse.json()["detail"]

    with TestClient(app) as client:
        bad_type = client.post(
            "/v1/knowledge/upload",
            files={"file": ("policy.exe", b"raw bytes", "application/octet-stream")},
        )
    assert bad_type.status_code == 415


def test_upload_endpoint_rejects_body_that_exceeds_size_after_read(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    app, _registry, _store, _backend = _make_app()
    monkeypatch.setattr(knowledge_api, "_validate_content_length", lambda raw: None)
    monkeypatch.setattr(knowledge_api, "_MAX_UPLOAD_BYTES", 2)

    with TestClient(app) as client:
        response = client.post(
            "/v1/knowledge/upload",
            files={"file": ("policy.txt", b"abc", "text/plain")},
        )

    assert response.status_code == 413
    assert "File exceeds" in response.json()["detail"]


class _ChunkedUpload:
    """Minimal async UploadFile stub that yields the body in small reads."""

    def __init__(self, data: bytes, per_read: int = 4) -> None:
        self._data = data
        self._pos = 0
        self._per_read = per_read
        self.bytes_read = 0

    async def read(self, size: int = -1) -> bytes:
        remaining = len(self._data) - self._pos
        if size is None or size < 0:
            size = remaining
        take = min(size, self._per_read, remaining)
        out = self._data[self._pos : self._pos + take]
        self._pos += take
        self.bytes_read += take
        return out


async def test_read_within_limit_returns_full_body(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    monkeypatch.setattr(knowledge_api, "_MAX_UPLOAD_BYTES", 100)
    up = _ChunkedUpload(b"hello world payload", per_read=4)
    assert await knowledge_api._read_within_limit(up) == b"hello world payload"


async def test_read_within_limit_at_exact_limit_ok(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    monkeypatch.setattr(knowledge_api, "_MAX_UPLOAD_BYTES", 5)
    assert (
        await knowledge_api._read_within_limit(_ChunkedUpload(b"abcde", 2)) == b"abcde"
    )


async def test_read_within_limit_empty_body(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    monkeypatch.setattr(knowledge_api, "_MAX_UPLOAD_BYTES", 100)
    assert await knowledge_api._read_within_limit(_ChunkedUpload(b"")) == b""


async def test_read_within_limit_rejects_over_limit(monkeypatch) -> None:
    from fastapi import HTTPException

    import director_ai.knowledge_api as knowledge_api

    monkeypatch.setattr(knowledge_api, "_MAX_UPLOAD_BYTES", 5)
    with pytest.raises(HTTPException) as excinfo:
        await knowledge_api._read_within_limit(_ChunkedUpload(b"abcdef", 2))
    assert excinfo.value.status_code == 413


async def test_read_within_limit_does_not_buffer_whole_oversized_body(
    monkeypatch,
) -> None:
    from fastapi import HTTPException

    import director_ai.knowledge_api as knowledge_api

    monkeypatch.setattr(knowledge_api, "_MAX_UPLOAD_BYTES", 10)

    class _Huge:
        """A body with no Content-Length far larger than the limit."""

        def __init__(self, total: int) -> None:
            self._remaining = total
            self.bytes_read = 0

        async def read(self, size: int = -1) -> bytes:
            if self._remaining <= 0:
                return b""
            want = size if size and size > 0 else self._remaining
            n = min(want, self._remaining, 1024 * 1024)
            self._remaining -= n
            self.bytes_read += n
            return b"x" * n

    huge = _Huge(50 * 1024 * 1024)  # 50 MB body, 10-byte limit
    with pytest.raises(HTTPException):
        await knowledge_api._read_within_limit(huge)
    # Only a bounded prefix is read, never the whole 50 MB payload.
    assert huge.bytes_read <= 2 * 1024 * 1024


def test_ingest_duplicate_and_update_paths(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    unchanged_text = "same policy"
    changed_text = "new policy"
    existing = _Record(
        doc_id="doc1",
        source="policy",
        tenant_id="tenant-a",
        chunk_ids=["doc1:chunk:0"],
        created_at="created",
        updated_at="updated",
        content_hash=knowledge_api._content_hash(unchanged_text),
    )
    registry = _Registry([existing])
    backend = _EndpointBackend()
    backend.docs["doc1:chunk:0"] = {"text": unchanged_text, "metadata": {}}
    app, registry, store, _backend = _make_app(registry=registry, backend=backend)
    store.facts["doc1:chunk:0"] = unchanged_text
    monkeypatch.setattr(
        "director_ai.core.retrieval.doc_chunker.split",
        lambda text, config: [f"chunk:{text}"],
    )

    with TestClient(app) as client:
        duplicate = client.post(
            "/v1/knowledge/ingest",
            json={"text": "policy", "doc_id": "doc1", "source": "policy"},
        )
        unchanged = client.put(
            "/v1/knowledge/documents/doc1",
            json={"text": unchanged_text, "source": "policy"},
        )
        changed = client.put(
            "/v1/knowledge/documents/doc1",
            json={"text": changed_text, "source": "policy-v2"},
        )

    assert duplicate.status_code == 409
    assert unchanged.json()["unchanged"] is True
    assert changed.status_code == 200
    assert changed.json()["unchanged"] is False
    assert changed.json()["source"] == "policy-v2"
    assert registry.records["doc1"].content_hash == knowledge_api._content_hash(
        changed_text
    )


def test_update_missing_document_returns_404() -> None:
    app, _registry, _store, _backend = _make_app()

    with TestClient(app) as client:
        response = client.put(
            "/v1/knowledge/documents/doc1",
            json={"text": "new", "source": "policy"},
        )

    assert response.status_code == 404


def test_update_reports_stage_and_replace_failures(monkeypatch) -> None:
    import director_ai.knowledge_api as knowledge_api

    def _registry() -> _Registry:
        return _Registry(
            [
                _Record(
                    doc_id="doc1",
                    source="policy",
                    tenant_id="tenant-a",
                    chunk_ids=["doc1:chunk:0"],
                    created_at="created",
                    updated_at="updated",
                    content_hash=knowledge_api._content_hash("old"),
                )
            ]
        )

    monkeypatch.setattr(
        "director_ai.core.retrieval.doc_chunker.split",
        lambda text, config: ["new chunk"],
    )

    app, _registry_obj, _store, _backend = _make_app(registry=_registry())
    monkeypatch.setattr(
        knowledge_api,
        "_chunk_and_store",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stage failed")),
    )
    with TestClient(app) as client:
        staged = client.put(
            "/v1/knowledge/documents/doc1",
            json={"text": "new", "source": "policy"},
        )
    assert staged.status_code == 503
    assert "stage" in staged.json()["detail"]

    monkeypatch.undo()
    monkeypatch.setattr(
        "director_ai.core.retrieval.doc_chunker.split",
        lambda text, config: ["new chunk"],
    )
    backend = _EndpointBackend()
    backend.fail_delete = True
    app, _registry_obj, _store, _backend = _make_app(
        registry=_registry(),
        backend=backend,
    )
    with TestClient(app) as client:
        replaced = client.put(
            "/v1/knowledge/documents/doc1",
            json={"text": "new", "source": "policy"},
        )
    assert replaced.status_code == 503
    assert "replace" in replaced.json()["detail"]


def test_tune_embeddings_endpoint_validation_and_success(monkeypatch) -> None:
    import director_ai.core.retrieval.embedding_tuner as tuner

    one_doc = _Registry(
        [
            _Record(
                doc_id="doc1",
                source="policy",
                tenant_id="tenant-a",
                chunk_ids=["c1", "c2"],
                created_at="created",
                updated_at="updated",
                content_hash="h1",
            )
        ]
    )
    app, _registry, _store, _backend = _make_app(registry=one_doc)
    with TestClient(app) as client:
        response = client.post("/v1/knowledge/tune-embeddings")
    assert response.status_code == 422
    assert "at least 2 documents" in response.json()["detail"]

    two_docs_without_pairs = _Registry(
        [
            _Record(
                doc_id="doc1",
                source="one",
                tenant_id="tenant-a",
                chunk_ids=["c1"],
                created_at="created",
                updated_at="updated",
                content_hash="h1",
            ),
            _Record(
                doc_id="doc2",
                source="two",
                tenant_id="tenant-a",
                chunk_ids=["c2"],
                created_at="created",
                updated_at="updated",
                content_hash="h2",
            ),
        ]
    )
    app, _registry, _store, _backend = _make_app(registry=two_docs_without_pairs)
    with TestClient(app) as client:
        response = client.post("/v1/knowledge/tune-embeddings")
    assert response.status_code == 422
    assert "2+ chunks" in response.json()["detail"]

    ready_docs = _Registry(
        [
            _Record(
                doc_id="doc1",
                source="one",
                tenant_id="tenant-a",
                chunk_ids=["c1", "c2"],
                created_at="created",
                updated_at="updated",
                content_hash="h1",
            ),
            _Record(
                doc_id="doc2",
                source="two",
                tenant_id="tenant-a",
                chunk_ids=["c3", "c4"],
                created_at="created",
                updated_at="updated",
                content_hash="h2",
            ),
        ]
    )
    app, _registry, store, _backend = _make_app(registry=ready_docs)
    store.facts.update({"c1": "a", "c2": "b", "c3": "c", "c4": "d"})
    monkeypatch.setattr(
        tuner,
        "tune_embeddings",
        lambda documents: SimpleNamespace(
            model_path="/tmp/model",
            train_samples=len(documents),
            epochs=3,
        ),
    )

    with TestClient(app) as client:
        response = client.post("/v1/knowledge/tune-embeddings")

    assert response.status_code == 200
    assert response.json() == {
        "model_path": "/tmp/model",
        "train_samples": 2,
        "epochs": 3,
        "message": "Re-ingest documents to use tuned embeddings",
    }
