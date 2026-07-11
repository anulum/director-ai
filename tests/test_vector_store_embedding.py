# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — embedding vector backend tests

from __future__ import annotations

import sys
import types

import pytest

from director_ai.core.retrieval.vector_store.embedding import (
    ChromaBackend,
    SentenceTransformerBackend,
    _DirectorChromaEmbeddingAdapter,
    _normalise_chroma_embedding_function,
)


def _install_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str | None = None) -> None:
            self.model_name = model_name
            self.device = device

        def encode(self, text: str, *, normalize_embeddings: bool):
            assert normalize_embeddings is True
            if "opposite" in text:
                return [-1.0, 0.0]
            if "vertical" in text:
                return [0.0, 1.0]
            return [1.0, 0.0]

    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_sentence_transformer_backend_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError, match="sentence-transformers"):
        SentenceTransformerBackend()


def test_sentence_transformer_backend_add_query_count_and_delete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sentence_transformer(monkeypatch)
    backend = SentenceTransformerBackend("local-embedder")

    assert backend.query("anything") == []
    backend.add("doc-a", "alpha policy", {"tenant_id": "tenant-a"})
    backend.add("doc-b", "vertical policy", {"tenant_id": "tenant-b"})
    backend.add("doc-c", "opposite policy", {"tenant_id": "tenant-a"})

    assert backend.count() == 3
    tenant_results = backend.query("alpha question", n_results=3, tenant_id="tenant-a")
    assert [row["id"] for row in tenant_results] == ["doc-a"]
    assert backend.query("alpha question", tenant_id="missing") == []
    assert [
        row["id"] for row in backend.query("opposite question", tenant_id="tenant-a")
    ] == ["doc-c"]

    with pytest.raises(ValueError, match="doc_ids must be a list"):
        backend.delete("doc-a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        backend.delete(["doc-a", " "])
    assert backend.delete([]) == 0
    assert backend.delete(["doc-a", "missing"]) == 1
    assert backend.count() == 2


def test_sentence_transformer_backend_accepts_injected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    class InjectedModel:
        def encode(self, text: str, *, normalize_embeddings: bool):
            assert normalize_embeddings is True
            return [0.0, 1.0] if "vertical" in text else [1.0, 0.0]

    backend = SentenceTransformerBackend(model=InjectedModel())

    backend.add("doc-a", "alpha policy")
    backend.add("doc-b", "vertical policy")
    assert [row["id"] for row in backend.query("alpha question")] == ["doc-a"]


def test_sentence_transformer_backend_rejects_model_without_encode() -> None:
    class NotAModel:
        encode = "not callable"

    with pytest.raises(ValueError, match="callable encode method"):
        SentenceTransformerBackend(model=NotAModel())  # type: ignore[arg-type]


class _FakeCollection:
    def __init__(self) -> None:
        self.added: list[dict[str, object]] = []
        self.deleted: list[list[str]] = []
        self.query_result: dict[str, object] = {}
        self._count = 0

    def add(self, *, ids, documents, metadatas) -> None:
        self.added.append({"ids": ids, "documents": documents, "metadatas": metadatas})
        self._count += len(ids)

    def query(self, *, query_texts, n_results, where):
        return self.query_result

    def count(self) -> int:
        return self._count

    def delete(self, *, ids) -> None:
        self.deleted.append(ids)
        self._count = max(self._count - len(ids), 0)


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collection = _FakeCollection()
        self.create_kwargs: dict[str, object] | None = None

    def get_or_create_collection(self, **kwargs):
        self.create_kwargs = kwargs
        return self.collection


def _install_chroma(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_embedding_function: bool = True,
) -> dict[str, object]:
    calls: dict[str, object] = {}
    chromadb = types.ModuleType("chromadb")

    def client_factory():
        client = _FakeChromaClient()
        calls["client"] = client
        return client

    def persistent_factory(*, path: str):
        client = _FakeChromaClient()
        calls["persistent_path"] = path
        calls["client"] = client
        return client

    chromadb.Client = client_factory
    chromadb.PersistentClient = persistent_factory
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)

    if include_embedding_function:
        utils = types.ModuleType("chromadb.utils")
        embedding_functions = types.ModuleType("chromadb.utils.embedding_functions")

        class SentenceTransformerEmbeddingFunction:
            def __init__(self, *, model_name: str) -> None:
                self.model_name = model_name

        embedding_functions.SentenceTransformerEmbeddingFunction = (
            SentenceTransformerEmbeddingFunction
        )
        monkeypatch.setitem(sys.modules, "chromadb.utils", utils)
        monkeypatch.setitem(
            sys.modules,
            "chromadb.utils.embedding_functions",
            embedding_functions,
        )
    else:
        monkeypatch.delitem(sys.modules, "chromadb.utils", raising=False)
        monkeypatch.delitem(
            sys.modules, "chromadb.utils.embedding_functions", raising=False
        )

    return calls


def test_chroma_backend_requires_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "chromadb", None)

    with pytest.raises(ImportError, match="ChromaDB backend"):
        ChromaBackend()


def test_chroma_backend_add_query_count_delete_and_embedding_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_chroma(monkeypatch)

    backend = ChromaBackend(
        collection_name="tenant_facts",
        persist_directory="/tmp/chroma",
        embedding_model="local-embedder",
    )
    client = calls["client"]
    collection = client.collection

    assert calls["persistent_path"] == "/tmp/chroma"
    assert collection.count() == 0
    assert backend.query("policy") == []

    backend.add("doc-a", "alpha", {"tenant_id": "tenant-a"})
    collection.query_result = {
        "documents": [["alpha"]],
        "metadatas": [[{"tenant_id": "tenant-a"}]],
        "ids": [["doc-a"]],
        "distances": [[0.2]],
    }

    results = backend.query("policy", n_results=5, tenant_id="tenant-a")

    assert collection.added == [
        {
            "ids": ["doc-a"],
            "documents": ["alpha"],
            "metadatas": [{"tenant_id": "tenant-a"}],
        }
    ]
    assert client.create_kwargs["name"] == "tenant_facts"
    assert client.create_kwargs["embedding_function"].model_name == "local-embedder"
    assert results == [
        {
            "id": "doc-a",
            "text": "alpha",
            "metadata": {"tenant_id": "tenant-a"},
            "distance": 0.2,
        }
    ]
    assert backend.count() == 1
    assert backend.delete(["doc-a"]) == 1
    assert collection.deleted == [["doc-a"]]


def test_chroma_backend_defaults_and_delete_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_chroma(monkeypatch, include_embedding_function=False)
    default_backend = ChromaBackend()
    default_collection = calls["client"].collection
    assert "embedding_function" not in calls["client"].create_kwargs
    assert default_backend.count() == 0
    assert default_collection.count() == 0

    backend = ChromaBackend(embedding_model="missing-extra")
    collection = calls["client"].collection
    collection._count = 2
    collection.query_result = {"documents": [["alpha"]]}

    assert backend.query("policy") == [
        {"id": "doc_0", "text": "alpha", "metadata": {}, "distance": 0.0}
    ]
    with pytest.raises(ValueError, match="doc_ids must be a list"):
        backend.delete("doc-a")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty strings"):
        backend.delete([""])
    assert backend.delete([]) == 0


def _local_embedder(input: list[str]) -> list[list[float]]:
    return [[float(len(text)), 0.0] for text in input]


def test_chroma_embedding_adapter_wraps_local_callable() -> None:
    adapter = _DirectorChromaEmbeddingAdapter(_local_embedder)

    assert adapter(["ab", "c"]) == [[2.0, 0.0], [1.0, 0.0]]
    assert adapter.embed_query(["abc"]) == [[3.0, 0.0]]
    assert adapter.name() == "director-ai-local-embedding"
    assert adapter.default_space() == "l2"
    assert adapter.supported_spaces() == ["cosine", "l2", "ip"]
    assert adapter.get_config() == {"provider": "director-ai-local-embedding"}
    assert adapter.is_legacy() is False


def test_chroma_embedding_adapter_rejects_config_deserialisation() -> None:
    with pytest.raises(ValueError, match="must be provided at runtime"):
        _DirectorChromaEmbeddingAdapter.build_from_config(
            {"provider": "director-ai-local-embedding"},
        )


def test_normalise_chroma_embedding_function_passes_through_full_providers() -> None:
    adapter = _DirectorChromaEmbeddingAdapter(_local_embedder)

    assert _normalise_chroma_embedding_function(adapter) is adapter


def test_normalise_chroma_embedding_function_wraps_plain_callables() -> None:
    normalised = _normalise_chroma_embedding_function(_local_embedder)

    assert isinstance(normalised, _DirectorChromaEmbeddingAdapter)
    assert normalised(["xy"]) == [[2.0, 0.0]]


def test_chroma_backend_rejects_model_and_function_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_chroma(monkeypatch)

    with pytest.raises(ValueError, match="mutually exclusive"):
        ChromaBackend(
            embedding_model="local-embedder",
            embedding_function=_local_embedder,
        )


def test_chroma_backend_normalises_injected_embedding_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_chroma(monkeypatch)

    ChromaBackend(embedding_function=_local_embedder)

    injected = calls["client"].create_kwargs["embedding_function"]
    assert isinstance(injected, _DirectorChromaEmbeddingAdapter)
    assert injected(["ab"]) == [[2.0, 0.0]]
