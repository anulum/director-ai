# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Vector Store Backend Tests (Pinecone, Weaviate, Qdrant)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch

import pytest


class TestPineconeBackend:
    def test_pinecone_import_error(self):
        with (
            patch.dict("sys.modules", {"pinecone": None}),
            pytest.raises(ImportError, match="PineconeBackend requires pinecone"),
        ):
            from director_ai.core.vector_store import PineconeBackend

            PineconeBackend(api_key="key", index_name="idx")

    def test_pinecone_add_query(self):
        mock_pinecone = MagicMock()
        mock_index = MagicMock()
        mock_pinecone.Pinecone.return_value.Index.return_value = mock_index
        mock_index.query.return_value = {
            "matches": [{"id": "d1", "score": 0.9, "metadata": {"text": "hello world"}}]
        }

        with patch.dict("sys.modules", {"pinecone": mock_pinecone}):
            from director_ai.core.vector_store import PineconeBackend

            def embed_fn(text):
                return [0.1, 0.2, 0.3]

            backend = PineconeBackend(api_key="k", index_name="i", embed_fn=embed_fn)
            backend.add("d1", "hello world")
            mock_index.upsert.assert_called_once()

            results = backend.query("hello", n_results=1)
            assert len(results) == 1
            assert results[0]["id"] == "d1"
            assert results[0]["distance"] == pytest.approx(0.1)


class TestWeaviateBackend:
    def test_weaviate_import_error(self):
        with (
            patch.dict("sys.modules", {"weaviate": None}),
            pytest.raises(
                ImportError, match="WeaviateBackend requires weaviate-client"
            ),
        ):
            from director_ai.core.vector_store import WeaviateBackend

            WeaviateBackend(url="http://localhost:8080")


class TestQdrantBackend:
    def test_qdrant_import_error(self):
        with (
            patch.dict("sys.modules", {"qdrant_client": None}),
            pytest.raises(ImportError, match="QdrantBackend requires qdrant-client"),
        ):
            from director_ai.core.vector_store import QdrantBackend

            QdrantBackend(url="localhost")


class TestBackendImportErrors:
    def test_all_backends_give_clear_import_messages(self):
        backends = [
            ("pinecone", "PineconeBackend", "pinecone"),
            ("weaviate", "WeaviateBackend", "weaviate-client"),
            ("qdrant_client", "QdrantBackend", "qdrant-client"),
        ]
        for module, cls_name, pkg_name in backends:
            with (
                patch.dict("sys.modules", {module: None}),
                pytest.raises(ImportError, match=pkg_name),
            ):
                from director_ai.core import vector_store

                getattr(vector_store, cls_name)(
                    **{"api_key": "k", "index_name": "i"}
                    if cls_name == "PineconeBackend"
                    else {"url": "http://localhost:8080"}
                    if cls_name == "WeaviateBackend"
                    else {"url": "localhost"}
                )
