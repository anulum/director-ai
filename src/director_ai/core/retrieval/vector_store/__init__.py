# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector store package

"""Public surface of the vector_store package.

Mirrors the legacy flat module so existing imports such as
``from director_ai.core.retrieval.vector_store import ChromaBackend``
continue to work. New code may import from the submodules directly.
"""

from __future__ import annotations

from .base import (
    _VECTOR_EP_LOADED,  # noqa: F401 — legacy tests import the private
    _VECTOR_REGISTRY,  # noqa: F401 — legacy tests reach in directly
    RECOMMENDED_EMBEDDING_MODEL,
    InMemoryBackend,
    VectorBackend,
    get_vector_backend,
    list_vector_backends,
    register_vector_backend,
)
from .composite import HybridBackend, RerankedBackend
from .embedding import ChromaBackend, SentenceTransformerBackend
from .store import VectorGroundTruthStore
from .vendors import (
    ColBERTBackend,
    ElasticsearchBackend,
    FAISSBackend,
    PineconeBackend,
    QdrantBackend,
    WeaviateBackend,
)

__all__ = [
    "RECOMMENDED_EMBEDDING_MODEL",
    "ChromaBackend",
    "ColBERTBackend",
    "ElasticsearchBackend",
    "FAISSBackend",
    "HybridBackend",
    "InMemoryBackend",
    "PineconeBackend",
    "QdrantBackend",
    "RerankedBackend",
    "SentenceTransformerBackend",
    "VectorBackend",
    "VectorGroundTruthStore",
    "WeaviateBackend",
    "get_vector_backend",
    "list_vector_backends",
    "register_vector_backend",
]


# Auto-register built-in vector backends on import so
# ``get_vector_backend("memory")`` works without an explicit
# register call.
register_vector_backend("memory", InMemoryBackend)
register_vector_backend("sentence-transformer", SentenceTransformerBackend)
register_vector_backend("hybrid", HybridBackend)

try:
    import chromadb as _chromadb  # noqa: F401

    register_vector_backend("chroma", ChromaBackend)
except ImportError:
    pass

try:
    import pinecone as _pinecone  # noqa: F401

    register_vector_backend("pinecone", PineconeBackend)
except ImportError:
    pass

try:
    import weaviate as _weaviate  # noqa: F401

    register_vector_backend("weaviate", WeaviateBackend)
except ImportError:
    pass

try:
    from qdrant_client import QdrantClient as _QdrantClient  # noqa: F401

    register_vector_backend("qdrant", QdrantBackend)
except ImportError:
    pass

try:
    import faiss as _faiss  # noqa: F401

    register_vector_backend("faiss", FAISSBackend)
except ImportError:
    pass

try:
    from elasticsearch import Elasticsearch as _Elasticsearch  # noqa: F401

    register_vector_backend("elasticsearch", ElasticsearchBackend)
except ImportError:
    pass

# ColBERT imports its heavy dependency lazily from the constructor, so
# the class is always registrable — the ImportError surfaces only when
# a caller actually instantiates the backend.
register_vector_backend("colbert", ColBERTBackend)
