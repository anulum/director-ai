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

from importlib.util import find_spec

from .base import _VECTOR_EP_LOADED as _VECTOR_EP_LOADED
from .base import _VECTOR_REGISTRY as _VECTOR_REGISTRY
from .base import RECOMMENDED_EMBEDDING_MODEL as RECOMMENDED_EMBEDDING_MODEL
from .base import InMemoryBackend as InMemoryBackend
from .base import VectorBackend as VectorBackend
from .base import get_vector_backend as get_vector_backend
from .base import list_vector_backends as list_vector_backends
from .base import register_vector_backend as register_vector_backend
from .composite import HybridBackend as HybridBackend
from .composite import RerankedBackend as RerankedBackend
from .embedding import ChromaBackend as ChromaBackend
from .embedding import SentenceTransformerBackend as SentenceTransformerBackend
from .store import VectorGroundTruthStore as VectorGroundTruthStore
from .vendors import ColBERTBackend as ColBERTBackend
from .vendors import ElasticsearchBackend as ElasticsearchBackend
from .vendors import FAISSBackend as FAISSBackend
from .vendors import PineconeBackend as PineconeBackend
from .vendors import QdrantBackend as QdrantBackend
from .vendors import WeaviateBackend as WeaviateBackend

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
# ``get_vector_backend("memory")`` works without an explicit call.
register_vector_backend("memory", InMemoryBackend)
register_vector_backend("sentence-transformer", SentenceTransformerBackend)
register_vector_backend("hybrid", HybridBackend)


# Probe optional vendor packages via ``importlib.util.find_spec`` —
# this checks availability without actually importing the heavy
# modules, so a missing vendor never raises and no unused-import
# warning is ever triggered.
if find_spec("chromadb") is not None:
    register_vector_backend("chroma", ChromaBackend)
if find_spec("pinecone") is not None:
    register_vector_backend("pinecone", PineconeBackend)
if find_spec("weaviate") is not None:
    register_vector_backend("weaviate", WeaviateBackend)
if find_spec("qdrant_client") is not None:
    register_vector_backend("qdrant", QdrantBackend)
if find_spec("faiss") is not None:
    register_vector_backend("faiss", FAISSBackend)
if find_spec("elasticsearch") is not None:
    register_vector_backend("elasticsearch", ElasticsearchBackend)

# ColBERT imports its heavy dependency lazily from the constructor, so
# the class is always registrable — the ImportError surfaces only when
# a caller actually instantiates the backend.
register_vector_backend("colbert", ColBERTBackend)
