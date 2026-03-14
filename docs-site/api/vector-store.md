# VectorGroundTruthStore

Semantic vector store for RAG-based factual grounding. Ingest documents, then pass to `CoherenceScorer` for fact-checked scoring. Supports pluggable backends via a registry pattern.

## Usage

```python
from director_ai.core.vector_store import VectorGroundTruthStore

store = VectorGroundTruthStore()
store.ingest([
    "Refunds are available within 30 days of purchase.",
    "Standard shipping takes 5-7 business days.",
    "Pro plan costs $49/month.",
])

# Use with scorer
from director_ai import CoherenceScorer

scorer = CoherenceScorer(
    threshold=0.6,
    ground_truth_store=store,
    use_nli=True,
)
```

## VectorGroundTruthStore Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `VectorBackend \| None` | `None` | Backend instance (default: `InMemoryBackend`) |
| `top_k` | `int` | `3` | Number of chunks to retrieve |

## Methods

### ingest()

```python
store.ingest(documents: list[str]) -> None
```

Add documents to the store. Each document is embedded and indexed.

### retrieve_context()

```python
chunks = store.retrieve_context(query: str, top_k: int = 3) -> list[EvidenceChunk]
```

Retrieve the top-K most relevant chunks for a query.

---

## VectorBackend {: #vectorbackend }

Abstract protocol for vector storage backends. Implement `add()` and `query()` to create a custom backend.

```python
from director_ai.core.vector_store import VectorBackend

class MyBackend(VectorBackend):
    def add(self, texts: list[str], ids: list[str] | None = None) -> None:
        ...

    def query(self, text: str, top_k: int = 3) -> list[tuple[str, float]]:
        # Returns list of (text, distance) pairs
        ...
```

## Built-in Backends

| Backend | Install | Description |
|---------|---------|-------------|
| `InMemoryBackend` | included | TF-IDF cosine similarity. No deps, good for testing. |
| `SentenceTransformerBackend` | `pip install director-ai[embeddings]` | Dense embeddings via `sentence-transformers`. Production-quality. |
| `ChromaBackend` | `pip install director-ai[vector]` | ChromaDB persistent store. Scales to millions of documents. |

### ChromaBackend

```python
from director_ai.core.vector_store import ChromaBackend

backend = ChromaBackend(
    collection_name="legal_contracts",
    persist_directory="/data/chroma",
    embedding_model="BAAI/bge-large-en-v1.5",
)
store = VectorGroundTruthStore(backend=backend)
```

### SentenceTransformerBackend

```python
from director_ai.core.vector_store import SentenceTransformerBackend

backend = SentenceTransformerBackend(
    model_name="BAAI/bge-large-en-v1.5",
)
store = VectorGroundTruthStore(backend=backend)
```

## Backend Registry

Register custom backends for use with `DirectorConfig.vector_backend`:

```python
from director_ai.core.vector_store import register_vector_backend, get_vector_backend

register_vector_backend("qdrant", MyQdrantBackend)
backend = get_vector_backend("qdrant", **kwargs)
```

| Function | Purpose |
|----------|---------|
| `register_vector_backend(name, cls)` | Register a backend class |
| `get_vector_backend(name, **kwargs)` | Instantiate a registered backend |
| `list_vector_backends()` | List registered backend names |

## Reranking

Enable cross-encoder reranking for improved retrieval precision:

```python
scorer = CoherenceScorer(
    ground_truth_store=store,
    reranker_enabled=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_top_k_multiplier=3,  # Retrieve 3x, rerank to top_k
)
```

## Full API

::: director_ai.core.vector_store.VectorGroundTruthStore

::: director_ai.core.vector_store.VectorBackend

::: director_ai.core.vector_store.InMemoryBackend

::: director_ai.core.vector_store.ChromaBackend

::: director_ai.core.vector_store.SentenceTransformerBackend
