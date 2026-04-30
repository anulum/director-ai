# VectorGroundTruthStore

Semantic vector store for RAG-based factual grounding. Ingest documents, then pass to `CoherenceScorer` for fact-checked scoring. Supports pluggable backends via a registry pattern.

## Usage

```python
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

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
| `tenant_id` | `str` | `""` | Default tenant ID for multi-tenant stores |

## Methods

### add()

```python
store.add(
    key="refund-policy",
    value="Refunds are available within 30 days.",
    metadata={"kb_version_bump": "patch"},  # patch, minor, or major
)
```

Facts start at `1.0.0`. Replacing a fact with different content bumps the patch
version by default. Pass `kb_version_bump="minor"` or `"major"` in metadata when
the source change is a larger schema or policy change. The vector metadata is
stamped with:

- `kb_version`
- `kb_chunk_version`
- `kb_content_hash`
- `kb_previous_hash`
- `kb_record_kind`
- `kb_source_key`
- `kb_chunk_index`

Use `fact_version(key)`, `fact_version_record(key)`, or `version_manifest()` to
inspect the in-process version ledger.

### retract_fact()

```python
store.retract_fact("refund-policy", reason="source withdrawn")
```

Retraction records mark a fact or derived chunk source as unusable for retrieval
without deleting backend rows. `retrieve_context()` and
`retrieve_context_with_chunks()` filter matching vector results and keyword
fallback facts after retraction. Use `retraction_records()` to inspect the
event log.

### replace_fact()

```python
store.replace_fact(
    "refund-policy",
    "Refunds are available within 45 days.",
    reason="policy update",
)
```

Replacement records preserve the superseded version and content hash while the
new value is indexed under the same key. Use `replacement_records()` to inspect
the event log.

### kb_snapshot_root()

```python
root = store.kb_snapshot_root(tenant_id="acme")
audit_payload = store.kb_snapshot_audit_record(tenant_id="acme")
```

`kb_snapshot_root()` returns a deterministic SHA-256 Merkle root over the
tenant-visible KB version ledger. Leaves are sorted by tenant, key, record
kind, and chunk index, so the same KB state produces the same root regardless
of ingestion order. Retractions and replacements update the root because the
snapshot includes status, current content hash, previous hash, and semantic
version fields.

`kb_snapshot_audit_record()` returns a compact payload with:

- `event`
- `tenant_id`
- `revision`
- `record_count`
- `retraction_count`
- `replacement_count`
- `merkle_root`

Pass this payload to `AuditLogger.log_review(kb_snapshot=audit_payload)` when a
review decision must carry the KB state it was grounded against.

### freshness_status_signals()

```python
store.add_fact(
    "trial-paper",
    "Trial X reported a 12 percent response rate.",
    metadata={
        "external_id": "doi:10.example/trial-paper",
        "source_timestamp": "1710000000",
        "citation_status": "active",
        "status_source": "publisher-feed",
    },
)

signals = store.freshness_status_signals()
```

Use `freshness_status_signals()` to pass KB source age and external citation
status metadata into `score_temporal_freshness(citation_statuses=signals)`.
The method emits tenant-scoped dictionaries with `source_id`, `status`,
`status_source`, and any available timestamp fields.

### ingest()

```python
store.ingest(documents: list[str]) -> None
```

Add documents to the store. Each document is embedded and indexed as a derived
vector chunk with `kb_record_kind="derived_chunk"` and a semantic chunk version.

### retrieve_context()

```python
context = store.retrieve_context(query: str, top_k: int = 3, tenant_id: str = "") -> str | None
```

Retrieve concatenated context string for a query (matching parent `GroundTruthStore` interface). Use `retrieve_context_with_chunks()` for structured `EvidenceChunk` results.

---

## VectorBackend {: #vectorbackend }

Abstract protocol for vector storage backends. Implement `add()` and `query()` to create a custom backend.

```python
from director_ai.core.retrieval.vector_store import VectorBackend

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
from director_ai.core.retrieval.vector_store import ChromaBackend

backend = ChromaBackend(
    collection_name="legal_contracts",
    persist_directory="/data/chroma",
    embedding_model="BAAI/bge-large-en-v1.5",
)
store = VectorGroundTruthStore(backend=backend)
```

### SentenceTransformerBackend

```python
from director_ai.core.retrieval.vector_store import SentenceTransformerBackend

backend = SentenceTransformerBackend(
    model_name="BAAI/bge-large-en-v1.5",
)
store = VectorGroundTruthStore(backend=backend)
```

## Backend Registry

Register custom backends for use with `DirectorConfig.vector_backend`:

```python
from director_ai.core.retrieval.vector_store import register_vector_backend, get_vector_backend

register_vector_backend("qdrant", MyQdrantBackend)
BackendClass = get_vector_backend("qdrant")  # returns the class, not an instance
backend = BackendClass(**kwargs)
```

| Function | Purpose |
|----------|---------|
| `register_vector_backend(name, cls)` | Register a backend class |
| `get_vector_backend(name)` | Look up a registered backend class |
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

::: director_ai.core.retrieval.vector_store.VectorGroundTruthStore

::: director_ai.core.retrieval.vector_store.VectorBackend

::: director_ai.core.retrieval.vector_store.InMemoryBackend

::: director_ai.core.retrieval.vector_store.ChromaBackend

::: director_ai.core.retrieval.vector_store.SentenceTransformerBackend
