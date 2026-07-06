# Knowledge Base Ingestion

Director-AI scores responses against a knowledge base (KB). The KB provides the "ground truth" that the scorer compares responses to. Without a KB, NLI still works (using the prompt as premise), but factual divergence (H_factual) degrades to a neutral 0.5. Loading domain facts substantially improves scoring discrimination.

## Option 1: Simple Key-Value Store

For small KBs (< 1000 facts), use `GroundTruthStore` directly:

```python
from director_ai import GroundTruthStore, CoherenceScorer

store = GroundTruthStore()  # starts empty — add your own facts
store.add("boiling_point", "Water boils at 100°C at standard atmospheric pressure.")
store.add("speed_of_light", "The speed of light in vacuum is 299,792 km/s.")
store.add("capital_france", "The capital of France is Paris.")

scorer = CoherenceScorer(
    threshold=0.5,
    ground_truth_store=store,
)

approved, score = scorer.review(
    "What temperature does water boil at?",
    "Water boils at 100 degrees Celsius at standard pressure.",
)
# approved=True, score.score ≈ 0.9+
```

Facts are matched by key similarity to the prompt. The scorer retrieves the best-matching fact and computes word overlap (heuristic) or NLI entailment (when `use_nli=True`).

## Option 2: Grounded Retrieval Recipe for RAG Pipelines

For larger KBs or when you need semantic search, build the store through
`DirectorConfig`. This exposes the default grounded recipe as an explicit
contract:

- dense retrieval from the configured vector backend
- BM25 sparse retrieval fused with dense retrieval by Reciprocal Rank Fusion
  (`hybrid_rrf_k=60`)
- cross-encoder reranking when `director-ai[reranker]` dependencies are
  installed
- retrieval abstention for low-quality matches instead of false confidence
  (`retrieval_abstention_threshold=0.3` by default)

Use the in-memory backend for local development and test corpora. Use a
persistent backend such as ChromaDB for production storage.

```python
from director_ai.core.config import DirectorConfig

cfg = DirectorConfig(
    mode="grounded",
    vector_backend="chroma",
    chroma_persist_dir="/data/chroma",
)

store = cfg.build_store()

# Ingest documents (splits into chunks, embeds, indexes)
store.ingest([
    "Water boils at 100°C at standard atmospheric pressure.",
    "The speed of light in vacuum is 299,792 km/s.",
    "DNA has four nucleotide bases: adenine, thymine, guanine, cytosine.",
])

# Or add individual facts
store.add_fact("gravity", "Earth's gravitational acceleration is 9.81 m/s².")

scorer = cfg.build_scorer(store=store)
```

For user-supplied or externally synchronized facts, place
`ConflictAwareKnowledgeGuard` in front of the store so contradictory updates are
checked before they become retrievable context:

```python
from director_ai import ConflictAwareKnowledgeGuard, KnowledgeFact

guard = ConflictAwareKnowledgeGuard(store, score_fn=contradiction_score)
result = guard.add_fact(
    KnowledgeFact(
        key="refund_policy",
        value="Refunds are unavailable after delivery confirmation.",
        metadata={"contradicts": "claim-refund-v1"},
    )
)

if result.decision == "block":
    raise RuntimeError("KB update rejected before retrieval admission")
```

Conflict reports carry hashes and evidence references, not raw fact text. Use
this guard at ingestion boundaries where tenant users, signed facts, passport
claims, or upstream synchronizers can introduce mutually incompatible facts.

## External Source Plugins

Use the ingestion plugins when facts are synchronized from operator-owned
systems such as S3, Notion, or Google Drive. The plugins accept authenticated
SDK clients, stream `IngestedDocument` records, skip blank text, and write
metadata through the same `GroundTruthStore.add()` contract used by
`VectorGroundTruthStore`.

```python
from director_ai.core.retrieval.ingestion import S3Plugin
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

store = VectorGroundTruthStore(tenant_id="acme")
plugin = S3Plugin.from_default_client(
    "acme-knowledge-base",
    prefix="policies/",
)

written = plugin.ingest(store, tenant_id="acme", max_documents=100)
print(f"Loaded {written} knowledge documents")
```

Install only the source adapters you need:

```bash
pip install "director-ai[ingestion-s3]"
pip install "director-ai[ingestion-notion]"
pip install "director-ai[ingestion-gdrive]"
```

Inspect the active recipe without exposing API keys:

```python
recipe = cfg.retrieval_recipe()
assert recipe["name"] == "grounded-hybrid-rerank-v1"
```

Production deployments should still benchmark the configured corpus and domain
thresholds before enforcement. The default recipe is a strong retrieval baseline,
not a substitute for corpus-specific calibration.

## Option 3: ChromaDB Persistent Backend

```python
from director_ai.core.vector_store import ChromaBackend, VectorGroundTruthStore

backend = ChromaBackend(
    collection_name="prod_facts",
    persist_directory="/data/chroma",
    embedding_model="BAAI/bge-large-en-v1.5",
)

store = VectorGroundTruthStore(backend=backend)

# Ingest once — data persists across restarts
store.ingest(your_documents)

# Later, just connect to the same persist_directory
```

Install ChromaDB:

```bash
pip install director-ai[vector]
```

## Multi-Tenant KB

Use `TenantRouter` to isolate KBs per tenant:

```python
from director_ai.enterprise import TenantRouter

router = TenantRouter()

# Each tenant gets an isolated KB
router.add_fact("acme_corp", "refund_policy", "Refunds within 30 days of purchase.")
router.add_fact("globex", "refund_policy", "No refunds after delivery confirmation.")

# Get a scorer scoped to one tenant
scorer = router.get_scorer("acme_corp", threshold=0.6)

# This scorer only sees acme_corp's facts
approved, score = scorer.review(
    "What is the refund policy?",
    "You can get a refund within 30 days.",
)
```

When using the Director-AI server with `tenant_routing=True`, pass the `X-Tenant-ID` header to route requests to the correct KB:

```bash
curl -X POST http://localhost:8080/v1/review \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: acme_corp" \
  -d '{"prompt": "refund policy?", "response": "Refunds within 30 days."}'
```

## Ingestion Best Practices

1. **Chunk size** — keep documents under 500 tokens each. Long documents dilute retrieval precision.
2. **Deduplication** — avoid ingesting the same content twice; it inflates retrieval scores without adding information.
3. **Metadata** — use `add_fact(key, value)` with meaningful keys for small KBs. The key helps with retrieval matching.
4. **Incremental sync** — keep a stable upstream document id and call `PUT /v1/knowledge/documents/{doc_id}` when the upstream file changes. Director-AI stores a SHA-256 `content_hash`; unchanged `PUT` requests return `unchanged: true` without deleting chunks or re-embedding.
5. **Update strategy** — use `PUT /v1/knowledge/documents/{doc_id}` to replace one document and `DELETE /v1/knowledge/documents/{doc_id}` to remove its chunks. Re-ingest the full corpus only when the embedding model, chunking policy, or parser configuration changes.
6. **Test coverage** — run `python -m benchmarks.regression_suite` after KB changes to verify the scorer still behaves correctly.
