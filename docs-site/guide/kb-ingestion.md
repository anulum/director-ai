# Knowledge Base Ingestion

Director-AI scores responses against a knowledge base (KB). The KB provides the "ground truth" that the scorer compares responses to. Without a KB, the scorer falls back to heuristic word-overlap only.

## Option 1: Simple Key-Value Store

For small KBs (< 1000 facts), use `GroundTruthStore` directly:

```python
from director_ai import GroundTruthStore, CoherenceScorer

store = GroundTruthStore()
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

## Option 2: Vector Store for RAG Pipelines

For larger KBs or when you need semantic search, use `VectorGroundTruthStore`:

```python
from director_ai import VectorGroundTruthStore, CoherenceScorer

store = VectorGroundTruthStore(auto_index=True)

# Ingest documents (splits into chunks, embeds, indexes)
store.ingest([
    "Water boils at 100°C at standard atmospheric pressure.",
    "The speed of light in vacuum is 299,792 km/s.",
    "DNA has four nucleotide bases: adenine, thymine, guanine, cytosine.",
])

# Or add individual facts
store.add_fact("gravity", "Earth's gravitational acceleration is 9.81 m/s².")

scorer = CoherenceScorer(
    threshold=0.6,
    use_nli=True,
    ground_truth_store=store,
)
```

The in-memory backend works for up to ~10K documents. For production, use ChromaDB.

## Option 3: ChromaDB Persistent Backend

```python
from director_ai.core.vector_store import ChromaBackend, VectorGroundTruthStore

backend = ChromaBackend(
    collection_name="prod_facts",
    persist_directory="/data/chroma",
    embedding_model="BAAI/bge-large-en-v1.5",
)

store = VectorGroundTruthStore(backend=backend, auto_index=False)

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
from director_ai import TenantRouter

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
4. **Update strategy** — re-ingest the full corpus when facts change. `VectorGroundTruthStore` does not support incremental deletion (yet).
5. **Test coverage** — run `python -m benchmarks.regression_suite` after KB changes to verify the scorer still behaves correctly.
