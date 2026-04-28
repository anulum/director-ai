# Knowledge Ingestion API

Upload, manage, and search customer documents for grounded hallucination detection.

## Endpoints

All endpoints are tenant-scoped via `X-Tenant-ID` header.

### Upload File

```bash
curl -X POST http://localhost:8080/v1/knowledge/upload \
  -H 'X-API-Key: key' -H 'X-Tenant-ID: acme' \
  -F 'file=@product_manual.pdf'
```

Supported formats: PDF, DOCX, HTML, CSV, TXT, Markdown.

### Ingest Text

```bash
curl -X POST http://localhost:8080/v1/knowledge/ingest \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: acme' \
  -d '{"text": "Our return policy allows 30 days...", "source": "policy_v2"}'
```

Optional request fields:

| Field | Default | Notes |
|---|---|---|
| `doc_id` | generated | Stable document id. Duplicate ids return `409`; use `PUT` to replace. |
| `chunk_size` | `512` | Must be between `64` and `4096`. |
| `overlap` | `64` | Must be smaller than `chunk_size`. |
| `source` | `text` | Stored in document metadata and updated on replacement. |

### List Documents

```bash
curl http://localhost:8080/v1/knowledge/documents \
  -H 'X-Tenant-ID: acme'
```

### Get Document

```bash
curl http://localhost:8080/v1/knowledge/documents/{doc_id} \
  -H 'X-Tenant-ID: acme'
```

### Delete Document

```bash
curl -X DELETE http://localhost:8080/v1/knowledge/documents/{doc_id} \
  -H 'X-Tenant-ID: acme'
```

### Update Document

```bash
curl -X PUT http://localhost:8080/v1/knowledge/documents/{doc_id} \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: acme' \
  -d '{"text": "Updated content...", "source": "policy_v3", "chunk_size": 512, "overlap": 64}'
```

Update replaces all chunks for that document id, refreshes the source metadata,
and leaves other tenant documents untouched.

### Search

```bash
curl 'http://localhost:8080/v1/knowledge/search?query=return+policy&top_k=5' \
  -H 'X-Tenant-ID: acme'
```

Search returns at most 500 characters of each matching chunk plus distance and
metadata. `top_k` must be between `1` and `50`.

### Tune Embeddings

Fine-tune the embedding model on ingested documents for domain-specific retrieval:

```bash
curl -X POST http://localhost:8080/v1/knowledge/tune-embeddings
```

Requires at least 2 documents with 2+ chunks each. After tuning, re-ingest documents to use the improved embeddings.

## Chunking

Documents are automatically split using recursive character splitting with sentence-boundary snapping. Configure via `chunk_size` (default 512) and `overlap` (default 64) in the ingest request body.

For semantic chunking (splits on topic boundaries), set `semantic: true` in the chunk config.

## Retrieval Pipeline

```
Upload → Parse → Chunk → Embed → Store
                                    ↓
Query → Hybrid Retrieval (BM25 + Dense + RRF) → Rerank → NLI Score
```

Default pipeline uses `BAAI/bge-large-en-v1.5` embeddings with hybrid BM25+dense retrieval and `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder reranking.
