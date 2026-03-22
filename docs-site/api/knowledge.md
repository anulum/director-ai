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
  -d '{"text": "Our return policy allows 30 days...", "source": "policy_v2"}'
```

### List Documents

```bash
curl http://localhost:8080/v1/knowledge/documents \
  -H 'X-Tenant-ID: acme'
```

### Delete Document

```bash
curl -X DELETE http://localhost:8080/v1/knowledge/documents/{doc_id}
```

### Update Document

```bash
curl -X PUT http://localhost:8080/v1/knowledge/documents/{doc_id} \
  -H 'Content-Type: application/json' \
  -d '{"text": "Updated content...", "source": "policy_v3"}'
```

### Search

```bash
curl 'http://localhost:8080/v1/knowledge/search?query=return+policy&top_k=5'
```

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
