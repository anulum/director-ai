# Production Deployment Guide

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   LLM API    │───▶│  Director-AI │───▶│  Your App    │
│  (OpenAI/    │    │  Guardrail   │    │  (FastAPI/   │
│   Anthropic) │    │              │    │   Flask/etc) │
└──────────────┘    └──────┬───────┘    └──────────────┘
                           │
                    ┌──────┴───────┐
                    │  Knowledge   │
                    │  Base (RAG)  │
                    │  ChromaDB/   │
                    │  Qdrant/etc  │
                    └──────────────┘
```

## Recommended Setup

```bash
pip install director-ai[nli,vector,embeddings,openai]
```

```python
from director_ai import guard, CoherenceScorer, VectorGroundTruthStore, ChromaBackend

# Production vector store with persistent storage
backend = ChromaBackend(
    collection_name="prod_facts",
    persist_directory="/data/chroma",
    embedding_model="BAAI/bge-large-en-v1.5",
)
store = VectorGroundTruthStore(backend=backend, auto_index=False)

# Ingest your knowledge base
store.ingest(your_documents)

# Create scorer with caching and NLI
scorer = CoherenceScorer(
    threshold=0.6,
    soft_limit=0.7,
    use_nli=True,
    nli_model="lytang/MiniCheck-DeBERTa-L",
    ground_truth_store=store,
    cache_size=2048,
    cache_ttl=600,
    nli_quantize_8bit=True,
    nli_device="cuda",
)
```

## Scaling

### Horizontal: Multiple Workers

```bash
uvicorn director_ai.server:app --workers 4 --host 0.0.0.0 --port 8080
```

Each worker gets its own scorer instance. The NLI model is loaded once per worker via `lru_cache`.

### GPU Sharing

For multi-worker GPU sharing, load the model once and share via torch multiprocessing:

```python
scorer = CoherenceScorer(
    use_nli=True,
    nli_device="cuda:0",
    nli_torch_dtype="float16",
    nli_quantize_8bit=True,
)
```

8-bit quantization reduces VRAM from ~1.5GB to ~400MB per model.

### Score Caching

Enable caching to reduce NLI inference by 60-80% in streaming workloads:

```python
scorer = CoherenceScorer(
    cache_size=4096,
    cache_ttl=300,
)

# Monitor cache performance
print(f"Hit rate: {scorer.cache.hit_rate:.1%}")
```

## Resource Sizing

| Workload | CPU | RAM | GPU | Latency (measured) |
|----------|-----|-----|-----|--------------------|
| Heuristic only | 1 core | 256MB | None | <0.1 ms |
| **ONNX GPU batch** | 2 cores | 2GB | 1.2GB VRAM | **14.6 ms/pair** |
| PyTorch GPU batch | 2 cores | 2GB | 1.2GB VRAM | 19 ms/pair |
| ONNX GPU sequential | 2 cores | 2GB | 1.2GB VRAM | 65 ms/pair |
| PyTorch GPU sequential | 2 cores | 2GB | 1.2GB VRAM | 197 ms/pair |
| ONNX CPU batch | 4 cores | 2GB | None | 383 ms/pair |
| MiniCheck (GPU) | 2 cores | 1GB | 400MB VRAM | ~60 ms |
| + bge-large embeddings | +1 core | +500MB | +200MB VRAM | +5 ms |

## Monitoring

### Prometheus Metrics

```python
from director_ai.core.metrics import metrics

# Built-in metrics exposed at /metrics
print(metrics.prometheus_format())
```

Available metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `director_ai_reviews_total` | counter | Total review requests |
| `director_ai_reviews_approved` | counter | Approved reviews |
| `director_ai_reviews_rejected` | counter | Rejected reviews |
| `director_ai_halts_total` | counter | Safety kernel halts (by reason) |
| `director_ai_coherence_score` | histogram | Score distribution |
| `director_ai_review_duration_seconds` | histogram | Review latency |
| `director_ai_active_requests` | gauge | In-flight requests |

### Health Check

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "nli_loaded": scorer._nli and scorer._nli.model_available,
        "cache_hit_rate": scorer.cache.hit_rate if scorer.cache else None,
        "cache_size": scorer.cache.size if scorer.cache else 0,
    }
```

## Security

- Store API keys in environment variables, not config files
- Use `DirectorConfig._REDACTED_FIELDS` for safe serialization
- Enable `InputSanitizer` to filter prompt injection attempts
- Audit all rejections via `AuditLogger`
