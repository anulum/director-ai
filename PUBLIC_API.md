# Public API — Director-AI v3.9.4

Frozen API surface. Breaking changes to items listed here require a major version bump.

## Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `CoherenceScorer` | `core.scorer` | Weighted NLI divergence scorer |
| `CoherenceAgent` | `core.agent` | End-to-end orchestrator (generator → scorer → kernel) |
| `HaltMonitor` | `core.kernel` | Threshold-based halt gate for output stream |
| `SafetyKernel` | `core.kernel` | Alias for `HaltMonitor` (backward compat) |
| `StreamingKernel` | `core.streaming` | Token-by-token oversight with halt |
| `AsyncStreamingKernel` | `core.async_streaming` | Async version of StreamingKernel |
| `NLIScorer` | `core.nli` | NLI-based divergence scorer |
| `ShardedNLIScorer` | `core.sharded_nli` | Multi-GPU round-robin NLI |
| `LiteScorer` | `core.lite_scorer` | Lightweight heuristic scorer |
| `GroundTruthStore` | `core.knowledge` | In-memory keyword fact store |
| `VectorGroundTruthStore` | `core.vector_store` | Vector DB RAG store |
| `ScoreCache` | `core.cache` | LRU score cache |
| `InputSanitizer` | `core.sanitizer` | Prompt injection detection |
| `ConversationSession` | `core.session` | Multi-turn conversation tracker |
| `DirectorConfig` | `core.config` | Configuration dataclass |
| `VerifiedScorer` | `core.verified_scorer` | Sentence-level multi-signal fact verifier |
| `DatasetTypeClassifier` | `core.meta_classifier` | Adaptive threshold via dataset-type prediction |
| `MetaClassifier` | `core.meta_classifier` | Alias for `DatasetTypeClassifier` (backward compat) |

## Verified Scoring Types

| Type | Module | Description |
|------|--------|-------------|
| `ClaimVerdict` | `core.verified_scorer` | Per-claim verdict with source citation |
| `VerificationResult` | `core.verified_scorer` | Overall verification with per-claim breakdown |

## Data Types

| Type | Module | Description |
|------|--------|-------------|
| `CoherenceScore` | `core.types` | Score result with h_logical, h_factual, evidence |
| `ReviewResult` | `core.types` | Full review result (output, coherence, halted) |
| `HaltEvidence` | `core.types` | Halt reason with evidence chunks |
| `EvidenceChunk` | `core.types` | Single retrieval chunk |
| `ScoringEvidence` | `core.types` | NLI scoring evidence |
| `TokenEvent` | `core.streaming` | Single token event from streaming |
| `StreamSession` | `core.streaming` | Streaming session metrics |
| `SanitizeResult` | `core.sanitizer` | Sanitizer check result |
| `Turn` | `core.session` | Single conversation turn |

### Retrieval Strategy

`CoherenceScorer.review(prompt, response)` retrieves KB context using the **prompt** only (not the response claims). This is by design: retrieval must happen before scoring to provide the fact basis against which claims are evaluated. Searching by response claims would create a circular dependency — you can't retrieve evidence for claims you haven't scored yet. For vague prompts, the heuristic and NLI scoring layers compensate by comparing the response against whatever context the prompt retrieves.

## Plugin API

| Symbol | Module | Description |
|--------|--------|-------------|
| `ScorerBackend` | `core.backends` | ABC for custom scorer backends |
| `register_backend()` | `core.backends` | Register a custom backend by name |
| `get_backend()` | `core.backends` | Retrieve a backend class by name |
| `list_backends()` | `core.backends` | List all registered backend names |

## Vector Store Backends

| Class | Module | Description |
|-------|--------|-------------|
| `VectorBackend` | `core.vector_store` | ABC for vector backends |
| `InMemoryBackend` | `core.vector_store` | Word-overlap proxy (testing) |
| `ChromaBackend` | `core.vector_store` | ChromaDB production backend |
| `SentenceTransformerBackend` | `core.vector_store` | bge-large-en-v1.5 embeddings |
| `RerankedBackend` | `core.vector_store` | Reranking wrapper |
| `PineconeBackend` | `core.vector_store` | Pinecone backend |
| `WeaviateBackend` | `core.vector_store` | Weaviate backend |
| `QdrantBackend` | `core.vector_store` | Qdrant backend |
| `FAISSBackend` | `core.vector_store` | FAISS backend |
| `ElasticsearchBackend` | `core.vector_store` | Elasticsearch backend |
| `HybridBackend` | `core.vector_store` | BM25 + dense with Reciprocal Rank Fusion |
| `ColBERTBackend` | `core.vector_store` | ColBERT v2 late-interaction retrieval |

## Vector Backend Plugin API

| Symbol | Module | Description |
|--------|--------|-------------|
| `register_vector_backend()` | `core.vector_store` | Register a custom vector backend by name |
| `get_vector_backend()` | `core.vector_store` | Retrieve a vector backend class by name |
| `list_vector_backends()` | `core.vector_store` | List all registered vector backend names |

## Generators

| Class | Module | Description |
|-------|--------|-------------|
| `MockGenerator` | `core.actor` | Deterministic mock candidates |
| `LLMGenerator` | `core.actor` | Real LLM candidate generation |

## Enterprise (Lazy-Loaded)

| Class | Module | Description |
|-------|--------|-------------|
| `TenantRouter` | `core.tenant` | Per-tenant config resolution |
| `Policy` | `core.policy` | Policy rule engine |
| `AuditLogger` | `core.audit` | Structured audit logging |

## Batch Processing

| Class | Module | Description |
|-------|--------|-------------|
| `BatchProcessor` | `core.batch` | Thread-pool batch processing wrapper |
| `BatchResult` | `core.batch` | Batch operation result |
| `ReviewQueue` | `core.review_queue` | Server-level continuous batching accumulator |

### CoherenceScorer Batch Methods

| Method | Description |
|--------|-------------|
| `review(prompt, response)` | Single review → `(bool, CoherenceScore)` |
| `areview(prompt, response)` | Async single review (thread pool offload) |
| `review_batch(items)` | Batch review — routes each item through `review()` for full scoring parity → `list[(bool, CoherenceScore)]` |

### ReviewQueue (Continuous Batching)

Server-level request accumulator. Collects incoming `/v1/review` requests and
flushes them as a single `review_batch()` call per tenant. Session-bound requests
bypass the queue.

| Config Field | Type | Default | Description |
|-------------|------|---------|-------------|
| `review_queue_enabled` | `bool` | `False` | Enable continuous batching queue |
| `review_queue_max_batch` | `int` | `32` | Flush after N requests accumulate |
| `review_queue_flush_timeout_ms` | `float` | `10.0` | Flush after N ms (whichever first) |

## Fine-Tuning

| Symbol | Module | Description |
|--------|--------|-------------|
| `FinetuneConfig` | `core.finetune` | Training configuration dataclass |
| `FinetuneResult` | `core.finetune` | Training result (metrics, output path) |
| `finetune_nli()` | `core.finetune` | Fine-tune NLI model on domain data |
| `validate_finetune_data()` | `core.finetune_validator` | Validate JSONL training data |
| `DataQualityReport` | `core.finetune_validator` | Validation report dataclass |
| `benchmark_finetuned_model()` | `core.finetune_benchmark` | Benchmark fine-tuned vs baseline |
| `RegressionReport` | `core.finetune_benchmark` | Benchmark result dataclass |

## Threshold Tuning

| Symbol | Module | Description |
|--------|--------|-------------|
| `TuneResult` | `core.tuner` | Tuning grid search result |
| `tune()` | `core.tuner` | Threshold + weight grid search |

## Functions

| Function | Module | Description |
|----------|--------|-------------|
| `nli_available()` | `core.nli` | Check NLI readiness |
| `export_onnx()` | `core.nli` | Export model to ONNX |
| `export_tensorrt()` | `core.nli` | Export ONNX model to TensorRT |
| `guard()` | `integrations.sdk_guard` | One-liner SDK interceptor (duck-type detection) |
| `get_score()` | `integrations.sdk_guard` | Retrieve last score from context |
| `create_app()` | `server` | Create FastAPI app |
| `create_grpc_server()` | `grpc_server` | Create gRPC server (optional TLS) |
| `score()` | `integrations.sdk_guard` | One-call scoring convenience function |
| `clear_model_cache()` | `core.nli` | Evict cached NLI models to free GPU memory |
| `create_knowledge_router()` | `knowledge_api` | Knowledge ingestion API router |

## REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/review` | Score a prompt/response pair |
| `POST` | `/v1/verify` | Sentence-level multi-signal verification |
| `POST` | `/v1/process` | Full agent pipeline (generate + score) |
| `POST` | `/v1/batch` | Batch review/process |
| `GET` | `/v1/health` | Liveness probe (version, mode, uptime) |
| `GET` | `/v1/ready` | Readiness probe (503 if NLI not loaded) |
| `GET` | `/v1/source` | AGPL §13 source access |
| `GET` | `/v1/config` | Config introspection |
| `GET` | `/v1/metrics` | JSON metrics |
| `GET` | `/v1/metrics/prometheus` | Prometheus text format |
| `POST` | `/v1/knowledge/upload` | Upload file → parse → chunk → embed |
| `POST` | `/v1/knowledge/ingest` | Ingest raw text → chunk → embed |
| `GET` | `/v1/knowledge/documents` | List documents per tenant |
| `GET` | `/v1/knowledge/documents/{id}` | Document metadata |
| `DELETE` | `/v1/knowledge/documents/{id}` | Delete document and chunks |
| `PUT` | `/v1/knowledge/documents/{id}` | Re-ingest updated content |
| `GET` | `/v1/knowledge/search` | Test retrieval quality |
| `POST` | `/v1/knowledge/tune-embeddings` | Fine-tune embeddings on ingested docs |

## Exceptions

| Exception | Module |
|-----------|--------|
| `DirectorAIError` | `core.exceptions` |
| `CoherenceError` | `core.exceptions` |
| `KernelHaltError` | `core.exceptions` |
| `GeneratorError` | `core.exceptions` |
| `ValidationError` | `core.exceptions` |
| `DependencyError` | `core.exceptions` |
| `HallucinationError` | `core.exceptions` |
| `NumericalError` | `core.exceptions` |
| `PhysicsError` | `core.exceptions` |

## Removed Aliases (v3.0.0)

All 1.x deprecated aliases were removed in v3.0.0. See CHANGELOG for migration.

## Configuration Profiles

`DirectorConfig.from_profile(name)`:

| Profile | Threshold | NLI | Backend | Use Case |
|---------|-----------|-----|---------|----------|
| `fast` | 0.5 | off | deberta | Low-latency, non-critical |
| `thorough` | 0.6 | on | hybrid | NLI + LLM judge |
| `research` | 0.7 | on | hybrid | Full scoring |
| `medical` | 0.30 | on | hybrid | Healthcare (measured on PubMedQA) |
| `finance` | 0.30 | on | hybrid | Financial services (measured on FinanceBench) |
| `legal` | 0.30 | on | hybrid | Legal documents (not yet measured) |
| `creative` | 0.4 | off | deberta | Creative writing |
| `customer_support` | 0.55 | off | deberta | Support agents |
| `summarization` | 0.15 | on | hybrid | Document summarization |
| `lite` | 0.5 | off | lite | Zero-dependency fast path |
