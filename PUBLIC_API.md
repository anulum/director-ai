# Public API — Director-AI v2.8.0

Frozen API surface. Breaking changes to items listed here require a major version bump.

## Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `CoherenceScorer` | `core.scorer` | Dual-entropy coherence scorer |
| `CoherenceAgent` | `core.agent` | End-to-end orchestrator (generator → scorer → kernel) |
| `SafetyKernel` | `core.kernel` | Output interlock with hard limit |
| `StreamingKernel` | `core.streaming` | Token-by-token oversight with halt |
| `AsyncStreamingKernel` | `core.async_streaming` | Async version of StreamingKernel |
| `NLIScorer` | `core.nli` | NLI-based divergence scorer |
| `ShardedNLIScorer` | `core.sharded_nli` | Multi-GPU round-robin NLI |
| `LiteScorer` | `core.lite_scorer` | Lightweight heuristic scorer |
| `GroundTruthStore` | `core.knowledge` | In-memory RAG store |
| `VectorGroundTruthStore` | `core.vector_store` | Vector DB RAG store |
| `ScoreCache` | `core.cache` | LRU score cache |
| `InputSanitizer` | `core.sanitizer` | Prompt injection detection |
| `ConversationSession` | `core.session` | Multi-turn conversation tracker |
| `DirectorConfig` | `core.config` | Configuration dataclass |

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
| `guard()` | `integrations.sdk_guard` | One-liner SDK interceptor (duck-type detection) |
| `get_score()` | `integrations.sdk_guard` | Retrieve last score from context |
| `create_app()` | `server` | Create FastAPI app |
| `create_grpc_server()` | `grpc_server` | Create gRPC server (optional TLS) |

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

## Removed Aliases (v3.0.0)

All 1.x deprecated aliases were removed in v3.0.0. See CHANGELOG for migration.

## Configuration Profiles

`DirectorConfig.from_profile(name)`:

| Profile | Threshold | NLI | Backend | Use Case |
|---------|-----------|-----|---------|----------|
| `fast` | 0.5 | off | deberta | Low-latency, non-critical |
| `thorough` | 0.6 | on | hybrid | NLI + LLM judge |
| `research` | 0.7 | on | hybrid | Full scoring |
| `medical` | 0.75 | on | hybrid | Healthcare |
| `finance` | 0.7 | on | hybrid | Financial services |
| `legal` | 0.68 | on | hybrid | Legal documents |
| `creative` | 0.4 | off | deberta | Creative writing |
| `customer_support` | 0.55 | off | deberta | Support agents |
| `summarization` | 0.55 | on | hybrid | Document summarization |
| `lite` | 0.5 | off | lite | Zero-dependency fast path |
