# Roadmap

## v2.8.0

### Done
- Rust-accelerated scorer backend wired into `CoherenceScorer(scorer_backend="rust")`
- WebSocket multiplexed streaming (concurrent sessions per connection, cancel, backpressure)
- VectorBackend entry-point registry (`register_vector_backend`, `get_vector_backend`, `list_vector_backends`)
- Tenant-isolated VectorStores via `TenantRouter.get_vector_store()`
- `/v1/tenants/{tenant_id}/vector-facts` REST endpoint
- ONNX INT8/FP16 quantization shipped in v2.3.0 (retired from planned)

## v2.6.0

### Done
- StreamingKernel wired into `CoherenceAgent.stream()` for unified token-level oversight
- gRPC incremental streaming with per-chunk coherence scores
- CLI multi-worker config propagation (`--workers`)
- ONNX batch config wiring end-to-end (`onnx_path` in DirectorConfig)
- Prompt content removed from logs — HMAC audit hashing
- `guard()` duck-type detection for OpenAI/Anthropic SDK interceptor
- `strict_mode` reject in `CoherenceScorer.review()`
- Domain profiles: medical, finance, legal, creative, customer_support, summarization
- Discord bot + CI/release webhook automation
- E2E benchmark context leakage fix (per-sample store isolation)
- InputSanitizer additive scoring model fix

## v2.5.0

### Done
- Domain-specific scoring profiles (medical, finance, legal, creative, customer_support, summarization)
- `strict_mode` reject in CoherenceScorer
- `guard()` SDK interceptor with duck-type provider detection
- Persistent stats backend (SQLite)
- AGPL §13 `/v1/source` compliance endpoint

## v2.3.0

### Done
- Lite scorer backend (`scorer_backend="lite"`) — word overlap + negation heuristics, ~0.5ms/pair
- Multi-turn conversation tracking (`ConversationSession`) with cross-turn divergence blending
- ONNX GPU batch optimization (`OnnxDynamicBatcher`) with IO binding for zero-copy transfers
- Plugin architecture for scorer backends (`ScorerBackend` ABC + entry-point registry)
- gRPC transport (`proto/director.proto`, `--transport grpc` on CLI)
- Multi-GPU sharding (`ShardedNLIScorer`) with round-robin device routing
- Security audit preparation: threat model, SBOM generation, Hypothesis fuzz tests, `InputSanitizer` hardening
- Public API freeze: `__all__` on all modules, deprecated aliases emit `DeprecationWarning`

## v2.2.1

### Done
- API autodoc pages for DirectorConfig, Enterprise, InputSanitizer
- Troubleshooting guide, enterprise guide, streaming cadence examples
- Validation rules section in scorer reference

## v2.2.0

### Done
- `score_every_n`, `adaptive`, `max_cadence` on StreamingKernel + AsyncStreamingKernel
- Runtime validation on threshold, soft_limit, w_logic, w_fact
- Streaming overhead benchmark (tokens/sec by cadence)
- Enterprise modules lazy-loaded via `__getattr__`
- `[enterprise]` optional dependency group + pytest marker

## v2.1.0

### Done
- `director-ai bench` CLI subcommand (--dataset, --seed, --output)
- `scorer_backend="hybrid"` mode (NLI + LLM judge)
- Architecture deep-dive, production checklist, threshold tuning docs
- PineconeBackend, WeaviateBackend, QdrantBackend
- Bandit + Semgrep SAST in CI

## v2.0.0

### Done
- Case-sensitivity fix in GroundTruthStore
- LLM judge error handling hardened
- SafetyKernel hard_limit validation
- Thread-safe OTel setup
- Histogram bucket_counts O(n log n) optimization

## v3.0.0

### Done
- **Simplified public API**: `guard()` as the primary interface; enterprise behind `director_ai.enterprise`
- **Adaptive threshold calibration**: `director-ai tune` with labeled data → optimal threshold + weights
- **Remove deprecated 1.x aliases**: all 6 deprecated methods removed; 1.x class name aliases already removed in 2.x
- **Drop Python 3.10**: minimum Python 3.11 for `ExceptionGroup` and `TaskGroup` support

## v3.1.0

### Hybrid Scorer Hardening (Done)
- Fix NLI confidence margin calculation — `nli_margin` never computed, hybrid escalation broken
- LLM judge verdict caching (LRU keyed on prompt+response hash) to avoid redundant API calls
- Retry with exponential back-off on transient LLM API failures
- Escalation-rate telemetry via `metrics.counter("llm_judge_escalations")`
- Run hybrid-mode E2E benchmark on HaluEval (300 traces) and publish numbers

### Enterprise Module Completion (Done)
- `PostgresAuditSink.log()` implementation with async connection pooling (`asyncpg`)
- Schema migration framework (version-tracked DDL with forward-only migrations)
- `RedisGroundTruthStore.retrieve_context()` implementation with Redis Vector Search (RediSearch)
- Redis connection pooling, TTL management, batch `add_many()`/`retrieve_batch()`

### WASM Edge Runtime (Deferred)
- CI pipeline: `wasm-pack build` in wheels.yml, publish `.wasm` + JS glue to npm
- Browser integration tutorial (vanilla JS + webpack example)
- Benchmark script exists (`benchmarks/wasm_overhead_bench.py`) but no production WASM code ships

### Rust Backend (Done)
- PyO3 0.23 → 0.24 upgrade (unblocks Python 3.14 wheels)
- SIMD vectorization of backfire-ssgf micro-cycle inner loop (`std::simd` or `packed_simd2`)

### Benchmarks (Done)
- Run RAGTruth + FreshQA full-scale GPU benchmark, publish results in BENCHMARK_REPORT.md
- Cross-platform latency profiling (Windows/macOS/Linux) with memory + GC overhead
- Quantify PyO3 FFI overhead (Rust-native vs Python-via-FFI round-trip)

### Vector Backends (Done)
- FAISS backend (in-process dense search for edge/offline deployments)
- Elasticsearch backend (hybrid BM25 + dense retrieval)

## v3.2.0

### Bug Fixes (Done)
- Fix `quickstart` CLI scaffolding broken `asyncio.run()` on sync methods
- Implement `BatchProcessor.process_batch_async()` — docstring-advertised method missing

### Async Correctness (Done)
- Add `__aiter__` to Bedrock/Gemini/Cohere guarded stream wrappers (parity with OpenAI/Anthropic)
- Add `async aadd()`/`aquery()` defaults on `VectorBackend` ABC for non-blocking server use
- Parallelize `AnthropicProvider`/`HuggingFaceProvider` multi-candidate requests

### API Consistency (Done)
- Add `LiteScorer.review()` returning `(bool, CoherenceScore)` to match `CoherenceScorer` interface

### Configuration Hardening (Done)
- Validate `reranker_model`/`embedding_model` non-empty when feature enabled
- Warn on unknown YAML keys in `DirectorConfig.from_yaml()`

### Test Coverage (Done)
- End-to-end `scorer.review(session=...)` cross-turn divergence test
- `review_batch()` ordering, partial failure, and timeout tests
- `build_store()` with `vector_backend="sentence-transformer"` branch test

## v3.3.0

### Done
- Version bump to 3.3.0 in pyproject.toml, `__init__.py`, CITATION.cff
- CHANGELOG.md entries for v3.1.0, v3.2.0, v3.3.0
- Deprecated 1.x alias table removed from PUBLIC_API.md
- Generated `director_pb2.py` / `director_pb2_grpc.py` from proto/director.proto
- Removed SimpleNamespace fallback; fail-fast if protobuf stubs missing
- `CoherenceAgent.aprocess()` async counterpart
- CLI `--chunk-size` validation (reject <= 0)
- `cors_origins` default changed from `"*"` to `""` (require explicit config)
- `--cors-origins` flag on `director-ai serve`
- 8 new tests in test_v330_hardening.py (1927 total, 0 failures)

### Performance Sprint (Done)
- H_logical and H_factual parallelised via `ThreadPoolExecutor` (~40% latency reduction)
- `CoherenceScorer.review_batch()` — coalesced batch NLI (2 GPU calls when NLI available)
- `BatchProcessor.review_batch()` delegates to scorer with serial fallback
- `ReviewQueue` — server-level continuous batching for `/v1/review` with flush window
- Config fields: `review_queue_enabled`, `review_queue_max_batch`, `review_queue_flush_timeout_ms`
- TensorRT path verified deployment-ready (no code changes needed)
- Async hygiene: 5 sync→async fixes in server.py, sessions lock, OTel lazy init
- 1966 tests, 0 failures

## v3.4.0

### Done
- Local DeBERTa-v3-base binary judge replaces LLM judge for borderline NLI escalation (F1=0.915, latency ~15ms vs 1.3–14.2s, zero API cost)
- Summarization FPR reduced from 95% to 25.5% (three-phase fix):
  - Phase 1: MIN inner aggregation (95% → 60%)
  - Phase 2: `premise_ratio=0.85` + logic aggregation bug fix (60% → 42.5%)
  - Phase 3: `w_logic=0` (eliminate h_logic==h_fact duplication), `_use_prompt_as_premise=True` (bypass lossy vector store), `trimmed_mean` outer aggregation (42.5% → 25.5%)
- Summarization profile: `w_logic=0.0, w_fact=1.0`, `coherence_threshold=0.15`, `nli_fact_outer_agg="trimmed_mean"`, `nli_use_prompt_as_premise=True`
- `_heuristic_coherence` short-circuits logical divergence when `W_LOGIC < 1e-9`
- Configurable `nli_fact_retrieval_top_k` and `nli_use_prompt_as_premise` config fields
- Summarization FPR diagnostic benchmark (`benchmarks/summarization_fpr_diag.py`)
- `workflow_dispatch` added to `publish.yml` and `docker.yml` (fix GITHUB_TOKEN anti-loop)
- 2038 tests, 0 failures
- Dialogue FPR: 97.5% → 4.5% via bidirectional NLI + baseline calibration
  - `_detect_task_type()` classifies dialogue via speaker-turn regex
  - `_dialogue_factual_divergence()` scores both directions, applies baseline calibration
  - Logical divergence skipped for dialogue (entailment is meaningless)
  - Diagnostic benchmark: `benchmarks/dialogue_fpr_diag.py` (4 baseline configs)

## v3.5.0

### Done
- Summarization FPR: 25.5% → 10.5% via bidirectional NLI + baseline=0.20
  - `_summarization_factual_divergence()` scores source→summary and summary→source, takes min
  - Baseline calibration: `adjusted = max(0, (raw - 0.20) / 0.80)`
  - `nli_summarization_baseline` config field (default 0.20)
  - Bidirectional FPR diagnostic benchmark (`benchmarks/summarization_fpr_diag.py`)
  - 13 new tests (`tests/test_summarization_bidir.py`)
  - 2065 tests, 0 failures

## v3.6.0

### Done
- Summarization FPR: 10.5% → 2.0% via Layer C (claim decomposition + coverage scoring)
  - `NLIScorer.score_claim_coverage()` decomposes summaries into atomic claims
  - Config: `nli_claim_coverage_enabled`, `nli_claim_support_threshold` (0.6), `nli_claim_coverage_alpha` (0.4)
  - `ScoringEvidence` includes `claim_coverage`, `per_claim_divergences`, `claims`
  - 21 new tests, 2084 total

## v3.7.0

### Done
- Sentence-level attribution: `ClaimAttribution` maps claims to source sentences
- Cost transparency: `ScoringEvidence.token_count`, `estimated_cost_usd`
- Domain benchmarks: medical_eval (MedNLI + PubMedQA), legal_eval (ContractNLI + CUAD), finance_eval (FinanceBench)
- Fine-tuning pipeline: `finetune_nli()`, `FinetuneConfig`, `FinetuneResult`, CLI `director-ai finetune`
- TensorRT export: `export_tensorrt()`, CLI `director-ai export --format tensorrt`
- ONNX CUDA: 4.5ms/pair median (2.4x faster than PyTorch)

### Deferred
- Distill smaller NLI model (DeBERTa-base from FactCG-Large teacher + hybrid labels) — deferred, 22/23 fine-tunes hurt (catastrophic forgetting)
- ReviewQueue adaptive flushing (dynamic max_batch based on request rate) — deferred to v3.10+

## v3.8.0

### Done
- Per-task-type adaptive thresholds (+0.86pp balanced accuracy over global baseline)
- AggreFact evaluation pipeline with cached scoring + per-dataset sweep
- GPU benchmarking infrastructure (UpCloud L40S)
- 32 fine-tuning experiments (LoRA, distillation) — all negative, documented

## v3.9.0

### Done
- Dataset-type classifier (RF-20-d6, 77.08% BA, +1.22pp over global)
- Classifier auto-discovery from bundled model when `adaptive_threshold_enabled=True`
- Threshold hierarchy: global → per-task-type → per-dataset (most specific wins)
- Security: proxy HMAC auth, HTTPS enforcement, tenant-to-API-key binding, session ownership, WebSocket info leak fix
- Documentation: why-director-ai, migration-v2-v3, glossary, runbooks
- Secret removal from git history (expired HF token)

## v3.10.0

### Planned
- **B615 HF revision pinning**: pin `revision="<sha>"` on all 11 `from_pretrained()` calls to prevent supply-chain risk
- SPDX `AGPL-3.0-or-later` headers on all source files
- ModernBERT-large (8192 tokens) as alternate NLI backend — only path to >78% BA
- Stripe Checkout page + HMAC-SHA256 license key generation
- B608 ChromaDB parameterised filters (low priority, no user input reaches these)
