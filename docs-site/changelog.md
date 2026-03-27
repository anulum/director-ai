# Changelog

See the full changelog in [CHANGELOG.md on GitHub](https://github.com/anulum/director-ai/blob/main/CHANGELOG.md).

## v3.11.0 (2026-03-27)

### Added
- **ProductionGuard**: batteries-included API (calibration + feedback loop + conformal CIs + agent tool verification)
- **Atomic claim verification**: `VerifiedScorer.verify(atomic=True)` with multi-span evidence attribution
- **VectorGroundTruthStore.grounded()**: hybrid BM25 + dense retrieval factory
- **Semantic Kernel + DSPy/Instructor integrations**
- **Helm chart**: `deploy/helm/director-ai/` with GPU toggle, HPA, Sigstore signing
- **Observability pack**: Grafana dashboard (9 panels) + Prometheus alerts (6 rules)
- Latency matrix benchmark, superiority demo

### Fixed
- Streaming false-halt benchmark: 3 bugs fixed, measured 4.4% (was broken 0.0%)
- Domain profile claims: medical/finance FPR=100% without KB grounding now documented
- Summarization auto-routing to bidirectional NLI
- Sigstore signing + SLSA provenance on publish workflow

## v3.10.1 (2026-03-25)

### Added
- **Meta-confidence scoring**: `verdict_confidence`, `signal_agreement` on every `CoherenceScore`
- **Cross-turn contradiction tracking**: pairwise NLI across conversation turns, `contradiction_index`
- **Structured output verification**: `verify_json()`, `verify_tool_call()`, `verify_code()` — stdlib only
- **Online calibration**: `FeedbackStore` + `OnlineCalibrator` for deployment-specific threshold tuning with confidence intervals
- 100 new tests, 17 new source files, zero new dependencies

## v3.9.4 (2026-03-20)

### Fixed
- Domain profile thresholds: medical 0.75→0.30, finance 0.70→0.30, legal 0.68→0.30 (measured on PubMedQA and FinanceBench).
- Score calibration: rescales [0.25, 0.55] → [0, 1] when NLI available but no KB loaded.
- README claims verified: model attribution, hardware context, FPR 2.0%→10.5%.
- Docker: removed dead registry links, GPU Dockerfile fixed (`optimum` dep).
- Code scanning: blake2b for API key hash, pip deps pinned with hashes.
- All notebooks fixed: correct API signatures, field access patterns, thresholds.
- `__init__.py`: all PUBLIC_API.md symbols now importable from top-level.

## v3.9.3 (2026-03-19)

### Fixed
- Rust scorer word-overlap heuristic tests.
- Rust FFI borrow lifetime fix.
- License tests use env-var signing key consistently.

## v3.9.2 (2026-03-19)

### Fixed
- License validation hardened (UUID, HMAC, expiry).
- Cache scope isolation prevents cross-session replay.
- Tenant-aware retrieval consistency across all vector store methods.
- Batch processor catches per-item exceptions gracefully.
- Config `from_profile()` re-applies values after `__post_init__` override.
- Redis enterprise store: tenant-prefixed keys, TTL, connection handling.
- Fine-tuning lazy imports for torch/transformers.

### Added
- 353 new tests — coverage from 81% to 90%+.

## v3.9.1 (2026-03-19)

### Fixed
- Cross-tenant cache replay: cache key includes `tenant_id`.
- Batch/single scoring parity: `review_batch()` routes through `review()`.
- Vector fallback cross-tenant leak: `add_fact()` prefixes tenant_id.
- streaming_oversight crash: `ingest_token()` → `check_halt()`.
- Timeout kills stream: all streaming paths catch `TimeoutError`.

## v3.9.0 (2026-03-15)

### Added
- **VerifiedScorer**: sentence-level multi-signal fact verification. 5 independent signals: NLI, entity consistency, numerical consistency, negation detection, traceability (fabrication). `POST /v1/verify` endpoint.
- **Document ingestion API**: `POST /v1/knowledge/upload` (PDF/DOCX/HTML), `/ingest`, `/search`, CRUD by doc ID. Chunker, parser, registry modules.
- **Mode selector**: `--mode general|grounded|auto`. Single field replaces 8+ manual config settings.
- **Dataset-type classifier**: `DatasetTypeClassifier` predicts per-input NLI threshold from text features.
- **ColBERT backend**: late-interaction retrieval via RAGatouille for higher retrieval recall.
- **Domain embedding tuner**: `POST /v1/knowledge/tune-embeddings` fine-tunes embeddings on ingested documents.
- **Calibrated abstention**: returns neutral when retrieval confidence is below threshold.
- **Readiness probe**: `GET /v1/ready` returns 503 when scorer/NLI not loaded.
- `HaltMonitor` class (renamed from `SafetyKernel`, alias kept).
- LLM judge confidence now scales blend weight.
- `clear_model_cache()` for explicit GPU memory release.

### Defaults Changed
- Hybrid retrieval (BM25 + dense + RRF) enabled by default.
- Cross-encoder reranker enabled by default.
- RAG claim decomposition enabled by default for all grounded scoring.

### Infrastructure
- 180 files updated to canonical SPDX 5-line headers.
- Connection pooling on PostgreSQL audit sink.
- Redis-backed rate limiting when redis_url configured.
- Latency gate tightened to 5ms avg / 15ms p95.
- NLI pipeline consistency gate in regression suite.

## v3.8.0 (2026-03-14)

### Added
- `score()` one-call convenience function.
- `DirectorGuard` FastAPI middleware.
- `create_proxy_app()` OpenAI-compatible guardrail proxy.
- Config profiles: medical, finance, legal, creative, customer_support, summarization, lite.
- `from_env()` environment variable configuration.

### Security
- Input sanitizer: prompt injection, unicode escape, YAML injection detection.
- PII redactor for privacy mode.
- Constant-time API key comparison via HMAC.

## v3.7.0 (2026-03-10)

### Added
- **Sentence-level attribution**: `ClaimAttribution` dataclass maps each summary claim to the source sentence with lowest divergence. Available in `ScoringEvidence.attributions` and the `/v1/review` API response.
- **Cost transparency**: `ScoringEvidence.token_count` and `estimated_cost_usd` track NLI token consumption per check.
- **Domain benchmarks**: `medical_eval.py` (MedNLI + PubMedQA), `legal_eval.py` (ContractNLI + CUAD/RAGBench), `finance_eval.py` (FinanceBench + Financial PhraseBank).
- **Fine-tuning pipeline**: `finetune_nli()`, `FinetuneConfig`, `FinetuneResult`. CLI: `director-ai finetune train.jsonl`. Install: `pip install director-ai[finetune]`.
- **Load testing benchmark**: concurrent RPS measurement with P50/P95/P99 latency.
- `export_tensorrt()` — pre-builds TRT engine cache from ONNX model.
- CLI `director-ai export` subcommand (`--format onnx|tensorrt`).

### Performance
- ONNX CUDA: 4.5ms/pair median (2.4x vs PyTorch 10.9ms), L4 GPU.
- ONNX FP16: 4.2ms/pair. ONNX CPU: 4.1ms/pair (competitive at batch=4).

## v3.6.0 (2026-03-10)

### Fixed
- **Summarization FPR reduced from 10.5% to 2.0%** — Layer C (claim decomposition + coverage scoring) decomposes summaries into atomic claims, scores each against source via NLI, computes coverage. Blended with Layer A: `final = 0.4 * (1 - coverage) + 0.6 * layer_a`. All three task types now below 5% FPR.

### Added
- `NLIScorer.score_claim_coverage()` method
- Config: `nli_claim_coverage_enabled`, `nli_claim_support_threshold` (0.6), `nli_claim_coverage_alpha` (0.4)
- `ScoringEvidence` includes `claim_coverage`, `per_claim_divergences`, `claims`
- 21 new tests (2084 total)
- Claim coverage FPR diagnostic benchmark

## v3.5.0 (2026-03-10)

### Fixed
- **Summarization FPR reduced from 25.5% to 10.5%** — bidirectional NLI scores both source→summary and summary→source, takes min. Baseline calibration (0.20) shifts expected NLI noise to zero.
- **Dialogue FPR reduced from 97.5% to 4.5%** — bidirectional NLI + baseline=0.80.

### Added
- `_summarization_factual_divergence()` method with bidirectional NLI scoring
- `nli_summarization_baseline` config field (default 0.20)
- `_detect_task_type()` static method for dialogue vs summarization routing
- 13 new tests in `tests/test_summarization_bidir.py`
- Bidirectional FPR diagnostic benchmark with 6 baseline profiles

## v3.4.0 (2026-03-09)

### Fixed
- **Summarization FPR reduced from 95% to 25.5%** — three-phase fix: MIN inner aggregation, direct NLI scoring (bypass vector store), w_logic=0 (eliminate h_logic==h_fact duplication), trimmed_mean outer aggregation.

### Added
- `trimmed_mean` outer aggregation for chunked NLI scoring
- `_use_prompt_as_premise` flag — direct document→summary NLI scoring
- Configurable `nli_fact_retrieval_top_k` and `nli_use_prompt_as_premise` config fields
- Summarization FPR diagnostic benchmark (`benchmarks/summarization_fpr_diag.py`)
- 5 new tests, `TestWLogicZeroShortCircuit` test class

### Changed
- Summarization profile: `w_logic=0.0, w_fact=1.0`, `coherence_threshold=0.15`, `nli_fact_outer_agg="trimmed_mean"`, `nli_use_prompt_as_premise=True`
- `_heuristic_coherence` short-circuits logical divergence when `W_LOGIC < 1e-9`

## v3.3.0 (2026-03-07)

### Added
- Generated gRPC protobuf stubs from `proto/director.proto`
- `CoherenceAgent.aprocess()` and `CoherenceAgent.astream()` async methods
- `CoherenceScorer.review_batch()` — coalesced batch NLI (2 GPU calls when NLI available)
- `ReviewQueue` — server-level continuous batching with configurable flush window
- `--cors-origins` flag on `director-ai serve`

### Changed
- `cors_origins` default changed from `"*"` to `""` (no CORS by default)
- H_logical and H_factual computed in parallel via `ThreadPoolExecutor` (~40% latency reduction)

## v3.2.0 (2026-03-07)

### Added
- `BatchProcessor.process_batch_async()` and `review_batch_async()` — async batch processing
- `__aiter__` on Bedrock, Gemini, Cohere guarded streams (parity with OpenAI/Anthropic)
- `VectorBackend.aadd()` / `aquery()` async defaults via `run_in_executor`
- `LiteScorer.review()` returning `(bool, CoherenceScore)` matching `CoherenceScorer` interface
- Config validation: `reranker_model` / `embedding_model` non-empty when feature enabled

## v3.1.0 (2026-03-07)

### Added
- Hybrid scorer hardening: NLI confidence margin fix, LLM judge verdict caching, retry with back-off
- Enterprise modules: `PostgresAuditSink`, `RedisGroundTruthStore`
- WASM edge runtime: CI pipeline, browser integration tutorial, overhead benchmark
- Rust backend: PyO3 0.24 upgrade, SIMD micro-cycle vectorization
- Vector backends: FAISS (dense search), Elasticsearch (hybrid BM25 + dense)
- RAGTruth + FreshQA GPU benchmark, cross-platform latency profiling

## v3.0.0 (2026-03-07)

### Breaking Changes
- Minimum Python 3.11 (dropped 3.10)
- Enterprise classes moved: `TenantRouter`, `Policy`, `AuditLogger` → `director_ai.enterprise`
- Removed deprecated 1.x aliases (`calculate_factual_entropy`, `review_action`, etc.)
- Slimmed root `__all__`: internal classes removed from public API surface

### Added
- `director_ai.enterprise` package re-exporting all 5 enterprise classes
- `director-ai tune` adaptive threshold calibration
- Python 3.13 in CI matrix

## v2.0.0 (2026-03-02)

### Fixed
- Case-sensitivity bug in `GroundTruthStore.retrieve_context()` — mixed-case keys now match
- LLM judge error handling: bare `except Exception` replaced with structured try/except
- `SafetyKernel` validates `hard_limit` in [0, 1] range
- OTel `setup_otel()` is now thread-safe
- `case-studies.md` code snippets corrected (wrong constructors, phantom methods)

### Added
- Named constants for LLM judge blending formula
- `.editorconfig`, `.pre-commit-config.yaml`, `py.typed` PEP 561 marker
- Documentation URL in `pyproject.toml`
- Non-root user in Dockerfiles
- Histogram `bucket_counts()` O(n log n) optimization
- New tests: knowledge, kernel validation, ingest, cache
- 12 `inspect.getsource` fragile tests replaced with behavioral equivalents

## v1.9.0 (2026-03-02)

### Added
- Soft-halt mode: `StreamingKernel(halt_mode="soft")`
- JSON structured logging: `log_json=True`
- OpenTelemetry integration: `otel_enabled=True`
- Request ID propagation: `X-Request-ID` header
- 100-passage false-halt benchmark
- Coverage threshold raised to 80%

## v1.7.0 (2026-03-01)

### Added
- Domain presets: `DirectorConfig.from_profile()`
- Structured halt evidence
- Pluggable scorer backend: `deberta`, `onnx`, `minicheck`
- Batched MiniCheck support
- False-halt assertion in CI

## v1.6.0 (2026-03-01)

### Added
- API key auth, correlation IDs, audit logging
- Tenant routing, rate limiting
- Streaming WebSocket oversight
- E2E benchmarks (300 traces)
- 835 tests

## v1.5.0 — v1.5.1 (2026-03-01)

### Added
- Bidirectional chunked NLI
- Prometheus metric compliance
- Real streaming halt with evidence
- RAG retrieval bench

## v1.4.0 — v1.4.1 (2026-03-01)

### Added
- Batched NLI inference (10.8x speedup)
- ONNX export + runtime (14.6 ms/pair GPU)
- GPU Docker image, TensorRT provider

## v1.3.0 (2026-03-01)

- Default NLI model: FactCG-DeBERTa-v3-Large (75.8% balanced accuracy)

## v1.2.0 — v1.2.1 (2026-02-27)

- Score caching, LangGraph/Haystack/CrewAI integrations
- MkDocs documentation, strict_mode, configurable weights

## v1.1.0 (2026-02-27)

- SDK Guard `guard()` for OpenAI/Anthropic
- Streaming guards, `HallucinationError`

## v1.0.0 (2026-02-26)

- Production stable release
- Enterprise modules: Policy, AuditLogger, TenantRouter, InputSanitizer
- LangChain + LlamaIndex integrations
