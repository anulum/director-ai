# Changelog

See the full changelog in [CHANGELOG.md on GitHub](https://github.com/anulum/director-ai/blob/main/CHANGELOG.md).

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
