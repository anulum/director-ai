# Changelog

See the full changelog in [CHANGELOG.md on GitHub](https://github.com/anulum/director-ai/blob/main/CHANGELOG.md).

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
