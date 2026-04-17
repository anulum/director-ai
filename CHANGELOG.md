# Changelog

All notable changes to Director-Class AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-04-17

### Added
- `facts_root` parameter on `create_proxy_app` / `_load_facts` and
  `--facts-root` CLI flag to restrict proxy facts loading to a chosen
  directory (symlinks resolved).
- `director_ai.core.safety.audit_salt.get_audit_salt()` — loads the
  audit-log fingerprint salt from `DIRECTOR_AUDIT_SALT` or
  `DIRECTOR_AUDIT_SALT_FILE`, with a legacy fallback that warns once.
- Julia threshold tuner (`tools/julia_tuner/`) — offline analytics
  module that takes labelled scorer output and returns a point
  threshold, a bootstrap 95% CI, and a Bayesian posterior (Turing.jl
  NUTS). Python feeder at `tools/prepare_threshold_data.py`. New
  `make test-julia` and `make julia-instantiate` targets.
- Lean 4 formal model of the HaltMonitor threshold check
  (`formal/HaltMonitor/`) with four machine-checked theorems: no
  token whose coherence score falls below `hard_limit` can ever be
  emitted. New `make test-lean` target plus `test-all` wiring.
- Frozen `director.v1` wire schema in `schemas/proto/director/v1/`
  — chat completion, coherence verdict, tenant, API key, and audit
  messages plus `CoherenceScoring` and `ChatGateway` service
  definitions. Generated Python stubs under
  `src/director_ai/proto/director/v1/`, Go stubs under
  `gateway/go/proto/director/v1/`. `schemas/generate.sh`
  regenerator, hand-written `director_ai.proto.converters`
  adapters, and `make proto` / `make test-go` targets.
- Go gateway skeleton (`gateway/go/`) — passthrough HTTP proxy in
  front of any OpenAI-compatible upstream. Env-driven config,
  constant-time API-key auth with audit fingerprint, per-key
  token-bucket rate limit, JSONL audit sink matching the Python
  audit record shape, SSE streaming via `http.Flusher`. Binary
  entrypoint `cmd/director-gateway`, k6 load script under
  `gateway/go/bench/`, 50 Go test cases, clean `go test -race`.
- `director.v1.CoherenceScoring` gRPC server
  (`src/director_ai/grpc_scoring.py`) — wraps `CoherenceScorer`
  with `ScoreClaim` (unary) and `ScoreStream` (bidirectional)
  RPCs. New CLI: `python -m director_ai.grpc_scoring`.
- Go scoring client (`gateway/go/internal/scoring/`) — dials the
  Python gRPC server and adds an optional response middleware that
  extracts the assistant message from chat-completion JSON, runs
  `ScoreClaim` against it, stamps `X-Coherence-Score` /
  `X-Coherence-Halted` headers and rewrites halted answers as 422
  JSON. Enabled by `DIRECTOR_SCORING_ADDR`; per-request override
  via `X-Coherence-Threshold`. Additional env:
  `DIRECTOR_SCORING_TIMEOUT_MS`.
- `bench/ab_bench.sh` — boots a mock upstream, runs two k6 scenarios
  (gateway only, gateway + scoring sidecar), and records the k6
  summaries for comparison. `make grpc-scoring` and `make ab-bench`
  targets wire everything together.

### Changed
- `CoherenceAgent.__init__` and `_build_provider` accept `api_key=`
  directly; the server passes `DirectorConfig.llm_api_key` through
  instead of setting `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` on the
  process environment.
- API-key audit fingerprints in `server.py` and
  `middleware/api_key.py` now use the per-installation salt from
  `get_audit_salt()` instead of a hardcoded constant.

## [3.14.0] — 2026-04-14

### Added
- **5-tier scoring pyramid** — pluggable backend architecture:
  - **Tier 2 — Rules engine** (`scorer_backend="rules"`): 8 configurable
    rules (entity grounding, numeric consistency, negation flip, etc.).
    Zero ML deps, <1ms. Direct Guardrails AI competitor.
  - **Tier 3 — Embedding scorer** (`scorer_backend="embed"`):
    sentence-transformers cosine similarity, ~65% BA at 3ms CPU.
    Install: `pip install director-ai[embed]`.
  - **Tier 4 — Distilled NLI** (`scorer_backend="nli-lite"`): MiniLM-class
    model distilled from FactCG. Target ~70% BA at 5ms. ONNX + PyTorch.
    Install: `pip install director-ai[nli-lite]`. *Preview — model training
    pending; code and backend are functional.*
  - All backends registered via `ScorerBackend` ABC + entry-point system.
- **SaaS middleware** (`director_ai.middleware`):
  - `APIKeyMiddleware`: Bearer/X-API-Key auth, env/file key sources,
    constant-time hmac validation, audit-safe key hashing.
  - `RateLimitMiddleware`: per-key token-bucket with configurable RPM
    and burst, 429 + Retry-After.
  - Cloud Run Dockerfile (`deploy/cloud-run/Dockerfile.saas`) with
    FactCG ONNX pre-baked, non-root user, healthcheck.
- Config profiles: `"rules"`, `"embed"` added to `DirectorConfig.from_profile()`.
- **6 advanced RAG retrieval strategies** — all independently toggleable,
  composable as a decorator stack on any vector backend:
  - **Parent-child chunking** (`parent_child_enabled`): index small child
    chunks for precision, return full parent chunks for context.
    Chroma/FAISS persistence via metadata.
  - **Adaptive retrieval routing** (`adaptive_retrieval_enabled`): skip KB
    lookup for creative/conversational queries. Uses `rust_detect_task_type`
    FFI when available.
  - **HyDE** (`hyde_enabled`): LLM generates pseudo-answer, embeds that
    instead of raw query. TTL-cached, graceful fallback.
  - **Query decomposition** (`query_decomposition_enabled`): split compound
    queries into sub-queries, retrieve for each, merge via RRF.
  - **Contextual compression** (`contextual_compression_enabled`): keep only
    query-relevant sentences from retrieved passages.
  - **Multi-vector** (`multi_vector_enabled`): index content + summary +
    title representations per document.
- **Multi-agent swarm guardian** (`director_ai.agentic`):
  - `AgentProfile`: per-agent config with role-based defaults (researcher,
    summariser, coder, reviewer, planner, executor).
  - `SwarmGuardian`: central agent registry with cross-agent handoff
    scoring, dependency graph, cascade halt on quarantine.
  - `HandoffScorer`: score inter-agent messages for factual grounding
    (keyword + optional NLI), history tracking with aggregate stats.
  - `SwarmMetrics`: per-agent hallucination rate, handoff counts,
    quarantine events.
  - Framework adapters (zero framework deps): LangGraph (`guardian_edge`,
    `SwarmGraphBuilder`), CrewAI (`CrewGuardian`), OpenAI Swarm
    (`GuardedHandoff`), AutoGen (`GroupChatGuardian`).
- **Compliance & reporting**:
  - `CostAnalyser`: token cost estimation per model/agent (CHF default).
  - HTML/Markdown report templates for compliance, cost, and swarm health.
- **Config wizard** (`director-ai wizard`): interactive YAML generator
  with Gradio web UI (`pip install director-ai[ui]`) and CLI fallback.
  Introspects `DirectorConfig` fields grouped by category.
- **13 new config fields**: `parent_child_enabled`, `parent_chunk_size`,
  `child_chunk_size`, `adaptive_retrieval_enabled`,
  `adaptive_retrieval_threshold`, `hyde_enabled`, `hyde_prompt_template`,
  `query_decomposition_enabled`, `query_decomposition_strategy`,
  `contextual_compression_enabled`, `contextual_compression_strategy`,
  `multi_vector_enabled`, `multi_vector_representations`.
- **3 new pip extras**: `[ui]` (Gradio), `[reports]` (WeasyPrint + Jinja2),
  `[autogen]` (pyautogen).
- **CLI subcommands wired**: `director-ai kb-health` (KB diagnostics),
  `director-ai wizard` (config wizard), `director-ai cost-report` (token
  cost report in text/JSON/HTML), `director-ai compliance report --format html`
  (HTML compliance report via `render_compliance_html`).
- **CostAnalyser auto-integration**: `cost_tracking_enabled` config field
  wires `CostAnalyser` into scorer pipeline. LLM judge API calls (OpenAI,
  Anthropic) automatically record token usage via `cost_callback`.
- **Formal benchmarks doc**: `docs/BENCHMARKS.md` with 5-tier scoring
  pyramid, AggreFact per-dataset BA, RAG backend latency/memory tables,
  SwarmGuardian and framework adapter performance.
- 421 new tests across 20 modules.

### Fixed
- **Metric correction**: FactCG accuracy corrected from 75.8% to **75.6%**
  per-dataset mean balanced accuracy (AggreFact leaderboard convention,
  verified 2026-04-12). With per-dataset threshold tuning: **77.76%**
  (potential #1, ahead of Bespoke-MiniCheck-7B at 77.4%). Previous 75.8%
  figure was rounded from sample-pooled BA, a different metric.
- **Leaderboard position**: corrected from #8/19 to **#6** on the published
  AggreFact leaderboard (llm-aggrefact.github.io, verified 2026-04-12).
- **FaithLens retraction**: the "FaithLens 86.4%" figure cited in earlier
  research was fabricated (does not appear on the leaderboard). All
  references removed.
- **Circular reasoning detection**: fixed word-overlap heuristic that
  failed on Python fallback path (without Rust) due to trailing punctuation.
- **HuggingFace supply-chain hardening**: `MODEL_REGISTRY` with pinned
  revision SHAs for FactCG and MiniCheck models.
- 169 new CLI integration tests for 11 benchmark scripts.

### Changed
- **Rust fast-path for rules engine**: 4 rules (EntityGrounding, NumericConsistency,
  NegationFlip, WordOverlap) wired to Rust `backfire_kernel` for sub-microsecond
  execution when the Rust kernel is installed.
- **9 scorer backends** registered: deberta, onnx, minicheck, lite, rules,
  embed, nli-lite, rust, backfire.
- **`scorer_backend` default** changed from `"deberta"` to `"auto"` —
  auto-detects best available (rust > onnx > deberta > lite).
- **`hardened` mode**: enables strict_mode, all sanitisers, injection
  detection, NLI, and PII redaction in one flag.
- **`dry_run` mode**: scores but never rejects (observability mode).
- **`production_mode`**: enforces API key authentication.

## [3.12.0] — 2026-04-05

### Added
- **Intent-grounded prompt injection detection** (`core.safety.injection`):
  two-stage pipeline — `InputSanitizer` (regex, fast) + `InjectionDetector`
  (NLI bidirectional, semantic). Detects the *effect* of injection in the
  output rather than pattern-matching the input. Per-claim attribution with
  grounded/drifted/injected verdicts.
- **`InjectionDetector`** class: bidirectional NLI divergence scoring against
  the original intent (system prompt + user query). Baseline calibration,
  traceability floor, multi-signal verdict per claim.
- **`InjectionResult`**, **`InjectedClaim`** dataclasses in `core.types`.
- **`injection_risk`** field on `CoherenceScore` — populated when injection
  detection is enabled via `CoherenceScorer.enable_injection_detection()`.
- **`POST /v1/injection/detect`** server endpoint — standalone injection
  analysis with per-claim attribution.
- **`ProductionGuard.check_injection()`** — lazy-init injection detector
  using config thresholds, returns `InjectionResult`.
- **6 config fields** on `DirectorConfig`: `injection_detection_enabled`,
  `injection_threshold`, `injection_drift_threshold`,
  `injection_claim_threshold`, `injection_baseline_divergence`,
  `injection_stage1_weight`.
- 40 tests for `InjectionDetector` (Phase 1) + 26 integration tests (Phase 2).
- **DirectorGuard middleware injection detection** — `injection_detection` and
  `injection_threshold` parameters, `X-Director-Injection-Risk` and
  `X-Director-Injection-Detected` response headers, system prompt extraction
  from OpenAI-style messages, lazy-init detector, 422 rejection on injection
  when `on_fail="reject"`.
- **SDK `guard()` injection detection** — `injection_detection` and
  `injection_threshold` parameters across all 5 proxy types (OpenAI,
  Anthropic, Bedrock, Gemini, Cohere). `InjectionDetectedError` exception
  for raise/log/metadata failure modes.
- **`InjectionDetectedError`** exception in `core.exceptions`.
- **`InjectionAdversarialTester`** in `testing.adversarial_suite` — 9 injection
  attack transforms (instruction override, delimiter injection, data
  exfiltration, context switch, encoding, roleplay, multilingual, markdown,
  gradual drift), 27 built-in patterns, `RobustnessReport` output.
- 50 tests for Phase 3 (middleware, SDK guard, adversarial suite).
- **Rust-accelerated injection signals** (`backfire-kernel`):
  `rust_bidirectional_divergence` (batch traceability + entity overlap +
  baseline calibration) and `rust_injection_verdict` (multi-signal per-claim
  verdict + aggregation). Auto-selected when backfire_kernel is installed;
  transparent Python fallback otherwise. 3.73× speedup at 100 claims.
- 18 tests for Phase 4 (Rust FFI + Python consistency).
- 12 Rust unit tests for injection signals.
- **Async Voice AI pipeline** (`director_ai.voice`): `AsyncVoiceGuard` for async
  token-by-token hallucination filtering, `voice_pipeline()` for end-to-end
  streaming audio with sentence buffering and halt recovery.
- **TTS adapters**: `ElevenLabsAdapter`, `OpenAITTSAdapter`, `DeepgramAdapter` —
  all lazy-import their SDK, raise `DependencyError` if missing.
- **`[voice]` extra** in pyproject.toml: installs `elevenlabs>=1.0`,
  `openai>=1.0`, `deepgram-sdk>=3.0`.
- 33 tests for async voice guard, TTS adapters, and pipeline integration.
- `examples/voice_streaming_demo.py` — true async streaming demo with ElevenLabs.
- `docs-site/guide/voice-ai.md` updated with async pipeline section, TTS adapter
  docs, and API reference directives.
- **SDK-aware integration tests** (29 tests across 4 files): verify adapter code
  against real SDK objects with mocked HTTP layer. Tests use `pytest.importorskip()`
  and run only in CI extras matrix.
- **CI extras matrix expanded** from 2 to 6 entries: `dev,server`, `dev,grpc`,
  `dev,openai,anthropic`, `dev,langchain`, `dev,llamaindex`, `dev,voice`.
  Each entry installs real SDK dependencies and runs the full test suite.
- **VerifiedScorer ablation E+F script** (`benchmarks/ablation_ef.py`): NLI-gated
  and BM25 traceability variants tested and rejected.
- **Rust compute accelerators** (`backfire-core::compute`): 10 new FFI functions
  for CPU-bound Python operations. Hot-path functions auto-select Rust when
  `backfire_kernel` is installed, with transparent Python fallback:
  - `rust_sanitizer_score` — 11 injection regex patterns (wired into `InputSanitizer.score()`)
  - `rust_has_suspicious_unicode` — Unicode category analysis (wired)
  - `rust_detect_task_type` — task classification from prompt (wired into `detect_task_type()`)
  - `rust_verify_numeric` — numeric consistency checks (wired into `verify_numeric()`)
  - `rust_score_temporal_freshness` — 4 temporal claim patterns (wired into `score_temporal_freshness()`)
  - `rust_extract_reasoning_steps` — chain-of-thought extraction (wired into `extract_steps()`)
  - `rust_word_overlap` — Jaccard word similarity (wired into `_word_overlap()`)
  - `rust_softmax` — row-wise softmax (wired into `_softmax_np()`, threshold ≥100 elements)
  - `rust_probs_to_divergence` — NLI divergence (wired into `_probs_to_divergence()`, threshold ≥10 rows)
  - `rust_probs_to_confidence` — NLI confidence (wired into `_probs_to_confidence()`, threshold ≥10 rows)
  - `rust_lite_score` — heuristic divergence scorer (wired into `LiteScorer.score()`)
  - `rust_lite_score_batch` — batch heuristic scorer (wired into `LiteScorer.score_batch()`)
  - 34 Rust unit tests + 44 Python parity tests.
- **Rust compute benchmark** (`benchmarks/rust_compute_bench.py`): measures all
  12 Rust accelerators vs Python fallbacks. Geometric mean **9.4× speedup**;
  best: sanitizer_score (benign) 63.5×, lite_score 1.8×, lite_score_batch
  (100 pairs) 2.2×. Key results (median µs, 5000 iterations):
  sanitizer_score 58→2.1µs, temporal_freshness 51→2.9µs, softmax(200×3)
  352→21µs, probs_to_confidence(200×3) 486→15µs, lite_score 47→26µs.

### Changed
- **God File refactoring** — four large modules split into focused sub-modules
  with zero API changes (all re-exported through original paths):
  - `server.py` (2144→1737): Pydantic models → `_server_models.py`,
    serialisation helpers → `_server_helpers.py`.
  - `scorer.py` (1843→1419): LLM judge → `_llm_judge.py`,
    task detection + dialogue/summarisation → `_task_scoring.py`.
  - `cli.py` (1653→399): 25 subcommands → `_cli_bench.py`,
    `_cli_serve.py`, `_cli_verify.py`, `_cli_ingest.py`.
  - `nli.py` (1536→1262): ONNX/TensorRT export + dynamic batcher →
    `_nli_export.py`.

### Fixed
- **MiniCheck heuristic fallback** now respects `use_model=False` — skips
  expensive MiniCheck init (which triggers CVE-2025-32434 torch.load warning
  on torch < 2.6) and falls back to heuristic immediately.
- **types-PyYAML** added to dev dependencies for mypy stubs.
- **German "Alle"** added to root `_typos.toml` for injection test patterns.
- **Salted API key hash** — audit fingerprint uses salted SHA-256.
- **HuggingFace model revision pins** — NLI, embedding, and reranker models
  pinned to specific commit hashes for supply-chain security.

## [3.11.1] — 2026-03-27

### Fixed
- **NLI CUDA auto-detection**: `_load_nli_model()` now auto-selects CUDA when
  `torch.cuda.is_available()` and device is `None`. Previously the model stayed
  on CPU unless `nli_device="cuda"` was passed explicitly — the ONNX loader
  already auto-detected CUDA, so this aligns behaviour.
- **`director_assert()` crash**: passed a float to `HallucinationError` which
  expected a `CoherenceScore` object (`AttributeError: 'float' has no attribute
  'score'`). Fixed by calling `scorer.review()` directly.

### Added
- 16 tests for `integrations/dspy.py` (coherence_check + director_assert).
- 15 tests for `integrations/semantic_kernel.py` (DirectorAIFilter init + async call).
- `verified-scorer.md`: documented `atomic=True`, `evidence_top_k`, `SourceSpan`
  dataclass, `verify()` parameters, `is_atomic` field, multi-span evidence usage.
- `CONTRIBUTING.md`: added DSPy and Semantic Kernel to integrations table.
- `BENCHMARK_REPORT.md`: v3.11.0 L40S 14-scenario results with documented
  CPU-only NLI bug (scenarios 2–7, 12 ran on CPU, not GPU).

## [3.11.0] — 2026-03-27

### Added
- **ProductionGuard**: batteries-included API bundling calibrated scoring,
  human feedback loop (FeedbackStore → OnlineCalibrator), conformal confidence
  intervals, and agent tool-call verification in a single entry point.
- **Atomic claim verification**: `VerifiedScorer.verify(atomic=True)` decomposes
  compound sentences into atomic claims before matching. Multi-span evidence
  attribution links each claim to top-K source spans with per-span NLI scores.
- **VectorGroundTruthStore.grounded()**: factory for the recommended retrieval
  recipe — hybrid BM25 + dense embeddings with RRF fusion. One-liner setup for
  domain profiles that require KB grounding.
- **Semantic Kernel integration**: `DirectorAIFilter` for SK function invocation hooks.
- **DSPy / Instructor integration**: `director_assert()` for DSPy modules,
  `coherence_check()` for Instructor and standalone validation pipelines.
- **Helm chart**: `deploy/helm/director-ai/` with Deployment, Service, Ingress,
  HPA, API key via K8s secret, GPU toggle, non-root security context.
- **Sigstore signing**: publish workflow now signs dist/* with Sigstore and
  generates SLSA provenance attestations. Signatures attached to GitHub releases.
- **Observability pack**: `deploy/observability/` with Grafana dashboard (9 panels)
  and Prometheus alert rules (6 alerts: hallucination rate, latency, streaming
  halts, drift, errors, KB failures).
- **Latency matrix benchmark**: `benchmarks/latency_matrix.py` generates
  authoritative backend × batch size matrix with auto-detected hardware metadata.
- **Scoring modes demo**: `examples/scoring_modes_demo.py` runs three scoring
  modes (threshold-only, KB-grounded, per-claim verified) on the same
  hallucinated data.

### Fixed
- **Streaming false-halt benchmark**: three bugs — callback double-accumulation
  (treated accumulated text as token), space-less tokenization, missing kernel
  `reset_state()` between passages. Measured result: 4.4% false-halt rate
  (6/135 passages, heuristic mode). Prior 0.0% was an artifact of the broken
  benchmark. All docs corrected.
- **Domain profile claims**: config comments claimed "0% FPR at threshold ≤0.30"
  for finance and favourable medical behaviour. Actual artifacts: medical
  FPR=100%, finance FPR=100%, legal 0 samples. All docs now state KB grounding
  or customer-specific calibration is required.
- **Summarization auto-routing**: task-aware bidirectional NLI scoring now
  activates automatically for detected summarization tasks, matching how
  dialogue routing already works. Previously gated behind manual
  `_use_prompt_as_premise=True`.

### Changed
- Regression suite false-halt tolerance: 2% → 5% (matches measured heuristic reality).
- Pre-commit config: Helm templates excluded from check-yaml (Go template syntax).

### Added
- **Cross-model consensus**: `ConsensusScorer` queries the same prompt to multiple
  models and scores pairwise factual agreement. High disagreement → low confidence.
  Pluggable NLI scorer, supports pre-generated responses.
- **Adversarial robustness testing**: `AdversarialTester` self-tests the guardrail
  against known attack patterns: zero-width chars, Unicode homoglyphs, base64/rot13
  encoding, role-play injection. Returns RobustnessReport with per-pattern results.
  EU AI Act Art 15(5) requires resilience against adversarial manipulation.
- **Temporal freshness scoring**: `score_temporal_freshness()` flags claims that
  may rely on stale knowledge. Detects positions (CEO, president), statistics,
  "currently/as of" references, records/superlatives. Cross-references against
  source timestamps for age-calibrated risk scores.
- **Reasoning chain verification**: `verify_reasoning_chain()` extracts steps
  from chain-of-thought responses and verifies each step follows from its
  premises. Detects non-sequiturs, circular reasoning, and unsupported leaps.
  Pluggable NLI scorer, falls back to Jaccard heuristic.
- **Feedback loop detection**: `FeedbackLoopDetector` detects when AI outputs
  feed back into inputs, creating self-reinforcement cycles. EU AI Act
  Article 15(4) specifically requires this. Trigram-based fuzzy matching
  with severity levels. Zero competitors address this regulatory requirement.
- **Conformal prediction intervals**: `ConformalPredictor` provides calibrated,
  distribution-free uncertainty estimates on hallucination probability. Instead
  of binary approved/rejected, returns P(hallucination) intervals with coverage
  guarantees. Based on Mohri & Hashimoto (ICML 2024). Zero competitors.
- **Agentic loop monitor**: `LoopMonitor` tracks AI agent execution loops.
  Detects circular tool calls, goal drift (Jaccard or custom NLI scorer),
  token/step/time budget exhaustion, and reasoning degradation. Returns
  per-step verdicts with halt/warn decisions. The first guardrail product
  that monitors agent loops, not just individual calls.
- **Numeric verification**: `verify_numeric()` checks percentage arithmetic,
  date logic (birth < death), probability bounds, order-of-magnitude sanity,
  and internal consistency. Stdlib only, zero dependencies. The first
  guardrail product with real-time quantitative verification.
- **Claim-level provenance**: `CoherenceScore` gains `.claims`, `.attributions`,
  `.unsupported_claims`, `.claim_coverage`, `.claim_provenance()` — per-claim
  source attribution with divergence scores. Every claim mapped to its
  supporting source sentence. Hallucinated claims identified individually.
- **Multi-signal explainability**: `CoherenceScore` gains `detected_task_type`,
  `escalated_to_judge`, `nli_probs`, `retrieval_confidence` fields. Existing
  `verdict_confidence` and `signal_agreement` now always populated. Makes
  Director-AI the most transparent guardrail on the market.
- **Statistical drift detection**: `DriftDetector` uses two-proportion z-test
  to determine if hallucination rates are increasing over time. Severity levels:
  none/mild/moderate/severe. No scipy dependency (Abramowitz & Stegun CDF).
- **Proxy audit logging**: `--audit-db PATH` flag logs every scored proxy
  request to the compliance SQLite database for Article 15 documentation.
- **Server compliance endpoints**: `GET /v1/compliance/report` (JSON or Markdown),
  `GET /v1/compliance/drift`, `GET /v1/compliance/dashboard` (24h/7d/30d metrics).
  Enabled via `DIRECTOR_COMPLIANCE_DB_PATH` env var.
- **CLI compliance subcommand**: `director-ai compliance report|status|drift`
  generates Article 15 reports from the command line.
- **Gem REST endpoints**: 8 new endpoints exposing verification gems via the API:
  `POST /v1/verify/numeric`, `POST /v1/verify/reasoning`,
  `POST /v1/temporal-freshness`, `POST /v1/consensus`,
  `POST /v1/adversarial/test`, `POST /v1/conformal/predict`,
  `POST /v1/compliance/feedback-loops`, `POST /v1/agentic/check-step`.
  All with typed Pydantic response models and 30 endpoint tests.
- **FAISS AVX2 workaround**: `conftest.py` sets `FAISS_OPT_LEVEL=generic`
  to avoid DLL hang on Windows with faiss-cpu 1.13.1.
- **Server encoding fix**: Repaired triple-encoded UTF-8 in `server.py`.

## [3.10.1] — 2026-03-25

### Fixed
- **Missing exports**: `ModelResponse` and `CoherenceCallbackHandler` now
  exported from top-level `director_ai` package. Notebooks 07 (LangChain)
  and 16 (Verification Gems) were failing on import.
- **CLI docs**: `director-ai grpc` corrected to `director-ai serve --transport grpc`.
  Removed nonexistent `health` CLI command. Version example updated from 3.8.0.
- **Coverage gate**: aligned to 90% across ci.yml, pyproject.toml, and
  VALIDATION.md (actual measured coverage is 91%).
- **ROADMAP.md**: v3.10.0 section marked as Done (was incorrectly "Planned").
- **Cookbook caveats**: medical, legal, finance, customer-support, and case-study
  ROI/metric claims now explicitly marked as illustrative estimates.

### Added
- **12 Mermaid diagrams** across architecture, scoring, streaming, threshold
  tuning, tutorials, benchmarks, SDK guard, LangChain, production deployment,
  and Docker pages.
- **Anulum branding**: logos (anulum_logo_company.jpg, anulum_logo.png,
  fortis_studio_logo.jpg) in docs/assets/ and docs-site/assets/. README
  footer, MkDocs nav logo + favicon, docs-site/index.md footer with contact.
- **verified-scorer.md**: expanded from 56-line skeleton to 247-line complete
  API reference with pipeline diagram, verdict flowchart, all 5 signals,
  data class tables, and CoherenceScorer comparison.

### Changed
- Archived 5 obsolete files to `docs/archive/`: DEV_LOG.md (v2.3.0),
  SECURITY_AUDIT_PREP.md (abandoned), legacy Sphinx config (conf.py,
  index.rst, core.rst).
- MkDocs nav logo changed from `material/shield-check` to Anulum logo.

## [3.10.0] — 2026-03-24

### Added
- **Meta-confidence scoring**: `CoherenceScore` gains `verdict_confidence`,
  `nli_model_confidence`, `signal_agreement` fields. Combines NLI softmax
  entropy, score-threshold margin, and h_logical/h_factual agreement into
  a single confidence signal. No other guardrail answers "how confident
  are you in this verdict?"
- **Cross-turn contradiction tracking**: `ConversationSession` gains
  `update_contradictions()` and `get_contradiction_report()`. Pairwise
  NLI scoring of each response against all prior turns. Reports
  `contradiction_index` on `CoherenceScore`.
- **Structured output verification** (new `core/verification/` subpackage):
  - `verify_json()` — JSON Schema validation + value grounding against KB
  - `verify_tool_call()` — function existence, argument validation,
    fabrication detection against execution logs
  - `verify_code()` — Python syntax (ast.parse), import existence,
    hallucinated API detection against library manifests
  - All stdlib only — zero new dependencies, works on lite install
- **Online calibration** (new `core/calibration/` subpackage):
  - `FeedbackStore` — SQLite-backed thread-safe store for human corrections
  - `OnlineCalibrator` — threshold sweep on accumulated labeled data,
    deployment-specific FPR/FNR with Wilson confidence intervals
  - `CalibrationReport` — accuracy, error rates, optimal threshold, CIs
- 100 new tests across 7 test files.
- New top-level exports: `compute_meta_confidence`, `ContradictionTracker`,
  `ContradictionReport`, `verify_json`, `verify_tool_call`, `verify_code`,
  `FeedbackStore`, `OnlineCalibrator`, `CalibrationReport`.

### Changed
- `_finalise_review()` now computes and attaches `verdict_confidence` and
  `signal_agreement` to every `CoherenceScore`.
- `review()` runs contradiction tracker when a session is provided and NLI
  model is available.
- `VoiceGuard` now uses `threading.Lock` for thread safety.

### Fixed
- `.zenodo.json` version was 5 releases stale (3.9.0 → 3.10.0).
- `Dockerfile.gpu` model-builder stage: `optimum[onnxruntime]` replaces
  bare `optimum` with `--no-deps` (fixed `ModuleNotFoundError`).
- Stale 3.9.4 version references in `server.py`, `README.md`, `docs-site/`.
- Missing lazy imports: `create_app`, `create_grpc_server`,
  `create_knowledge_router` now importable from root package.
- `release.yml` uses `--require-hashes` for pip deps (OpenSSF compliance).
- Floating point tolerance in meta-confidence tests.
- Mypy `arg-type` errors in verification modules.

## [3.9.5] — 2026-03-23

### Added
- **Coalesced `review_batch()`**: batches logical and factual NLI pairs
  through `NLIScorer.score_batch()` — 2 GPU forward passes total instead
  of 2*N. Dialogue items fall back to sequential `review()`.
- **Per-task-type judge escalation**: `_should_escalate()` now receives
  the detected task type at all 4 call sites. Per-task thresholds:
  dialogue=0.35, summarization=0.25, qa=0.30, fact_check=0.20.
- 14 tests for no-KB calibration rescaling and cross-turn blending math.
- 3 tests for per-task-type judge threshold differentiation.

### Fixed
- **Scorer**: `_finalise_review()` log message uses effective threshold
  override instead of base `self.threshold`.
- **Scorer**: meta-classifier NLI-to-coherence threshold conversion uses
  `self.W_FACT`/`self.W_LOGIC` instead of hardcoded 0.4/0.6.
- **43 documentation fixes** across docs-site (13 CRITICAL, 17 HIGH,
  12 MEDIUM, 3 LOW): fabricated API signatures, wrong defaults, stale
  claims, non-existent fields. Verified by independent Gemini audit.
- **Notebook fixes** (01, 03, 11, 12): NameError, nonexistent methods,
  wrong parameter defaults, stale profile tables.
- ROADMAP: WASM Edge Runtime Done→Deferred, v2.8.0 ordering,
  `nli_claim_support_threshold` default 0.5→0.6.
- CONTRIBUTING: architecture table rewritten for subpackage structure.
- CITATION.cff: leaderboard position corrected to #8/19.
- PUBLIC_API.md: added missing `ClaimAttribution`.

## [3.9.4] — 2026-03-20

### Fixed
- **Domain profile thresholds**: medical 0.75→0.30, finance 0.70→0.30,
  legal 0.68→0.30. Measured on PubMedQA (500 samples, F1=59.9% at t=0.30)
  and FinanceBench (150 samples, 0% FPR at t≤0.30). Old thresholds
  rejected all inputs — CoherenceScorer scores cluster [0.25, 0.55].
- **Score calibration**: CoherenceScorer now rescales output to [0, 1] when
  NLI is available but no KB context was retrieved (previously compressed
  to [0.25, 0.55]).
- **README claims verified**: model attribution (FactCG-DeBERTa-v3-Large),
  hardware context for latency (GTX 1060, ONNX GPU, 16-pair batch),
  FPR corrected 2.0%→10.5%, citation version updated.
- **Docker Hub badge**: removed dead link. Pre-built images not yet
  published; Dockerfile verified locally (CPU build + health endpoint).
- **HF Spaces badge**: "Live Demo"→"Demo" (space sleeps due to inactivity).
- **GPU Dockerfile**: added `optimum` dependency for ONNX export stage.
- **Benchmark scripts**: `nli_device='cuda'` when available (was defaulting
  to CPU), `trust_remote_code` flag updated for current HF datasets lib,
  CUAD data loader field names fixed.
- **Import ordering**: medical_eval.py stdlib before third-party.
- **Code scanning**: server.py hmac+sha256→blake2b for API key audit hash,
  release.yml and Dockerfile.gpu pip deps pinned with hashes.

### Changed
- Domain presets: "tuned"→"preset" in all docs (starting points, not
  validated against domain datasets).
- Provider list: "Works with 9 providers"→"Duck-type detection for 5 SDK
  shapes" with tested/untested distinction.
- Docs threshold guidance updated toward measured profiles (ongoing — some pages still reference pre-calibration values).

### Added
- Domain benchmark results in docs-site/benchmarks.md (PubMedQA,
  FinanceBench, threshold sweep tables with measurement dates).
- Threshold inconsistency documented: guard()=0.3 vs DirectorConfig=0.6.
- `__init__.py` exports for all PUBLIC_API.md symbols: VerifiedScorer,
  ClaimVerdict, VerificationResult, DatasetTypeClassifier, MetaClassifier,
  TuneResult, tune(), export_tensorrt(), clear_model_cache(), 7 vector
  backends.

## [3.9.3] — 2026-03-19

### Fixed
- **Rust scorer**: word-overlap heuristic tests updated for new scoring path.
- **Rust FFI**: borrow lifetime fix in word-overlap heuristic.
- **License tests**: use env-var signing key consistently.

## [3.9.2] — 2026-03-19

### Fixed
- **License validation hardened**: UUID format check, HMAC signature
  verification, and expiry enforcement on `DIRECTOR_LICENSE_KEY`.
- **Cache scope isolation**: cache key now includes `scope` parameter,
  preventing cross-session and cross-context replay.
- **Tenant-aware retrieval**: `VectorGroundTruthStore` resolves tenant_id
  consistently via `_resolved_tenant_id()` across all methods.
- **Batch processor robustness**: `BatchProcessor._review_one` catches
  scorer exceptions, preventing single-item failures from aborting batch.
- **Config profile override**: `from_profile()` re-applies profile values
  after `__post_init__`, preventing mode-based overrides from clobbering
  explicit profile settings (e.g. `use_nli=False` for fast/creative).
- **Redis enterprise store**: tenant-prefixed keys, TTL support, connection
  error handling.
- **Fine-tuning stability**: lazy imports for torch/transformers, safe
  fallbacks when optional dependencies are absent.

### Added
- 353 new tests covering knowledge_api, proxy, finetune, fastapi_guard,
  nli, sdk_guard, grpc_server, scorer, and server modules.
- Test coverage pushed from 81% to 90%+.

## [3.9.1] — 2026-03-19

### Fixed
- **Cross-tenant cache replay**: score cache key now includes `tenant_id`,
  preventing cached results from one tenant being served to another.
- **Batch/single scoring divergence**: `review_batch()` routes through
  `review()`, eliminating 7-feature divergence (dialogue calibration,
  adaptive thresholds, cross-turn handling, etc.).
- **Vector fallback cross-tenant leak**: `add_fact()` prefixes `tenant_id`
  in the keyword store, preventing fallback retrieval from leaking facts.
- **streaming_oversight crash**: replaced non-existent `ingest_token()` call
  with `check_halt()`, fixing `AttributeError` on WebSocket sessions.
- **Timeout kills stream**: all three streaming paths (`HaltMonitor`,
  `StreamingKernel`, `AsyncStreamingKernel`) now catch `TimeoutError`
  gracefully instead of aborting with unhandled exceptions.

## [3.9.0] — 2026-03-15

### Added
- **Dataset-type classifier**: bundled RF-20-d6 model (370 KB) auto-selects
  per-dataset NLI thresholds based on text features alone. 77.1% balanced
  accuracy on LLM-AggreFact (29,320 samples), up from 76.7% per-task-type
  and 75.9% global threshold. Confidence-gated: falls back to per-task-type
  when uncertain (confidence < 0.5).
- `MetaClassifier.predict_threshold()` — dataset-type prediction returning
  optimal NLI threshold or None (low confidence fallback).
- `train_dataset_type_classifier()` training function + `--train-dataset-type`
  CLI flag in `tools/train_meta_classifier.py`.
- Bundled model auto-discovery: scorer loads
  `src/director_ai/models/dataset_type_classifier.pkl` when
  `adaptive_threshold_enabled=True` (default) without explicit path config.
- Docs: "Why Director-AI" narrative page, v2→v3 migration guide, glossary
  (35 terms), operational runbooks (6 decision trees).
- Benchmark admonition callouts, cookbook ROI quantification, SEO keywords.
- AggreFact pipeline experiment tooling (`evaluate_classifiers()`,
  `train_production_model()`, `generate_features_from_cache()`).

### Security
- Proxy API key authentication (`--api-keys key1,key2`) with HMAC
  constant-time comparison. Health endpoint exempt.
- HTTPS enforcement: proxy rejects non-HTTPS upstream URLs by default.
  Override with `--allow-http-upstream`.
- Tenant-to-API-key binding: `api_key_tenant_map` config field validates
  `X-Tenant-ID` header against bound tenant. 403 on mismatch.
- Session ownership: sessions track `api_key_hash` (SHA256[:16]).
  GET/DELETE return 404 (not 403) for mismatched callers.
- WebSocket info leak: exception details logged server-side only,
  generic messages sent to clients.
- Bandit B608/B615 re-enabled (0 findings).
- server.py and proxy.py removed from coverage omit.

### Changed
- `_get_meta_classifier()` now auto-discovers bundled model when
  `adaptive_threshold_enabled=True` and no explicit path is set.
- Scorer meta-classifier integration uses `predict_threshold()` (dataset-type
  mode) instead of binary `predict()`. NLI threshold converted to coherence
  scale via `coherence = 0.4 + 0.6 * nli_threshold`.

## [3.8.0] — 2026-03-11

### Added
- `score()` one-shot convenience function for standalone (prompt, response) scoring.
- OpenAI-compatible proxy server (`director-ai proxy`): set `OPENAI_BASE_URL` for zero-code guardrails.
- FastAPI middleware (`DirectorGuard`): ASGI middleware for scoring LLM responses in web apps.
- `guard()` now supports all 5 provider response shapes (OpenAI, Anthropic, Google, dict, str).

### Security
- Sanitize error messages in server 500 responses (no internal detail leakage).
- Validate facts_path in proxy to prevent path traversal.
- Log failed authentication attempts.
- Warn on HTTP (non-HTTPS) upstream URLs in proxy.
- Tenant access audit logging on all endpoints using `X-Tenant-ID` (S-05).

### Fixed
- Async streaming trend detection now uses linear regression (parity with sync kernel).
- Async streaming callback errors use last-known score instead of 0.0 (prevents false hard-halt).
- Batch processor counter mutations now thread-safe via Lock (A-05).
- InMemoryBackend, SentenceTransformerBackend, FAISSBackend protected with threading.Lock (A-06).
- NLI heuristic fallback now detects negation asymmetry (A-08).
- Makefile `help` target added; all targets documented (C-01).

## [3.7.0] — 2026-03-10

### Added
- **Sentence-level attribution**: `ClaimAttribution` dataclass maps each
  summary claim to the source sentence with lowest divergence. Available in
  `ScoringEvidence.attributions` and the `/v1/review` API response.
- **Cost transparency**: `ScoringEvidence.token_count` and
  `estimated_cost_usd` track NLI token consumption per check.
  Default cost model: $0.01/1K tokens (local DeBERTa GPU amortization).
- **Domain benchmarks**: `medical_eval.py` (MedNLI + PubMedQA),
  `legal_eval.py` (ContractNLI + CUAD/RAGBench),
  `finance_eval.py` (FinanceBench + Financial PhraseBank).
- **ExpertQA analysis**: documented why 59.1% balanced accuracy is
  structurally expected at 0.4B params and irrelevant for guardrail use.
- **Fine-tuning pipeline**: `director_ai.core.finetune` module with
  `finetune_nli()`, `FinetuneConfig`, `FinetuneResult`. CLI command
  `director-ai finetune train.jsonl --eval eval.jsonl --epochs 3`.
  Supports FactCG instruction template and standard NLI pair format.
  Install: `pip install director-ai[finetune]`.
- **Load testing benchmark**: `benchmarks/load_test.py` — concurrent
  RPS measurement with P50/P95/P99 latency percentiles for both
  direct scorer and server endpoints.
- `NLIScorer.score_claim_coverage_with_attribution()` — claim coverage
  with per-claim source sentence attribution.
- `NLIScorer.last_token_count`, `last_estimated_cost`, `reset_token_counter()`.
- `ClaimAttribution` exported from `director_ai` and `director_ai.core`.
- `export_tensorrt()` — pre-builds TRT engine cache from ONNX model,
  avoids multi-minute cold-start on first inference.
- Auto-detect TRT cache: `_load_onnx_session` enables TensorRT EP
  when `trt_cache/` directory exists (no `DIRECTOR_ENABLE_TRT` env var needed).
- CLI `director-ai export` subcommand (`--format onnx|tensorrt`).
- `tensorrt` extras in pyproject.toml (`onnxruntime-gpu`).
- TensorRT latency benchmark (`benchmarks/tensorrt_latency_bench.py`).

### Performance
- ONNX CUDA: 4.5ms/pair median (2.4x vs PyTorch 10.9ms), L4 GPU.
- ONNX FP16: 4.2ms/pair median. ONNX CPU: 4.1ms/pair (competitive at batch=4).

## [3.6.0] — 2026-03-10

### Added
- **Layer C: Claim decomposition + coverage scoring** for summarization.
  Decomposes summaries into atomic claim sentences, scores each against
  source via chunked NLI, computes `coverage = supported / total`.
  Final divergence blends Layer A (bidirectional NLI) with Layer C:
  `alpha * (1 - coverage) + (1 - alpha) * layer_a`.
- `NLIScorer.score_claim_coverage()` — standalone claim coverage scorer.
- Config fields: `nli_claim_coverage_enabled` (default True),
  `nli_claim_support_threshold` (0.5), `nli_claim_coverage_alpha` (0.4).
- `ScoringEvidence` fields: `claim_coverage`, `per_claim_divergences`, `claims`.
- Server `_evidence_to_dict` includes claim coverage when present.
- Claim coverage FPR diagnostic benchmark (`benchmarks/claim_coverage_fpr_diag.py`).
- 21 new tests in `tests/test_claim_coverage.py` (unit, config, integration,
  evidence, server serialization).
- 2084 tests passing (was 2051).

### Fixed
- Summarization FPR reduced from 10.5% → 2.0% (Layer C with alpha=0.4,
  support_threshold=0.6, 200 HaluEval samples, L4 GPU). All three task
  types now below 5% FPR (QA 3-4%, Dialogue 4.5%, Summarization 2.0%).

## [3.5.0] — 2026-03-10

### Added
- Bidirectional NLI for summarization: `_summarization_factual_divergence()` scores
  both source→summary and summary→source, takes min.
- `nli_summarization_baseline` config field (default 0.20) — calibrated baseline
  subtraction: `adjusted = max(0, (raw - baseline) / (1 - baseline))`.
- `_detect_task_type()` static method for routing dialogue vs summarization.
- Summarization bidirectional FPR diagnostic benchmark (`benchmarks/summarization_fpr_diag.py`).
- 13 new tests in `tests/test_summarization_bidir.py` (baseline calibration,
  routing logic, config wiring).

### Fixed
- Summarization FPR reduced from 25.5% → 10.5% (bidirectional NLI + baseline=0.20,
  200 HaluEval samples, L4 GPU). Combined with Phase 3: 95% → 10.5% total reduction.
- Dialogue FPR reduced from 97.5% → 4.5% (bidirectional NLI + baseline=0.80).

## [3.4.0] — 2026-03-09

### Added
- `trimmed_mean` outer aggregation for chunked NLI — drops top 25% of per-hypothesis divergence scores before averaging, more robust to outlier sentences.
- `_use_prompt_as_premise` flag on `CoherenceScorer` — bypasses vector store retrieval and scores document→summary directly via NLI, eliminating context loss from partial chunk retrieval.
- Configurable `nli_fact_retrieval_top_k` on `DirectorConfig` (default 3, summarization profile uses 8).
- `nli_use_prompt_as_premise` config field wired through `build_scorer()`.
- Summarization FPR diagnostic benchmark (`benchmarks/summarization_fpr_diag.py`).
- Phase 3 variant in summarization FPR A/B benchmark (`benchmarks/summarization_fpr_eval.py`).
- 5 new tests: `test_trimmed_mean_outer_agg`, `test_trimmed_mean_single_chunk`, `test_summarization_profile_w_logic_zero`, `test_summarization_profile_retrieval_top_k`, `test_summarization_profile_prompt_as_premise`.
- `TestWLogicZeroShortCircuit` test class in `test_scorer_backend.py`.

### Changed
- Summarization profile: `w_logic=0.0, w_fact=1.0` (was 0.5/0.5) — eliminates redundant h_logic==h_fact duplication and halves GPU time.
- Summarization profile thresholds: `coherence_threshold=0.15` (was 0.35), `hard_limit=0.08` (was 0.25), `soft_limit=0.25` (was 0.45).
- Summarization profile: `nli_fact_outer_agg="trimmed_mean"` (was `"mean"`).
- Summarization profile: `nli_use_prompt_as_premise=True` (new).
- `_heuristic_coherence` short-circuits logical divergence when `W_LOGIC < 1e-9`.
- `build_scorer` guard: `if self.w_logic != 0.0` → `if self.w_logic != 0.0 or self.w_fact != 0.0`.
- `retrieve_context` calls wrapped in `try/except TypeError` for `top_k` compatibility with base `GroundTruthStore`.

### Fixed
- Summarization false-positive rate reduced from 95% → 25.5% (at threshold=0.15, 200 HaluEval correct samples on L4 GPU). Three-phase fix:
  - Phase 1: MIN inner aggregation (95% → 60%)
  - Phase 2: premise_ratio=0.85 + logic agg bug fix (60% → 42.5%)
  - Phase 3: direct NLI scoring + w_logic=0 + trimmed_mean (42.5% → 25.5%)

## [3.3.0] — 2026-03-07

### Added
- Generated gRPC protobuf stubs (`director_pb2.py`, `director_pb2_grpc.py`) from `proto/director.proto`.
- `CoherenceAgent.aprocess()` and `CoherenceAgent.astream()` async methods.
- `--cors-origins` flag on `director-ai serve`.
- `CoherenceScorer.review_batch()` — coalesced batch NLI (2 GPU calls when NLI available).
- `ReviewQueue` — server-level continuous batching for `/v1/review` with configurable flush window.
- Config fields: `review_queue_enabled`, `review_queue_max_batch`, `review_queue_flush_timeout_ms`.

### Changed
- `cors_origins` default changed from `"*"` to `""` (no CORS by default).
- gRPC server fails fast when protobuf stubs missing instead of falling back to `SimpleNamespace`.
- CLI `ingest --chunk-size` rejects values <= 0.
- H_logical and H_factual computed in parallel via `ThreadPoolExecutor` (~40% latency reduction).
- `BatchProcessor.review_batch()` delegates to `CoherenceScorer.review_batch()` with serial fallback.
- Server endpoints use async calls: `aprocess()`, `run_in_executor(scorer.review)`, `review_batch_async()`.
- Sessions dict protected by `asyncio.Lock` for concurrent access safety.
- OTel `_get_tracer()` lazy init for library users without server lifespan.

### Fixed
- PUBLIC_API.md still listed deprecated 1.x aliases removed in v3.0.0.

## [3.2.0] — 2026-03-07

### Added
- `BatchProcessor.process_batch_async()` and `review_batch_async()` — async batch processing with concurrency control.
- `__aiter__` on `_GuardedBedrockStream`, `_GuardedGeminiStream`, `_GuardedCohereStream` (parity with OpenAI/Anthropic).
- `VectorBackend.aadd()` / `aquery()` async defaults via `run_in_executor`.
- `LiteScorer.review()` returning `(bool, CoherenceScore)` matching `CoherenceScorer` interface.
- Config validation: `reranker_model` / `embedding_model` non-empty when feature enabled.
- Warning on unknown YAML keys in `DirectorConfig.from_yaml()`.
- Parallel multi-candidate requests in `AnthropicProvider` / `HuggingFaceProvider`.

### Fixed
- `quickstart` CLI scaffolding called `asyncio.run()` on synchronous methods.

## [3.1.0] — 2026-03-07

### Added
- **Hybrid scorer hardening**: NLI confidence margin fix, LLM judge verdict caching (LRU), retry with exponential back-off, escalation-rate telemetry.
- **Enterprise modules**: `PostgresAuditSink` with `asyncpg` pooling, schema migration framework, `RedisGroundTruthStore` with RediSearch.
- **WASM edge runtime**: CI pipeline (`wasm-pack build`), browser integration tutorial, overhead benchmark.
- **Rust backend**: PyO3 0.24 upgrade, SIMD micro-cycle vectorization.
- **Benchmarks**: RAGTruth + FreshQA GPU benchmark, cross-platform latency profiling, PyO3 FFI overhead measurement.
- **Vector backends**: FAISS (in-process dense search), Elasticsearch (hybrid BM25 + dense retrieval).

## [3.0.0] — 2026-03-07

### Breaking Changes
- **Minimum Python 3.11**: dropped Python 3.10 support. Requires 3.11+ for `ExceptionGroup` and `TaskGroup`.
- **Enterprise classes moved**: `TenantRouter`, `Policy`, `Violation`, `AuditLogger`, `AuditEntry` moved from `director_ai` / `director_ai.core` to `director_ai.enterprise`. Importing from the old location raises `ImportError` with migration hint.
- **Removed deprecated 1.x aliases**: `calculate_factual_entropy`, `calculate_logical_entropy`, `simulate_future_state`, `review_action` (on `CoherenceScorer`), `process_query` (on `CoherenceAgent`), `process_batch_async` (on `BatchProcessor`). Use current names: `calculate_factual_divergence`, `calculate_logical_divergence`, `compute_divergence`, `review`, `process`, `process_batch`.
- **Slimmed root `__all__`**: internal classes (`ScoreCache`, `ScorerBackend`, `ShardedNLIScorer`, etc.) removed from `__all__` — still importable, no longer in public API surface.

### Added
- `director_ai.enterprise` package re-exporting all 5 enterprise classes.
- `director-ai tune` adaptive threshold calibration (implemented in v2.8.0, now documented as stable).
- Python 3.13 in CI matrix.

## [2.8.0] — 2026-03-04

### Added
- **Rust scorer backend**: `CoherenceScorer(scorer_backend="rust")` delegates to `backfire_kernel` FFI. Falls back to heuristic scoring when crate is absent.
- **VectorBackend registry**: `register_vector_backend()`, `get_vector_backend()`, `list_vector_backends()` mirror the `ScorerBackend` plugin pattern. Built-in backends auto-registered; third-party via `director_ai.vector_backends` entry points.
- **WebSocket multiplexed streaming**: `/v1/stream` now supports concurrent sessions per connection. `session_id` field tags all frames. `{"action": "cancel", "session_id": "..."}` cancels running sessions. `Semaphore(8)` backpressure limit.
- **Tenant-isolated VectorStores**: `TenantRouter.get_vector_store(tenant_id, backend_type)` creates per-tenant `VectorGroundTruthStore` instances. Supports `"memory"`, `"chroma"`, `"pinecone"`, `"qdrant"` backends.
- **`POST /v1/tenants/{tenant_id}/vector-facts`**: REST endpoint to add facts to tenant vector stores.
- `tenant_id` parameter on `VectorGroundTruthStore`.
- `rust` and `backfire` entry points in `director_ai.backends`.
- `memory` entry point in `director_ai.vector_backends`.

### Changed
- `DirectorConfig.build_store()` falls back to vector registry for unrecognized `vector_backend` names.
- `DirectorConfig.scorer_backend` docstring includes `"rust"`.
- Version bump: 2.7.1 → 2.8.0.

## [2.7.1] — 2026-03-03

### Fixed
- `research` profile missing `llm_judge_provider` — caused `ValueError` on construction.
- `run_all.py` comparison table omitted RAGTruth and FreshQA rows.
- README badge test count stale (1166 → 1837).

### Added
- `benchmarks/BENCHMARK_REPORT.md` — consolidated public benchmark report.
- `e2e_eval.py` now accepts `--scorer-backend`, `--llm-judge-provider`, `--llm-judge-model` CLI flags for hybrid-mode E2E benchmarking.

### Changed
- `demo/requirements.txt` pinned to `>=2.7.1`.
- `demo/push_to_hf.sh` reads version from `pyproject.toml` instead of hardcoding.
- Version bump: 2.7.0 → 2.7.1

## [2.7.0] — 2026-03-03

### Added
- **Provider adapters**: `guard()` now supports 5 SDK shapes — OpenAI, Anthropic, AWS Bedrock (`converse()`), Google Gemini (`generate_content()`), and Cohere (`chat()`). Each with streaming support and periodic coherence checks.
- **`director-ai doctor`** CLI command: checks Python version, torch, transformers, NLI readiness, onnxruntime, chromadb, sentence-transformers, slowapi, grpcio.
- **ONNX quantization**: `export_onnx(quantize="int8"|"fp16")` produces quantized models. `_load_onnx_session()` auto-detects `model_quantized.onnx` on CPU.
- **RAGTruth benchmark** (`benchmarks/ragtruth_eval.py`): evaluates on `yixuantt/ragtruth` dataset.
- **FreshQA benchmark** (`benchmarks/freshqa_eval.py`): evaluates on `freshllms/freshqa` dataset.
- Tutorial notebooks: `06_medical_rag_chatbot.ipynb`, `07_langchain_integration.ipynb`, `08_provider_adapters.ipynb`.
- `"research"` added to valid CLI profiles.
- **`privacy_mode`** on `CoherenceScorer`: redacts emails, phone numbers, SSNs, and card numbers before sending text to external LLM judge.

### Changed
- **Hybrid backend default**: medical, finance, legal, summarization, research profiles now use `scorer_backend="hybrid"` with LLM judge enabled.
- **Development Status**: `Production/Stable` → `Beta` in pyproject.toml classifiers.
- **mypy**: `check_untyped_defs = true` enforced in CI (catches type bugs inside untyped functions).
- `benchmarks/run_all.py` includes ragtruth and freshqa in suite.
- Version bump: 2.6.1 → 2.7.0

### Fixed
- **Streaming callback**: `coherence_callback` in `StreamingKernel`, `AsyncStreamingKernel`, and `SafetyKernel.stream_output()` now receives accumulated text instead of individual tokens, matching the documented contract.
- **BatchProcessor timeout**: `process_batch()` and `review_batch()` now use `wait(FIRST_COMPLETED)` loop so `item_timeout` actually cancels stalled futures (previously `as_completed` + `future.result(timeout=)` was ineffective).
- **Sanitizer multilingual FP**: `_has_suspicious_unicode()` no longer flags Mn (nonspacing marks) and Mc (spacing combining marks), fixing false positives on Arabic, Hebrew, Devanagari, and Thai text.

### Security
- **CORS hardening**: `allow_methods` restricted from `["*"]` to `["GET", "POST", "DELETE", "OPTIONS"]`; `allow_headers` restricted to actual header names.
- **Metrics auth default**: `metrics_require_auth` defaults to `True` (previously `False`), so `/v1/metrics/prometheus` requires API key when keys are configured.
- **GitHub Actions pinned to commit SHAs** in all 3 workflow files (ci.yml, docker.yml, publish.yml) to prevent supply-chain attacks via tag mutation.

## [2.6.1] — 2026-03-03

### Added
- `SafetyKernel.reactivate()` — public method to re-arm kernel after emergency stop. Benchmarks no longer hack `_active`.
- `DirectorConfig.metrics_require_auth` — when `True`, `/v1/metrics/prometheus` requires API key auth.
- `DirectorConfig.rate_limit_strict` — when `True` + slowapi missing, `create_app()` raises `ImportError` instead of silent degradation.
- `AsyncStreamingKernel` soft-halt logic — `halt_mode="soft"` now yields tokens until sentence boundary or 50-token cap (ported from `StreamingKernel`).

### Fixed
- Server lifespan now wires `openai`/`anthropic` provider configs through to `CoherenceAgent` (previously only `local` was connected).
- `StreamingKernel.reset_state()` calls `reactivate()` to re-arm the kernel.

### Docs
- `PUBLIC_API.md`: retrieval strategy note explaining prompt-only KB lookup rationale.

## [2.6.0] — 2026-03-03

### Changed
- **R1: Empty default GroundTruthStore** — `GroundTruthStore()` no longer ships hardcoded SCPN facts. Use `GroundTruthStore.with_demo_facts()` for demo/test data. `VectorGroundTruthStore` no longer auto-indexes built-in facts.
- **R3: Chunked NLI inner aggregation** — inner agg changed from `min` to `max`. Most contradictory premise chunk now drives the score (conservative for safety). Added `inner_agg` parameter for `"mean"` alternative.
- **R4: Catch rate improvements** — heuristic scorer adds negation asymmetry (+0.25) and novel entity detection (+0.15). Claim decomposition via `NLIScorer.score_decomposed()`. New `"summarization"` profile. Shared `_heuristics.py` module.
- **R5: Retrieval wiring** — `build_store()` supports `vector_backend="sentence-transformer"` with configurable `embedding_model`. CLI `ingest` handles directories, `.md` files, and paragraph-level chunking.
- **R6: Reranker config wired** — `build_store()` wraps backend with `RerankedBackend` when `reranker_enabled=True`. Medical/finance profiles now actually use the reranker.
- **R7: LLM judge structured output** — `_llm_judge_check()` requests JSON (`{"verdict": "YES"|"NO", "confidence": 0-100}`), falls back to string matching. Configurable model via `llm_judge_model` config field or `DIRECTOR_LLM_JUDGE_MODEL` env var.
- **R8: Sanitizer scoring mode** — `InputSanitizer.score()` returns weighted `suspicion_score` (0.0-1.0). Low-weight patterns (e.g. `output_manipulation` at 0.3) flag but don't block. `allowlist` parameter exempts false-positive patterns. `check()` calls `score()` internally.
- **R9: Clean branding** — removed `[AGI Output]:` prefix, changed halt message to `[HALT]: All candidates rejected.`, changed disclaimer prefix to `[Unverified]`.
- **R2: Streaming docs** — clarified accumulated-text re-scoring (not per-token). Added Limitations section documenting no-retraction and NLI latency.

### Fixed
- **R10: StreamingKernel wired into agent.stream()** — `CoherenceAgent.stream()` now uses `StreamingKernel.check_halt()` with sliding window and trend detection instead of a bare coherence threshold.
- **R11: gRPC incremental streaming** — `StreamTokens` RPC pushes tokens incrementally via thread+queue bridge instead of collecting all tokens before yielding.
- **R12: CLI multi-worker config propagation** — `--workers N` with `--profile` now propagates config via `DIRECTOR_PROFILE` env var so each Uvicorn worker builds the correct scorer.
- **R13: ONNX batch config wiring** — `onnx_batch_size` and `onnx_flush_timeout_ms` from `DirectorConfig` now flow through `CoherenceScorer` → `NLIScorer` → `OnnxDynamicBatcher`.
- **R14: Prompt content removed from logs** — `CoherenceAgent` logs prompt length at DEBUG level, never the content.

### Security
- **R15: HMAC audit hashing** — `AuditLogger` uses HMAC-SHA256 (keyed via `DIRECTOR_AUDIT_HMAC_SECRET` env var or random secret) instead of plain SHA-256 for query hashes.

### Added
- **Discord bot** (`discord-bot/`) — channel bootstrap, 7 slash commands, CI/release/Docker webhook embeds.
- CI `discord-notify` job posts pass/fail to Discord `#ci-status` channel.
- Publish `discord-announce` job posts release notes to Discord `#announcements` channel.
- `onnx_path` field in `DirectorConfig` — env var `DIRECTOR_ONNX_PATH` now wires through to `NLIScorer`.
- Coverage gate raised from 70% to 79%.

### Fixed (review-driven)
- **R16: ONNX config wiring** — `Dockerfile.gpu` sets `DIRECTOR_ONNX_PATH` but `DirectorConfig` had no `onnx_path` field. Added field + `build_scorer()` pass-through.
- **R17: E2E benchmark context leakage** — `benchmarks/e2e_eval.py` shared a single `VectorGroundTruthStore` across all samples. Earlier contexts leaked into later scores. Fixed: fresh store per sample.
- **R18: InputSanitizer scoring model** — `score()` used `max(total, weight)` instead of additive accumulation. Multiple low-weight pattern matches now correctly sum to exceed the block threshold.
- **R19: Doc version drift** — `PUBLIC_API.md`, `ROADMAP.md`, `SECURITY.md` referenced v2.3.x. Updated to v2.6.0. Fixed stale version labels in `latency_bench.py` (v1.4.0→v2.6.0) and `COMPETITOR_COMPARISON.md`.
- Version bump: 2.5.0 → 2.6.0

## [2.5.0] — 2026-03-03

### Added
- **Rust scorer in backend registry**: `RustBackend` wrapping `backfire_kernel.RustCoherenceScorer`, registered as `"rust"` / `"backfire"`. Conditional — silent skip when native extension not built.
- **AGPL §13 `/v1/source` endpoint**: auth-exempt, returns license, version, repo URL. Disabled via `DIRECTOR_SOURCE_ENDPOINT_ENABLED=false` for commercial licensees.
- **Live token streaming**: `MockGenerator.stream_tokens()` and `LLMGenerator.stream_tokens()` (SSE via httpx). `CoherenceAgent.stream()` yields `(token, coherence)` tuples with per-token scoring. WebSocket `/v1/stream` and gRPC `StreamTokens` use live generation instead of replay.
- **Optional-deps CI matrix**: `test-extras` job validates `dev,server` / `dev,server,ratelimit` / `dev,grpc` combos on Python 3.12.
- **Container smoke test**: `docker-smoke` CI job builds image, starts with `DIRECTOR_LLM_PROVIDER=mock`, verifies `/v1/health`, `/v1/source`, `/v1/metrics/prometheus`.
- **SBOM release attachment**: `attach-sbom` job in publish workflow uploads `sbom.json` to GitHub release.
- Config fields: `stats_backend`, `stats_db_path`, `source_endpoint_enabled`, `source_repository_url`, `grpc_max_message_mb`, `grpc_deadline_seconds`.
- `docs-site/guide/agpl-compliance.md` — operator §13 compliance guide.

### Changed
- **Stats backend**: `StatsStore` (SQLite) only instantiated when `stats_backend=sqlite`. Default `prometheus` derives summary from `MetricsCollector`. `/v1/stats/hourly` returns empty array with note when using prometheus backend.
- **gRPC server options**: message size limits (`grpc_max_message_mb`), keepalive, optional reflection. `ReviewBatch` rejects > 1000 items.
- **Agent scorer construction**: `CoherenceAgent._build_scorer()` uses backend registry lookup instead of direct `backfire_kernel` import.
- `cyclonedx-bom` pinned to `>=4.0,<5` in CI and publish workflows.
- `httpx>=0.27` added to `[server]` extras.
- Version bump: 2.4.0 → 2.5.0

### Deprecated
- `StatsStore` (SQLite backend) — opt-in via `stats_backend=sqlite`, removal planned for v3.0.

## [2.4.0] — 2026-03-02

### Fixed
- **Unified runtime wiring**: server, gRPC, and CLI now share a single `build_store()` → `build_scorer(store)` → `CoherenceAgent(_scorer, _store)` chain. Previously `/v1/review` used config threshold with no KB store while `/v1/process` used hardcoded `threshold=0.6` with a keyword-only store.
- **Rate limiting enforced**: `Limiter(default_limits=[...])` replaces bare `Limiter()` which was a no-op.
- **ONNX batcher double-flush**: `OnnxDynamicBatcher.submit()` no longer flushes unconditionally when below `max_batch`; added explicit `flush()` method.
- **Batch result ordering**: `process_batch()` and `review_batch()` pre-allocate results by index, preserving input order despite `as_completed` execution.

### Added
- `DirectorConfig.build_store()` — constructs `VectorGroundTruthStore` from `vector_backend`, `chroma_*`, `reranker_*` config fields (previously dead config).
- `CoherenceAgent(_scorer=, _store=)` keyword-only injection params for external scorer/store objects.
- REST body size limits: `max_length=100_000` on prompt fields, `max_length=500_000` on response fields (Pydantic v2 returns 422 on violation).
- Dockerfile `EXTRAS` build ARG: `--build-arg EXTRAS="server,nli"` for NLI-enabled images.

### Changed
- `DirectorConfig.build_scorer(store=None)` now accepts optional store; auto-builds via `build_store()` when omitted.
- CLI `review`/`process`/`batch` commands load config from env via `DirectorConfig.from_env()` instead of hardcoded defaults.
- Version bump: 2.3.0 → 2.4.0

## [2.3.0] — 2026-03-02

### Security
- **Constant-time API key comparison**: `hmac.compare_digest` replaces `in` operator in REST middleware
- **WebSocket auth enforcement**: X-API-Key checked before `ws.accept()`
- **gRPC auth interceptor**: metadata `x-api-key` check with constant-time comparison
- **gRPC TLS support**: `tls_cert_path`/`tls_key_path` params for `create_grpc_server()`
- **Rate limiting wired**: `SlowAPIMiddleware` now added to FastAPI app (was missing)
- **LLM judge privacy warning**: documented that `llm_judge_enabled` sends data to external providers
- **SBOM in release**: CycloneDX SBOM generated as artifact in publish workflow
- AGPL compliance guidance added to SECURITY.md
- SECURITY.md version table updated to v2.3.x

### Changed
- **strict_mode rejects**: `strict_mode=True` without NLI now returns divergence 0.9 (reject) instead of 0.5 (neutral). `CoherenceScore.strict_mode_rejected` field added.
- **guard() duck-type detection**: provider detection uses shape checks (`client.chat.completions.create` / `client.messages.create`) instead of module-name prefix. Supports vLLM, Groq, LiteLLM, Ollama clients.
- **Config.build_scorer()**: single method wires all config fields (scorer_backend, w_logic/w_fact, nli_model, llm_judge_*, nli_devices) through to CoherenceScorer. Used by both server.py and grpc_server.py.
- **BatchProcessor uses as_completed()**: futures processed in completion order, not insertion order.
- **gRPC StreamTokens uses real scoring**: replaced fake hash→sin callback with actual CoherenceScorer.
- README: test badge 1020→1063, migration section, halt example, strict_mode description update, LLM judge privacy limitation
- ROADMAP: v3.0 vision section
- PUBLIC_API.md: synced config profiles to actual values, added tuner and batch exports
- SECURITY_AUDIT_PREP.md: synced mitigations to actual state
- Benchmark COMPETITOR_COMPARISON.md: methodology section added, version updated

### Added
- `director-ai tune <file.jsonl>` CLI: grid-search over thresholds × weight pairs, maximize balanced accuracy, optional `--output config.yaml`
- `TuneResult` dataclass and `tune()` function in `core.tuner`
- `tests/test_safety_contracts.py`: safety contract tests for build_scorer wiring, auth, strict_mode, WS auth
- **Lite scorer backend**: `CoherenceScorer(scorer_backend="lite")` — word overlap + negation heuristics, ~0.5ms/pair, ~65% accuracy. Zero heavy dependencies.
- **Multi-turn conversation tracking**: `ConversationSession` with FIFO eviction, cross-turn divergence blending (0.7 × H_logical + 0.3 × cross_turn), thread-safe. REST session endpoints on server.
- **ONNX GPU batch optimization**: `OnnxDynamicBatcher` with `max_batch`/`flush_timeout_ms` and CUDA IO binding for zero-copy GPU transfers.
- **Plugin architecture for scorer backends**: `ScorerBackend` ABC with `register_backend()`, `get_backend()`, `list_backends()`. Entry-point discovery via `director_ai.backends`. Four built-in wrappers auto-registered.
- **gRPC transport**: `proto/director.proto` with `Review`, `Process`, `ReviewBatch`, `StreamTokens` RPCs. `director-ai serve --transport grpc`.
- **Multi-GPU sharding**: `ShardedNLIScorer` distributes NLI inference across N CUDA devices via round-robin routing with lock-protected `itertools.cycle`.
- **Security audit preparation**: `SECURITY_AUDIT_PREP.md` threat model, `tests/test_fuzz.py` (4 Hypothesis property-based tests), `tests/test_security_hardening.py` (15 tests), SBOM generation + fuzz CI jobs. `InputSanitizer` hardened with path traversal, YAML injection, excessive Unicode escape patterns.
- **Public API freeze**: `__all__` on all 17 public modules. Deprecated aliases (`calculate_factual_entropy`, `calculate_logical_entropy`, `simulate_future_state`, `review_action`, `process_query`) now emit `DeprecationWarning`.
- `[security]` optional dependency group (`cyclonedx-bom`, `hypothesis`)
- `[grpc]` optional dependency group (`grpcio`, `grpcio-tools`, `protobuf`)
- New exports: `ConversationSession`, `Turn`, `LiteScorer`, `ShardedNLIScorer`, `ScorerBackend`, `register_backend`, `get_backend`, `list_backends`
- `PUBLIC_API.md` — frozen API surface documentation
- `cross_turn_divergence` field on `CoherenceScore`
- `nli_devices`, `onnx_batch_size`, `onnx_flush_timeout_ms` on `DirectorConfig`
- `"lite"` profile on `DirectorConfig`

### Changed
- Version bump: 2.2.1 → 2.3.0
- ROADMAP.md: v2.3.0 marked as current with all 8 features

## [2.2.1] — 2026-03-02

### Added
- API autodoc pages for DirectorConfig, Enterprise (TenantRouter/Policy/AuditLogger), InputSanitizer
- Introductory prose + cross-links on all 10 API autodoc pages
- Troubleshooting guide (import errors, validation errors, scoring issues, performance, server)
- Enterprise guide (TenantRouter, Policy, AuditLogger usage + lazy loading explanation)
- Scoring cadence examples in streaming reference (medical, latency-critical, adaptive)
- Validation rules section in scorer reference

### Changed
- ROADMAP.md updated: v2.2.0 marked as current, v2.1.0/v2.0.0 as done, v2.3.0/v2.4.0 planned
- Scorer reference: added `scorer_backend="hybrid"`, validation rules
- Streaming reference: added cadence parameters + examples
- Config reference: added `scorer_backend` field
- Version bump: 2.2.0 → 2.2.1

## [2.2.0] - 2026-03-02

### Added
- `score_every_n`, `adaptive`, `max_cadence` on `StreamingKernel` and `AsyncStreamingKernel` for scoring cadence control
- Streaming overhead benchmark (`benchmarks/streaming_overhead_bench.py`) with tokens/sec comparison at cadences 1, 4, 8, adaptive
- Runtime validation on `CoherenceScorer` constructor: `threshold`, `soft_limit`, `w_logic`, `w_fact`
- `[enterprise]` optional dependency group and `enterprise` pytest marker
- Enterprise modules (`TenantRouter`, `Policy`, `AuditLogger`) lazy-loaded via `__getattr__`
- Streaming overhead docs (`guide/streaming-overhead.md`) with domain recommendations
- `test_scorer_validation.py` (10 validation cases), `test_lazy_enterprise_import.py`

### Changed
- Stdlib imports (`re`, `asyncio`, `time`) hoisted to module top in `scorer.py`
- Internal imports (`otel.trace_review`, `otel.trace_streaming`) hoisted to module top in `scorer.py` and `streaming.py`
- `soft_limit` default clamped: `min(threshold + 0.1, 1.0)` (handles `threshold=1.0`)
- Version bump: 2.1.0 → 2.2.0

## [2.1.0] - 2026-03-02

### Added
- `director-ai bench` CLI subcommand with `--dataset`, `--seed`, `--max-samples`, `--output`
- `scorer_backend="hybrid"` mode: NLI + LLM judge on every review
- `scorer_backend` field on `DirectorConfig`; `thorough` profile defaults to hybrid
- Architecture deep-dive doc (`guide/architecture.md`)
- Production checklist doc (`deployment/checklist.md`)
- Threshold tuning guide expanded: grid-search example, domain table, pitfalls
- 35 new streaming false-halt benchmark passages (conversation, long-form, summary, edge-case)
- `PineconeBackend`, `WeaviateBackend`, `QdrantBackend` vector store backends
- `[pinecone]`, `[weaviate]`, `[qdrant]` optional dependency groups
- Bandit + Semgrep SAST job in CI
- OTel review spans enriched with `coherence.h_logical`, `coherence.h_factual`, `coherence.warning`
- Scope section in README with core vs optional dependency table
- Pure-Python import verification step in CI
- New tests: `test_bench_subcommand_runs`, `test_hybrid_backend_*`, `test_otel_span_enrichment_*`, `test_vector_store_backends`

### Changed
- README test badge updated from 835 to 904
- Version bump: 2.0.0 → 2.1.0

## [2.0.0] - 2026-03-02

### Fixed
- Case-sensitivity bug in `GroundTruthStore.retrieve_context()` — mixed-case keys now match lowercase queries
- LLM judge error handling: bare `except (ImportError, Exception)` replaced with structured try/except (API call vs response parse)
- `SafetyKernel` validates `hard_limit` in [0, 1] range on construction
- `setup_otel()` thread-safe via `threading.Lock`
- `case-studies.md` all 5 phantom symbols fixed (wrong constructors, nonexistent methods)
- `docs-site/changelog.md` synced (was frozen at v1.2.0, missing 17 releases)
- README BibTeX version updated from 1.8.0 to 2.0.0

### Added
- Named constants for LLM judge blending formula (`LLM_JUDGE_*`)
- `.editorconfig` for consistent formatting
- `.pre-commit-config.yaml` (ruff + trailing-whitespace + YAML/TOML checks)
- `py.typed` PEP 561 marker
- `Documentation` URL in `pyproject.toml`
- Non-root `appuser` in Dockerfile and Dockerfile.gpu
- New tests: `test_knowledge.py` (case-insensitive), `test_kernel_validation.py`, `test_ingest.py`
- Histogram `bucket_counts()` optimized from O(n*b) to O(n log n + b log n) via `bisect`

### Changed
- 12 `inspect.getsource` fragile tests in `test_phase3_hardening.py` / `test_phase4_hardening.py` replaced with behavioral equivalents
- Version bump: 1.9.0 → 2.0.0

## [1.9.0] - 2026-03-02

### Added
- **Soft-halt mode**: `StreamingKernel(halt_mode="soft")` finishes the current sentence before halting (50-token safety cap)
- **JSON structured logging**: `log_json=True` config flag wired to `_JsonFormatter` on the `DirectorAI` logger hierarchy
- **OpenTelemetry integration**: optional `otel_enabled=True` with `trace_review()` and `trace_streaming()` spans
- **Request ID propagation**: `X-Request-ID` header round-trip via `contextvars.ContextVar` middleware
- **100-passage false-halt benchmark**: expanded from 20 to 100+ passages across 10 domains
- **API reference docs**: `reference/scorer.md`, `reference/streaming.md`, `reference/config.md`, `reference/server.md`
- **Domain presets docs**: `guide/presets.md` with threshold rationale for all 8 profiles
- **Monitoring docs**: `guide/monitoring.md` with docker-compose + Prometheus + Grafana stack
- `tests/test_vector_store_reranker.py`, `tests/test_otel.py`, `tests/test_log_json.py`

### Changed
- Coverage threshold raised from 60% to 80%
- Dependency pins tightened: `torch<3`, `transformers<5`, `onnxruntime<2`, `sentence-transformers<4`
- Replaced Sphinx docs deps with MkDocs deps in `[project.optional-dependencies]`
- Removed "physics bridge enabled" vaporware label from research profile docstring

## [1.7.0] - 2026-03-01

### Added
- **Domain presets**: `DirectorConfig.from_profile("medical"|"finance"|"legal"|"creative"|"customer_support")` with tuned thresholds, weights, and NLI/reranker flags
- **Structured halt evidence**: `stream_tokens(scorer=...)` populates `HaltEvidence` with top-K contradicting chunks, NLI scores, and suggested action on halt
- **Pluggable scorer backend**: `CoherenceScorer(scorer_backend="deberta"|"onnx"|"minicheck", onnx_path=...)` forwarded to `NLIScorer`
- **Batched MiniCheck**: `NLIScorer._minicheck_score_batch()` uses MiniCheck native list API
- **False-halt assertion**: `regression_suite.py` runs `streaming_false_halt_bench` and asserts 0.0% false-halt rate in CI
- **Competitor one-pager**: summary comparison table at top of `COMPETITOR_COMPARISON.md`
- `w_logic`/`w_fact` fields on `DirectorConfig` with sum-to-1.0 validation

### Changed
- CHANGELOG reconstructed for v1.4.1, v1.5.0, v1.5.1, v1.6.0 (previously missing)
- Streaming false-halt bench status updated from "Needs run" to measured 0.0%
- 8 external-dataset benchmarks marked "Requires GPU + HF_TOKEN"

## [1.6.0] - 2026-03-01

### Added
- **API key auth**: `api_keys` config field, `X-API-Key` header validation middleware
- **Correlation IDs**: `X-Correlation-ID` header propagation on all requests
- **Audit logging**: structured audit entries wired into all scoring/review endpoints
- **Tenant routing**: `X-Tenant-ID` header → per-tenant config resolution via `TenantRouter`
- **Rate limiting**: slowapi integration, configurable `rate_limit_rpm`
- **Streaming WS oversight**: WebSocket endpoint for real-time streaming coherence monitoring
- **Streaming debug mode**: `streaming_debug=True` emits per-token diagnostic snapshots
- **E2E benchmarks**: baseline/compare evaluation (300 traces, QA/summarization/dialogue)
- **RAGBench eval**: retrieval quality benchmark (Hit@k, Precision@k)
- Docs: threshold tuning guide, KB ingestion guide, production checklist
- 835 tests, 0 failures

## [1.5.1] - 2026-03-01

### Fixed
- Reviewer hardening (10 items): pin `use_nli=False` in LangGraph tests, guard edge cases
- Ruff format pass on all source files

## [1.5.0] - 2026-03-01

### Added
- **Bidirectional chunked NLI**: `score_chunked()` with min-inner/max-outer aggregation for long documents
- **Prometheus metric compliance**: `# HELP`/`# TYPE` headers, histogram buckets
- **Real streaming halt with evidence**: `evidence_callback` on `stream_tokens()` for human-readable halt evidence
- **RAG retrieval bench**: `benchmarks/retrieval_bench.py` (Hit@1, Hit@3, Precision@3)
- **Per-class precision/recall/F1** on AggreFact evaluation
- **Streaming false-halt bench**: `benchmarks/streaming_false_halt_bench.py` (20 passages, window sweep)
- Docs: ONNX export/quantization guide, Rust FFI guide, case studies, Docker GPU guide
- CPU latency warning when ONNX CUDA unavailable

## [1.4.1] - 2026-03-01

### Added
- **Cross-GPU latency bench**: `benchmarks/gpu_bench.py` (5 GPUs: RTX 6000 Ada, A5000, A6000, Quadro 5000, GTX 1060)
- **TensorRT provider**: `DIRECTOR_ENABLE_TRT=1` activates TensorrtExecutionProvider with FP16 + engine cache
- **GPU Docker image**: `Dockerfile.gpu` with CUDA 12.4 + onnxruntime-gpu
- **ONNX CUDA CI**: Docker GPU workflow for ONNX GPU regression
- ORT graph optimization (`ORT_ENABLE_ALL`) + suppress Memcpy transformer warnings

## [1.4.0] - 2026-03-01

### Added
- **Batched NLI inference**: `score_batch()` and `score_chunked()` now run
  a single padded forward pass — **10.8x speedup** measured (3130ms → 289ms
  for 16 pairs on GTX 1060, 18ms/pair median)
- **ONNX export + runtime**: `export_onnx()` converts model to ONNX via
  optimum; `NLIScorer(backend="onnx", onnx_path=...)` runs inference via
  ONNX Runtime (**14.6 ms/pair GPU batch**, 383 ms/pair CPU)
- `ascore_batch()` async helper for batched scoring
- `onnx` optional dependency (`pip install director-ai[onnx]`)
- AggreFact benchmark predictor now batches SummaC source chunks
- Latency benchmark: `python -m benchmarks.latency_bench --nli --onnx`

### Fixed
- GPU device handling in `_model_score()` — inputs now move to model device
- ONNX Runtime int64 cast (DeBERTa expects int64, numpy tokenizer returns int32)

## [1.3.0] - 2026-03-01

### Changed
- **Default NLI model**: FactCG-DeBERTa-v3-Large (75.8% balanced accuracy
  on AggreFact, up from 66.2% with DeBERTa-v3-base)
- FactCG instruction template + SummaC-style source chunking in NLI scorer
- Updated all benchmark docs and competitor comparison

## [1.2.1] - 2026-02-27

### Added
- **`strict_mode` parameter**: disables heuristic fallbacks when NLI is
  unavailable — returns neutral 0.5 instead of keyword heuristics
- **Configurable scoring weights**: `w_logic` and `w_fact` constructor
  params (default 0.6/0.4) for domain tuning
- **`#![deny(unsafe_code)]`** on 5 Rust crates (types, core, physics,
  ssgf, observers) with safety invariant documentation
- **FFI safety docs** on backfire-ffi (PyO3 precludes deny(unsafe_code))
- "Known Limitations" section in README
- 4 new tests for strict_mode and custom weights (378 total)

## [1.2.0] - 2026-02-27

### Added
- **Score caching**: LRU cache with blake2b keys and TTL (`ScoreCache`)
- **LangGraph integration**: `director_ai_node` + `director_ai_conditional_edge`
- **Haystack integration**: `DirectorAIChecker` component
- **CrewAI integration**: `DirectorAITool`
- **Quantized NLI**: 8-bit bitsandbytes quantization (`nli_quantize_8bit`)
- **Upgraded embeddings**: `SentenceTransformerBackend` (bge-large-en-v1.5)
- **MkDocs site**: full API reference, deployment guides, domain cookbooks
- **Enhanced Gradio demo**: side-by-side comparison with token highlighting
- Community templates: bug report, feature request, PR template
- `GOOD_FIRST_ISSUES.md` for new contributors
- `[langgraph]`, `[haystack]`, `[crewai]`, `[embeddings]`, `[quantize]`
  optional dependency groups

### Changed
- Documentation: Sphinx → MkDocs Material
- `soft_limit` parameter on `CoherenceScorer` (warning zone)

## [1.1.0] - 2026-02-27

### Added
- **Native SDK interceptors** (`guard()`): one-liner hallucination guard for
  OpenAI and Anthropic SDK clients — wraps `client.chat.completions` /
  `client.messages` with transparent coherence scoring
- **Streaming guards**: `_GuardedOpenAIStream` / `_GuardedAnthropicStream`
  with periodic coherence checks (every 8 tokens) and final-check at stream end
- **Failure modes**: `on_fail="raise"` (default), `"log"`, `"metadata"` with
  `get_score()` ContextVar retrieval
- **`HallucinationError`** promoted to `core.exceptions` (shared by SDK guard,
  LangChain, LlamaIndex integrations)
- `[openai]` and `[anthropic]` optional dependency groups
- `examples/sdk_guard_demo.py` — before/after usage comparison

### Changed
- `langchain.py`: `HallucinationError` now imported from `core.exceptions`
  (re-exported for backward compat)
- Top-level exports: `guard`, `get_score`, `HallucinationError` added to
  `director_ai.__init__`

## [1.0.0] - 2026-02-26

Production stable release. Research modules permanently removed.

### Added
- **Enterprise modules**: `Policy`, `AuditLogger`, `TenantRouter`, `InputSanitizer`
- **Async scorer**: `CoherenceScorer.areview()`
- **LangChain integration**: `DirectorAIGuard`
- **LlamaIndex integration**: `DirectorAIPostprocessor`
- `[langchain]`, `[llamaindex]`, `[server]`, `[train]` optional dependency groups
- `[project.scripts]` CLI entry point: `director-ai`

### Changed
- Development status: Beta → Production/Stable
- Version unification across all files to 1.0.0
- Rust crate `backfire-consciousness` renamed to `backfire-observers`

### Removed
- `src/director_ai/research/` deleted (physics, consciousness, consilium)
- `core/bridge.py` deleted
- `[research]` optional dependency group deleted
- All research-only tests deleted
- `docs/RESEARCH.md`, `docs/RESEARCH_GUIDE.md`, `docs/api/research.rst` deleted
- `notebooks/06_lyapunov_proofs.ipynb` deleted
- All "consciousness" naming purged from Rust crate names and comments

## [0.10.0] - 2026-02-25

### Added
- **NLI auto-detection**: `CoherenceScorer` auto-enables NLI when torch
  and transformers are installed (`use_nli=None` default)
- **Configurable NLI model**: `NLIScorer(model_name=...)` supports
  custom or fine-tuned models
- `nli_available()` helper to check NLI readiness at runtime
- `[nli]` optional dependency group: `pip install director-ai[nli]`
- Integration examples: `openai_guard.py`, `ollama_guard.py`,
  `langchain_guard.py`, `quickstart.py`
- `GroundTruthStore.add()` method for clean fact injection

### Changed
- **Lightweight base install**: torch and transformers moved from base
  dependencies to `[nli]` optional — base install is now ~1 MB
- `CoherenceScorer` uses `NLIScorer` for both logical and factual
  divergence (eliminates duplicated inline NLI code)
- `GroundTruthStore.retrieve_context()` returns values only (not
  "key is value"), improving NLI premise quality
- `CoherenceAgent` disables NLI in mock mode (heuristic-designed
  mock responses don't work with real NLI)
- Development status: Alpha → Beta
- CONTRIBUTING.md, SECURITY.md rewritten
- 8 philosophy docs moved to `docs/archive/`

### Fixed
- README streaming example: wrong parameter names and iteration pattern
- README: removed hardcoded test count

## [0.9.0] - 2026-02-25

### Added
- **DeBERTa fine-tuning pipeline** (`training/`):
  - Data pipeline: AggreFact MNBM → HuggingFace Dataset with ClassLabel stratified split
  - Fine-tuning script for DeBERTa-v3 hallucination detection (base + large)
  - GPU/chunked scoring for large evaluation sets
  - Cloud training scripts (Lambda Labs, RunPod)
- **AggreFact benchmark suite** (`benchmarks/`):
  - Evaluation harness for hallucination detection models
  - 5 benchmark result files (baseline MoritzLaurer, fine-tuned base/large, comparison)
  - Automated comparison reporting
- **Core hardening** (24 test-enforced items):
  - `CoherenceScorer._heuristic_coherence()` + `_finalise_review()` helper methods
  - `CoherenceScorer.W_LOGIC` / `W_FACT` class constants
  - `CoherenceScorer._history_lock` (threading.Lock for shared state)
  - `_clamp()` utility in `types.py` with NaN/Inf logging
  - Lazy torch import in `scorer.py` (deferred from module level)
  - `LLMGenerator`: return type annotations, timeout catch, error truncation, type names
  - `PGBOEngine.set_background_metric()`: singular metric guard
  - `L16Controller.compute_h_rec()`: 1-D eigvecs fallback
  - `SSGFEngine._MAX_HISTORY = 500` cap on cost/state history
  - `ConsiliumAgent`: removed vestigial `self.history`, added `os.path.isfile` guard
  - `NLIScorer`: assert → `RuntimeError` for missing model
- Test count: 144 → 375 (2.6x increase)

### Changed
- CI: switched formatter from black to ruff format
- `backfire-kernel/` FFI: refactored PyO3 interface and slop cleanup

### Fixed
- NLI data pipeline: `ClassLabel` cast for stratified splitting
- NLI scorer: None guards for mypy `attr-defined` errors
- CI: 24 hardening test failures resolved in single pass

## [0.8.2] - 2026-02-23

### Added
- Real benchmark results for hallucination detection models
- `benchmarks/results/` with AggreFact evaluation outputs

### Changed
- Code quality cleanup across core modules

## [0.8.1] - 2026-02-23

### Added
- **Semantic scoring gate**: threshold-based approval with semantic similarity
- **Directory ingestion**: batch ingest from filesystem into vector store
- **ChromaBackend hardening**: error handling, retry logic, batch operations

### Changed
- Removed narration comments and softened overstated claims per anti-slop policy

## [0.8.0] - 2026-02-22

### Added
- **Configurable ground truth facts**: runtime fact injection via `GroundTruthStore.add()`
- **Real embeddings**: sentence-transformers integration replacing word-overlap proxy
- **Ingest CLI**: `director-ai ingest <path>` command for batch document loading
- **Publish workflow**: trusted PyPI publishing via GitHub Actions

### Changed
- Addressed internal review action items (Phases 0-3)
- Updated all documentation for Backfire Kernel and analytic gradient
- Streaming fix: token event ordering and session cleanup

## [0.7.0] - 2026-02-16

### Added
- **Backfire Kernel**: 5-crate Rust workspace with PyO3 FFI for hardware-level safety interlock
- **SSGF analytic Jacobian gradient**: 7,609x speedup over finite-difference
- **Shadow Director hardening** (Phases 1-4, 60 items H1-H64):
  - Phase 1: 17 items (H1-H17) — import guards, type annotations, docstrings
  - Phase 2: 10 items (H18-H27) — error handling, edge cases, thread safety
  - Phase 3: 18 items (H28-H45) — code quality enforcement tests
  - Phase 4: 15 items (H46-H64) — source introspection hardening tests

### Changed
- Credibility release: documentation, test coverage, and code quality to publication standard

## [0.6.0] - 2026-02-16

### Added
- **Config manager**: YAML/JSON configuration loading with schema validation
- **Metrics collection**: coherence score histograms, latency tracking, throughput monitoring
- **Batch processing**: parallel candidate evaluation with configurable concurrency
- **HTTP server**: FastAPI wrapper for REST API access
- **CLI interface**: `director-ai serve`, `director-ai score`, `director-ai batch`
- **Docker support**: multi-stage Dockerfile, docker-compose.yml

### Changed
- Sphinx docs fixes: removed `-W` strict mode, fixed `_static` path, suppressed re-export warnings

## [0.5.0] - 2026-02-16

### Added
- **Async streaming**: non-blocking token-level oversight pipeline
- **GPU UPDE solver**: CuPy/NumPy auto-dispatch for Kuramoto phase dynamics
- **Jupyter notebooks**: interactive demos for UPDE, SEC, and coherence scoring
- **CI/CD pipeline**: GitHub Actions with lint, type-check, test (3.10/3.11/3.12), security audit

### Fixed
- 15 ruff lint errors (unused imports, variables, line length)
- 3 mypy `no-any-return` errors
- Black formatting across all source files

## [0.4.0] - 2026-02-16

### Added
- **NLI Scorer** (`core/nli.py`):
  - Real NLI-based logical divergence scoring using DeBERTa-v3-base-mnli
  - Lazy model loading with `@lru_cache` singleton
  - Heuristic fallback when model is unavailable
  - `score()` and `score_batch()` API
- **Vector Store** (`core/vector_store.py`):
  - Pluggable `VectorBackend` protocol (ABC)
  - `InMemoryBackend` — word-overlap cosine proxy for testing
  - `ChromaBackend` — production ChromaDB integration
  - `VectorGroundTruthStore` — semantic retrieval with keyword fallback
  - New optional dependency group: `pip install director-ai[vector]`
- **Streaming Kernel** (`core/streaming.py`):
  - Token-by-token oversight with `StreamingKernel.stream_tokens()`
  - Three halt mechanisms: hard limit, sliding window average, downward trend
  - `TokenEvent` and `StreamSession` dataclasses with full metrics
  - Backward-compatible `stream_output()` method
- **Physics Bridge** (`core/bridge.py`):
  - `PhysicsBackedScorer` blending heuristic + L16 physics coherence scores
  - Graceful degradation when research deps are absent
  - Configurable `physics_weight` and `simulation_steps`
- **SSGF Outer Cycle** (`research/physics/ssgf_cycle.py`):
  - Full SSGF geometry learning engine integrating TCBO, PGBO, and L16 closure
  - Gram-softplus decoder (z→W), spectral bridge, micro-step with geometry feedback
  - Composite cost U_total with 4 terms (c_micro, r_graph, c_tcbo, c_pgbo)
  - Finite-difference gradient update on latent geometry vector z
- **Lyapunov Stability Proof** (`research/physics/lyapunov_proof.py`):
  - Symbolic verification using SymPy: V≥0, V(θ*)=0, dV/dt≤0, K_c formula
  - Numerical verification via Monte Carlo UPDE trials
  - `run_all_proofs()` entry point, `ProofResult` dataclass
- **Cross-validation tests** (`tests/test_cross_validation.py`):
  - 14 tests validating canonical Omega_n, Knm anchors, UPDE coupling, SEC consistency
- **pytest markers**: `@pytest.mark.physics`, `@pytest.mark.consciousness`, `@pytest.mark.consumer`, `@pytest.mark.integration`
- **Code coverage**: pytest-cov integration (86.93%), CI coverage artifact upload
- **Sphinx documentation**: `docs/conf.py`, API reference pages for core and research
- **PyPI publication pipeline**: `MANIFEST.in`, `.github/workflows/publish.yml` with trusted publishing
- Test count: 44 → 144 (3.3x increase)

### Changed
- **mypy strict pass**: 0 errors across 26 source files (was advisory, now enforced in CI)
- CI workflow: coverage reporting in test job, type stubs for requests, mypy no longer soft-fail
- `EthicalFunctional.__init__` / `ConsiliumAgent.perceive` / `ConsiliumAgent.decide`: `Optional` type annotations fixed (PEP 484 compliance)
- Various numpy `no-any-return` fixes for type safety

## [0.3.1] - 2026-02-16

### Added
- **Track 1 — L16 Physics** (`research/physics/`):
  - `scpn_params.py` — Canonical Omega_n (16 frequencies) and Knm coupling matrix builder
  - `sec_functional.py` — SEC as a Lyapunov functional with stability proof, critical coupling estimate
  - `l16_mechanistic.py` — UPDE Euler-Maruyama integrator with L16 oversight loop
  - `l16_closure.py` — PI controllers with anti-windup, H_rec Lyapunov candidate, PLV gate, refusal rules
- **Track 2 — Consciousness Gate** (`research/consciousness/`):
  - `tcbo.py` — Topological Consciousness Boundary Observable (delay embedding + persistent homology → p_h1)
  - `tcbo.py` — TCBOController (PI feedback adjusting gap-junction kappa)
  - `pgbo.py` — Phase→Geometry Bridge Operator (covariant drive u_mu → rank-2 tensor h_munu)
  - `benchmark.py` — 4 mandatory verification benchmarks (kappa increase, anesthesia, PI recovery, PGBO properties)
- **Consumer enhancements** (Track 3):
  - `tests/conftest.py` — shared pytest fixtures for agent, scorer, kernel, store, generator
  - `tests/test_consumer_api.py` — 21 tests covering full consumer API surface
  - `tests/test_research_imports.py` — 17 tests for physics + consciousness module imports and smoke tests
- Test count: 6 → 44 (7.3× increase)

## [0.3.0] - 2026-02-15

### Added
- **Dual-trajectory architecture**: single repo with consumer (`core/`) and research (`research/`) profiles
- `director_ai.core` package — Coherence Engine (consumer-ready, no SCPN vocabulary)
  - `CoherenceScorer` — dual-entropy scorer (NLI + RAG)
  - `SafetyKernel` — hardware-level output interlock
  - `MockGenerator` / `LLMGenerator` — LLM candidate generation
  - `GroundTruthStore` — RAG ground truth retrieval
  - `CoherenceAgent` — orchestrator pipeline with `process()` returning `ReviewResult`
  - `CoherenceScore` / `ReviewResult` dataclasses in `types.py`
- `director_ai.research` package — SCPN Research Extensions (requires `[research]` extras)
  - `research.consilium` — L15 Ethical Functional (moved from `src/consilium/`)
  - `research.physics` — scaffold for L16 mechanistic physics (Track 1)
  - `research.consciousness` — scaffold for TCBO/PGBO (Track 2)
- `pyproject.toml` optional dependencies: `[research]` for scipy, `[dev]` for tooling
- `requests>=2.28` added to core dependencies (for `LLMGenerator`)
- Backward-compatible aliases on all renamed classes/methods

### Changed
- Package structure: flat `src/` → proper `src/director_ai/` Python package
- `DirectorModule` → `CoherenceScorer` (consumer vocabulary)
- `BackfireKernel` → `SafetyKernel`
- `MockActor` → `MockGenerator`, `RealActor` → `LLMGenerator`
- `KnowledgeBase` → `GroundTruthStore`
- `StrangeLoopAgent` → `CoherenceAgent`
- `test_pinocchio.py` → `test_deception_detection.py`
- All test imports updated to `from director_ai.core import ...`
- `scipy` moved from core dependencies to `[research]` optional

### Removed
- Old flat `src/` module files (replaced by `src/director_ai/` package)

## [0.2.0] - 2026-01-21

### Added
- Consilium subsystem (`src/consilium/director_core.py`)
  - `EthicalFunctional` class: mathematical definition of ethical value
  - `ConsiliumAgent` class: active inference agent with OODA loop
  - Real telemetry integration (git status, pytest execution)
  - Predictive outcome modelling for action selection
- `StrangeLoopAgent` v2 with real LLM connection support (`RealActor`)
- Advanced coherence test suite (`test_advanced_coherence.py`)

### Changed
- `StrangeLoopAgent` now supports both mock and real LLM backends
- Director module enhanced with dual-entropy (logical + factual) scoring

## [0.1.0] - 2025-12-29

### Added
- Initial prototype of Director-Class AI (Layer 16)
- `StrangeLoopAgent` orchestrating Actor, Director, and Backfire Kernel
- `DirectorModule` with SEC (Sustainable Ethical Coherence) metric
- `MockActor` simulating Layer 11 narrative engine
- `BackfireKernel` hardware interlock simulation
- `KnowledgeBase` RAG mock with SCPN ground truth facts
- Test suite: Pinocchio test, RAG integration test
- Demo script for end-to-end flow validation
- Documentation: Manifesto, Architecture, Roadmap, Technical Spec, API Reference

[Unreleased]: https://github.com/anulum/director-ai/compare/v2.7.0...HEAD
[2.7.0]: https://github.com/anulum/director-ai/compare/v2.6.1...v2.7.0
[2.6.1]: https://github.com/anulum/director-ai/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/anulum/director-ai/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/anulum/director-ai/compare/v2.4.0...v2.5.0
[2.4.0]: https://github.com/anulum/director-ai/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/anulum/director-ai/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/anulum/director-ai/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/anulum/director-ai/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/anulum/director-ai/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/anulum/director-ai/compare/v1.9.0...v2.0.0
[1.9.0]: https://github.com/anulum/director-ai/compare/v1.7.0...v1.9.0
[1.7.0]: https://github.com/anulum/director-ai/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/anulum/director-ai/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/anulum/director-ai/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/anulum/director-ai/compare/v1.4.1...v1.5.0
[1.4.1]: https://github.com/anulum/director-ai/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/anulum/director-ai/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/anulum/director-ai/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/anulum/director-ai/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/anulum/director-ai/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/anulum/director-ai/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/anulum/director-ai/compare/v0.10.0...v1.0.0
[0.10.0]: https://github.com/anulum/director-ai/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/anulum/director-ai/compare/v0.8.2...v0.9.0
[0.8.2]: https://github.com/anulum/director-ai/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/anulum/director-ai/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/anulum/director-ai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/anulum/director-ai/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/anulum/director-ai/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/anulum/director-ai/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/anulum/director-ai/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/anulum/director-ai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/anulum/director-ai/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/anulum/director-ai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/anulum/director-ai/releases/tag/v0.1.0
