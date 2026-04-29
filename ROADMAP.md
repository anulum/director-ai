# Roadmap

Last updated: 2026-04-29

## Shipped Today - 2026-04-29

- Quickstart and onboarding work is marked complete for the default Docker
  Compose path, Python-only support path, `director-ai doctor`, starter
  profiles, and the configuration wizard.
- Safety surface simplification is marked complete for the public Safety
  Surface Map, experimental namespace boundary, and consolidation map.
- Benchmark transparency work is marked complete for raw runners, benchmark
  cards, external validation packets, nightly live red-team workflow, and the
  monthly adversarial validation update.
- Security follow-up planning now includes physical-action residual risks,
  attestation fuzzing, timing-parity checks, and standalone Rust-kernel
  extraction execution.

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
- `guard()` duck-type detection for provider SDK interceptors
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

### WASM Edge Runtime (Prioritised)
- CI pipeline builds `backfire-wasm` and uploads `.wasm` + JS glue artefacts
- Browser and Worker deployment guide lives in `docs-site/deployment/wasm-runtime.md`
- Release priority and host target matrix live in `requirements/wasm_release_plan.toml`
- Benchmark script remains `benchmarks/wasm_overhead_bench.py`

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
- Add `__aiter__` to cloud-provider guarded stream wrappers
- Add `async aadd()`/`aquery()` defaults on `VectorBackend` ABC for non-blocking server use
- Parallelize hosted-provider multi-candidate requests

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

### Done
- Meta-confidence scoring: `CoherenceScore` gains `verdict_confidence`, `nli_model_confidence`, `signal_agreement`
- Cross-turn contradiction tracking: `ConversationSession` pairwise NLI, `contradiction_index`
- Structured output verification: `verify_json()`, `verify_tool_call()`, `verify_code()` (stdlib only)
- Online calibration: `FeedbackStore`, `OnlineCalibrator`, `CalibrationReport` (Wilson CIs)
- EU AI Act compliance reporting: `AuditLog`, `ComplianceReporter`, `Article15Report`, `DriftDetector`
- Verification gems: `verify_numeric()`, `verify_reasoning_chain()`, `score_temporal_freshness()`, `ConsensusScorer`, `ConformalPredictor`, `FeedbackLoopDetector`, `AdversarialTester`, `LoopMonitor`
- 100+ new tests, 3200+ total across 187 files
- Coverage gate raised from 90% to 95%

### Deferred to v3.11+
- ModernBERT-large (8192 tokens) as alternate NLI backend — only path to >78% BA
- Stripe Checkout page + HMAC-SHA256 license key generation
- B608 ChromaDB parameterised filters (low priority, no user input reaches these)

## v3.15.0

### Product Simplification and Onboarding (Planned)
- [x] Make `director-ai quickstart` scaffold a default Docker Compose path with:
  - guarded chat proxy on port 8080
  - FastAPI service on port 8000
  - local Chroma persistence under `./chroma`
  - FactCG ONNX service hidden behind `docker compose --profile onnx`
- [x] Move non-Python runtime paths into clearly opt-in documentation sections:
  Rust kernel, Go gateway, Julia threshold tuner, Lean proofs, and WASM.
- [x] Document the Python-only supported path and add `director-ai doctor`
  runtime-stack audit output.
- [x] Collapse setup docs into one recommended path first, then an advanced backend matrix.
- [x] Add a "minimal support surface" policy: default Python + FastAPI + local Chroma,
  with every other runtime enabled only by explicit extra, flag, or Compose profile.
- [x] Evaluate extracting the Rust kernel and formal proof artefacts into a
  separately versioned package/repository so core users do not inherit that
  build surface.
- [x] Add a public Safety Surface Map to `ARCHITECTURE.md` that separates
  default streaming/HaltMonitor hooks from opt-in advanced hooks and disabled
  research modules.
- [x] Consolidate overlapping safety modules into fewer named responsibilities,
  with a deprecation map for duplicate meta-guard, self-evolution,
  counterfactual, provenance, and attestation paths.
- [x] Move disabled research hooks behind an explicit experimental namespace or
  feature flag so users do not mistake exploratory modules for default coverage.

### Documentation Synchronisation (Planned)
- [x] Add a visible last-updated banner to `ROADMAP.md` and `ARCHITECTURE.md`
  after each fast release burst.
- [x] Add a "shipped today" section to `ROADMAP.md` and `ARCHITECTURE.md` that
  is refreshed from the current changelog and recent commits.
- [x] Add a docs-sync checklist to release preparation so `ROADMAP.md`,
  `ARCHITECTURE.md`, `CHANGELOG.md`, `SECURITY.md`, and `VALIDATION.md` cannot
  drift after safety-hook or benchmark changes.
- [x] Reconcile `ARCHITECTURE.md` with Rust acceleration, `CoherenceAgent`
  containment, physical-grounding, and passport-verification wiring details
  already listed in the changelog.
- [x] Add a short user-facing hook decision table that answers which default,
  opt-in, and research hooks a new scientific deployment should enable first.

### Presets, Tuning, and Configuration UX (Planned)
- [x] Add profile metadata that states validation status, expected false-halt risk,
  required dependencies, and intended workload for each preset.
- [x] Extend `director-ai tune --dataset <eval.jsonl>` output so it can write a
  ready-to-use profile overlay, not just threshold and weight suggestions.
- [x] Make the Gradio UI from the `[ui]` extra the primary configuration wizard:
  profile selection, threshold tuning, facts ingestion, and calibration feedback.
- [x] Add calibrated starter presets for support, summarisation, RAG QA, finance,
  legal, medical, creative drafting, and low-latency edge/offline use.
- [x] Add YAML starter presets for STEM fact-heavy workflows, code generation,
  multi-agent swarm supervision, voice agents, and high-stakes medical review.
- [x] Extend tuner output with a confidence report explaining selected thresholds,
  trade-offs, and counterfactual examples near the decision boundary.

### Agent-Native Observability (Planned)
- [x] Add trace-level halt attribution: fact source, retrieval path, scorer path,
  token offset, threshold, and causal contribution.
- [x] Add `CounterfactualVerifier` support for "what single fact change would
  have prevented this halt?" diagnostics.
- [x] Export counterfactual and halt-cause fields as OpenTelemetry attributes.
- [x] Extend the UI with a visual trace explorer for agent, swarm, and streaming
  halt events.
- [x] Freeze a structured `SafetyEvent` schema as the single halt-reason record
  emitted by streaming, containment, attestation, ontology, trajectory, and
  cyber-physical hooks.
- [x] Wire every safety hook to emit `SafetyEvent` records with hook id, evidence
  reference, policy decision, threshold, latency, and tenant-safe explanation.
- [x] Add trace tests proving multi-hook halt attribution remains stable across
  streaming, swarm, and physical-precheck paths.

### Benchmark Transparency and External Validation (Planned)
- [x] Publish raw benchmark runners, dataset manifests, cache schema, and
  reproduction commands for every public accuracy table.
- [x] Add benchmark cards that separate pure NLI, hybrid judge, heuristic, and
  tuned-threshold modes so catch-rate claims cannot blur backend choices.
- [x] Prepare an external validation packet for a third-party accuracy benchmark.
- [x] Prepare an external security test packet focused on streaming interception,
  multi-tenant isolation, and knowledge-base ingestion.

### Dependency and Deployment Realism (Planned)
- [x] Tighten optional dependency locks with uv for `[nli]`, `[onnx]`, `[vector]`,
  `[ui]`, `[server]`, and enterprise extras.
- [x] Ship prebuilt ONNX artefact guidance and wheel coverage for common CPU/GPU
  deployment targets.
- [x] Prioritise the deferred WASM runtime for edge/offline users who cannot run
  Python services.
- [x] Add supply-chain notes for torch, transformers, ONNX Runtime, Chroma, and
  other heavy optional dependencies.
- [x] Add a tested airgap install example for the full safety stack, including
  local wheelhouse, pinned model revisions, ONNX artefacts, and optional Rust
  kernel wheels.
- [x] Convert the Rust kernel extraction decision into an execution plan:
  standalone crate API, versioning policy, Python wheel contract, proof artefact
  boundary, and release CI.
- [ ] Execute the standalone `backfire-kernel` crate extraction with independent
  versioning, release notes, crate CI, Python wheel contract tests, and proof
  artefact boundaries.
- [ ] Add a contributor path that lets Python-only changes run without Rust, Go,
  Julia, Lean, or WASM toolchains installed.

### Security Follow-Ups (Planned)
- [x] Add signed knowledge-base entries plus strict write ACLs for ingestion APIs.
- [x] Document public exposure rules for unauthenticated health, readiness, source,
  and metrics endpoints.
- [x] Add stricter CORS examples for public reverse-proxy deployments.
- [ ] Run an external security test focused on streaming paths and tenant isolation
  (execution gate and evidence validator are ready; independent report pending).
- [x] Add cross-language contract tests for Python, Rust, Go, and proto boundaries
  using property-based inputs.
- [x] Add a nightly live red-team workflow against current jailbreak/evasion
  datasets and the full scorer pyramid.
- [x] Publish a monthly public adversarial validation update in `VALIDATION.md`.
- [x] Add async contract tests and property-based fuzzing for the
  cyber-physical -> HaltMonitor path, including streaming races and cancelled
  actions.
- [x] Make physical hooks warn-only by default and require an explicit
  high-risk physical deployment flag before any blocking real-world action path
  is enabled.
- [x] Put ROS 2, MuJoCo, CARLA, and similar adapters behind a separate
  `[physical]` extra with pinned dependency guidance and isolation notes.
- [x] Add per-tenant budgets for inverse-kinematics solving, simulation checks,
  and physical action validation to prevent denial-of-service payloads.
- [ ] Extend the external security test packet to cover physical hooks,
  attestation, and cross-language trust-boundary failures.
- [ ] Document physical-action residual risks in `SECURITY.md`, including
  hardware damage, malformed action payloads, expensive solver payloads, and
  simulator dependency isolation.
- [ ] Add timing-parity tests for Rust and Python containment verification paths
  so missing Rust wheels do not weaken the side-channel story.
- [ ] Add property-based fuzzing for Merkle commitments, passport verification,
  and attestation failure cases before cross-organisation hand-off is promoted.
- [ ] Add pinned dependency guidance and sandbox notes for zk proof backends,
  simulation libraries, and physical adapters.

### Knowledge-Base Governance (Planned)
- [ ] Add semantic versioning for `VectorGroundTruthStore` facts and derived
  vector chunks.
- [ ] Add fact retraction and replacement records so stale or retracted sources
  can invalidate dependent chunks without full manual rebuilds.
- [ ] Add Merkle roots for KB snapshots and expose them through audit records.
- [ ] Combine temporal freshness with external citation/status signals for
  scientific and high-stakes domains.
- [ ] Add automatic conflict reports when a new fact contradicts an existing
  signed fact, passport claim, or retraction record.

### Future Differentiators (Planned)
- [ ] Prototype native inference-server hooks for vLLM, TGI, and llama.cpp so the
  guard can intervene before unsafe tokens are sampled.
- [ ] Design a human-reviewed self-improving guard loop from calibration feedback
  into LoRA fine-tuning jobs.
- [ ] Plan multi-modal hallucination checks for generated image, audio, and video
  outputs using caption/metadata grounding.
- [ ] Draft an open Director Safety Protocol for guard signals shared across agent
  frameworks.
- [ ] Add conflict-aware knowledge-base checks that detect contradictory user facts
  before they enter retrieval.
- [ ] Package the halt/interlock kernel as a standalone library for users who bring
  their own scorer.
- [ ] Add verifier backends for formal math and code outputs through theorem
  prover integrations.
- [ ] Design federated, privacy-preserving sharing of anonymised guard signals.
- [ ] Add sustainability scoring for token cost, energy estimates, and budget
  halts.
- [ ] Promote the recursive guard-of-the-guard layer into a guarded production
  option with drift and evasion checks.
- [ ] Publish a structured Director Safety Event schema for uniform halt telemetry
  across inference servers and downstream tools.
- [ ] Add a live physical-world grounding loop that compares sensor-fusion state
  with claimed state before and after high-risk actions.
- [ ] Design an agent passport registry for verifiable coherence history across
  organisational hand-offs.
- [ ] Promote irreversibility forecasting into the default policy path for actions
  above a calibrated risk threshold.
- [ ] Unify defence registry, self-evolution, and continual adversarial testing
  into one reviewed defence-update pipeline.
- [ ] Publish the Director safety telemetry schema as an independent
  interoperability specification for inference servers and agent frameworks.
- [ ] Add a reviewed no-go policy that makes irreversibility forecasts block
  high-risk actions above a calibrated conformal threshold.
- [ ] Design a closed-loop physical grounding evaluator that compares perception
  state, claimed state, and post-action state across camera, IMU, and simulator
  feeds.
