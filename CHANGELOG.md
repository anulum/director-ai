# Changelog

All notable changes to Director-Class AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/anulum/director-ai/compare/v1.1.0...HEAD
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
