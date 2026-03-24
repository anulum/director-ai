# Contributing to Director-AI

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<you>/director-ai.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Install in development mode: `pip install -e ".[dev]"`

## Development Setup

```bash
pip install -e ".[dev,research]"
make install-hooks
pytest tests/ -v
```

## Makefile Targets

| Target | Command | Description |
|--------|---------|-------------|
| `make test` | `pytest tests/ -v --cov=director_ai --cov-fail-under=90` | Run tests with coverage (90% gate) |
| `make test-rust` | `cargo test` in backfire-kernel | Run Rust tests |
| `make test-all` | test + test-rust | Both suites |
| `make lint` | `ruff format --check` + `ruff check` | Check style |
| `make fmt` | `ruff format` + `ruff check --fix` | Auto-fix style |
| `make bandit` | `bandit -r src/director_ai/` | SAST scan |
| `make preflight` | `python tools/preflight.py` | Full preflight gate |
| `make preflight-fast` | `preflight.py --no-tests` | Lint-only (~5s) |
| `make docs` | `mkdocs serve` | Local docs server |
| `make build` | `python -m build` | Build sdist + wheel |
| `make install-hooks` | `git config core.hooksPath .githooks` | Install pre-push hook |
| `make clean` | Remove dist/, build/, __pycache__ | Clean build artifacts |

## Code Style

- **Formatter**: [ruff format](https://docs.astral.sh/ruff/formatter/) (line length 88)
- **Linter**: [ruff check](https://docs.astral.sh/ruff/linter/) with rules E, F, W, I, N, UP, B, SIM
- **Type checker**: [mypy](https://mypy-lang.org/)

```bash
# Format
ruff format src/ tests/ examples/

# Lint
ruff check src/ tests/ examples/

# Type check
mypy src/
```

All three must pass in CI before merge.

## Pull Request Process

1. **Branch** from `main` with a descriptive name (`feature/`, `fix/`, `docs/`)
2. **Tests** must pass: `pytest tests/ -v`
3. **Lint** must pass: `ruff check && ruff format --check`
4. **Coverage**: add tests for new functionality
5. **Commits**: one logical change per commit, imperative mood, under 72 chars
6. **PR description**: summarize changes, link related issues

## Architecture

The package lives in `src/director_ai/` with two profiles:

### Consumer — `core/` (always installed)

| Subpackage | Module | Class | Purpose |
|------------|--------|-------|---------|
| `core/` | `agent.py` | `CoherenceAgent` | Main orchestrator pipeline |
| `core/` | `actor.py` | `LLMGenerator`, `MockGenerator` | LLM backend interface |
| `core/` | `config.py` | `DirectorConfig` | YAML/JSON configuration manager |
| `core/` | `metrics.py` | `MetricsCollector` | Prometheus-style metrics |
| `core/` | `types.py` | `CoherenceScore`, `ReviewResult` | Data types |
| `core/` | `tenant.py` | `TenantRouter` | Multi-tenant KB isolation |
| `core/scoring/` | `scorer.py` | `CoherenceScorer` | Dual-entropy (NLI + RAG) scoring |
| `core/scoring/` | `nli.py` | `NLIScorer` | DeBERTa NLI backend |
| `core/scoring/` | `lite_scorer.py` | `LiteScorer` | Heuristic-only scorer |
| `core/runtime/` | `kernel.py` | `HaltMonitor` | Output interlock |
| `core/runtime/` | `streaming.py` | `StreamingKernel` | Token-level streaming halt |
| `core/runtime/` | `async_streaming.py` | `AsyncStreamingKernel` | Non-blocking async streaming |
| `core/runtime/` | `batch.py` | `BatchProcessor` | Parallel candidate evaluation |
| `core/retrieval/` | `knowledge.py` | `GroundTruthStore` | In-memory fact store |
| `core/retrieval/` | `vector_store.py` | `VectorGroundTruthStore` | Vector store with pluggable backends |
| `core/safety/` | `policy.py` | `Policy`, `Violation` | YAML declarative policy engine |
| `core/safety/` | `audit.py` | `AuditLogger`, `AuditEntry` | Structured JSON audit trail |
| `core/safety/` | `sanitizer.py` | `InputSanitizer`, `SanitizeResult` | Prompt injection hardening |
| `core/scoring/` | `meta_confidence.py` | `compute_meta_confidence` | Verdict confidence estimation (v3.10.0) |
| `core/runtime/` | `contradiction_tracker.py` | `ContradictionTracker` | Cross-turn contradiction detection (v3.10.0) |
| `core/verification/` | `json_verifier.py` | `verify_json()` | JSON Schema + value grounding (v3.10.0) |
| `core/verification/` | `tool_call_verifier.py` | `verify_tool_call()` | Tool existence + fabrication detection (v3.10.0) |
| `core/verification/` | `code_verifier.py` | `verify_code()` | Python syntax + hallucinated API detection (v3.10.0) |
| `core/calibration/` | `feedback_store.py` | `FeedbackStore` | SQLite human correction store (v3.10.0) |
| `core/calibration/` | `online_calibrator.py` | `OnlineCalibrator` | Threshold calibration from feedback (v3.10.0) |
| `compliance/` | `audit_log.py` | `AuditLog` | Scored interaction audit trail (v3.10.0) |
| `compliance/` | `reporter.py` | `ComplianceReporter` | EU AI Act Article 15 reporting (v3.10.0) |

### Integrations — `integrations/`

| Module | Class | Purpose |
|--------|-------|---------|
| `sdk_guard.py` | `guard()` | OpenAI/Anthropic SDK interceptor |
| `langchain.py` | `DirectorAIGuard` | LangChain Runnable guardrail |
| `llamaindex.py` | `DirectorAIPostprocessor` | LlamaIndex postprocessor |
| `langgraph.py` | `director_ai_node()` | LangGraph state node + conditional edge |
| `haystack.py` | `DirectorAIChecker` | Haystack 2.x component |
| `crewai.py` | `DirectorAITool` | CrewAI tool |

## Reporting Issues

- **Bugs**: include reproduction steps, Python version, and error output
- **Features**: describe the problem and proposed solution
- **Security**: see [SECURITY.md](SECURITY.md) — do not open public issues

## License

By contributing, you agree that your contributions will be licensed under
AGPL-3.0-or-later. Commercial licensing also available — see [licensing](docs-site/licensing.md) for details.
