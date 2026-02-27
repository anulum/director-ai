# Contributing to Director-AI

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<you>/director-ai.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Install in development mode: `pip install -e ".[dev]"`

## Development Setup

```bash
pip install -e ".[dev,research]"
pytest tests/ -v
```

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

| Module | Class | Purpose |
|--------|-------|---------|
| `agent.py` | `CoherenceAgent` | Main orchestrator pipeline |
| `scorer.py` | `CoherenceScorer` | Dual-entropy (NLI + RAG) scoring |
| `kernel.py` | `SafetyKernel` | Output interlock |
| `streaming.py` | `StreamingKernel` | Token-level streaming halt |
| `async_streaming.py` | `AsyncStreamingKernel` | Non-blocking async streaming |
| `nli.py` | `NLIScorer` | DeBERTa NLI backend |
| `knowledge.py` | `GroundTruthStore` | In-memory fact store |
| `vector_store.py` | `VectorGroundTruthStore` | ChromaDB vector store |
| `actor.py` | `LLMGenerator`, `MockGenerator` | LLM backend interface |
| `policy.py` | `Policy`, `Violation` | YAML declarative policy engine |
| `audit.py` | `AuditLogger`, `AuditEntry` | Structured JSON audit trail |
| `tenant.py` | `TenantRouter` | Multi-tenant KB isolation |
| `sanitizer.py` | `InputSanitizer`, `SanitizeResult` | Prompt injection hardening |
| `config.py` | `DirectorConfig` | YAML/JSON configuration manager |
| `batch.py` | `BatchProcessor` | Parallel candidate evaluation |
| `metrics.py` | `MetricsCollector` | Prometheus-style metrics |
| `types.py` | `CoherenceScore`, `ReviewResult` | Data types |

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
GNU AGPL v3.0. See [NOTICE](NOTICE) for dual-licensing details.
