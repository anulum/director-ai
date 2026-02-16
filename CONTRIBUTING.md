# Contributing to Director-Class AI

Thank you for your interest in contributing to Director-Class AI.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<you>/director-ai.git`
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Install in development mode: `pip install -e ".[dev]"`

## Development Setup

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

### Python

- **Formatter**: [black](https://github.com/psf/black) with default settings (line length 88)
- **Linter**: [ruff](https://github.com/astral-sh/ruff) recommended
- **Type hints**: Use where practical, especially on public APIs
- **Docstrings**: Google or NumPy style

```bash
# Format
black src/ tests/

# Lint
ruff check src/ tests/
```

### Copyright Headers

**Every new source file** must include a copyright header. Use the following
template at the top of every `.py` file:

```python
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — <module description>
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
```

## Pull Request Process

1. **Branch**: Create from `main` with a descriptive name (`feature/`, `fix/`, `docs/`)
2. **Tests**: All tests must pass — `pytest tests/ -v`
3. **Style**: Code must be formatted (`black`) and lint-clean
4. **Coverage**: Add tests for new functionality
5. **Commits**: One logical change per commit; describe *why*, not just *what*
6. **PR description**: Summarize changes, link related issues, note breaking changes
7. **Review**: At least one maintainer approval required before merge

### What Makes a Good PR

- Focused scope (one feature or fix)
- Tests that fail without the change and pass with it
- No unrelated formatting changes
- Clear commit messages

## Reporting Issues

- **Bug Report**: Include reproduction steps, environment, and error output
- **Feature Request**: Describe the problem and proposed solution
- **Security Vulnerability**: See [SECURITY.md](SECURITY.md)

## Architecture Notes

The Python package lives in `src/director_ai/` with two profiles:

### Consumer — `core/` (always installed)

- `core/agent.py` — `CoherenceAgent`, main orchestrator pipeline
- `core/scorer.py` — `CoherenceScorer`, dual-entropy (NLI + RAG) scoring
- `core/kernel.py` — `SafetyKernel`, hardware-level output interlock
- `core/actor.py` — `MockGenerator` / `LLMGenerator` (candidate generation)
- `core/knowledge.py` — `GroundTruthStore` (RAG ground-truth retrieval)
- `core/types.py` — `CoherenceScore`, `ReviewResult` dataclasses

### Research — `research/` (requires `pip install director-ai[research]`)

- `research/physics/` — SCPN parameters, SEC Lyapunov functional, UPDE integrator, L16 closure
- `research/consciousness/` — TCBO observer/controller, PGBO engine, benchmarks
- `research/consilium/` — L15 Ethical Functional & active inference agent

## Priority Areas for Contribution

We especially welcome contributions in:

- **NLI Models**: Replacing mock entropy calculations with real NLI inference
- **RAG Integration**: Connecting to vector databases (FAISS, Chroma, Milvus)
- **Safety Kernel**: Rust/C++ hardware interlock implementation
- **Testing**: Property-based testing, adversarial prompt suites
- **Documentation**: Tutorials, architecture diagrams, API examples
- **Benchmarks**: Coherence score evaluation across model families
- **SSGF Integration**: Connecting SSGF outer-cycle geometry to research/physics

## License

By contributing, you agree that your contributions will be licensed under the
GNU AGPL v3.0. See [NOTICE](NOTICE) for dual-licensing details.
