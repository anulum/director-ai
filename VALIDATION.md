# Validation

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | 3545 | `pytest tests/` across 187 files |
| Rust unit/integration | — | `cargo test --workspace` in backfire-kernel |
| Property-based fuzz | 200 | Hypothesis-driven InputSanitizer + CoherenceScorer |
| Docker smoke | 3 | Health, source, metrics endpoints |

CI runs tests on Python 3.11, 3.12, 3.13.

## CI Validation Gates

All gates must pass before merge.

| Gate | Tool | What it enforces |
|------|------|------------------|
| lint | `ruff check` + `ruff format` + version sync | Style, formatting, version parity |
| typecheck | `mypy src/director_ai/` | Static type safety |
| test | `pytest --cov` (3 Python versions) | Correctness + ≥90% coverage |
| test-extras | `pytest` with server/grpc extras | Integration with optional deps |
| security | `pip-audit` | Supply-chain vulnerabilities |
| sast | `bandit` + `semgrep` | OWASP + code patterns |
| fuzz | `pytest test_fuzz.py` (Hypothesis) | Property-based edge cases |
| benchmark | `benchmarks.regression_suite` | Performance non-regression |
| rust | `cargo fmt/clippy/test` | Rust backend correctness |
| sbom | `cyclonedx-py` | Software bill of materials |
| docker-smoke | `docker build` + health check | Container builds and starts |

## Coverage Policy

- **Minimum gate**: 90% line + branch coverage on `src/director_ai/`
- **Exclusions**: `server.py` (requires FastAPI), `grpc_server.py` (requires grpcio)
- **Enforcement**: `--cov-fail-under=90` in CI (pyproject.toml `[tool.coverage.report]`)
- **Actual measured**: see Codecov badge or `pytest --cov` output — the gate passes on CI but the exact percentage is not hardcoded here to avoid stale claims

## Benchmark Suite (24 evaluators)

| Benchmark | Dataset | Metric |
|-----------|---------|--------|
| AggreFact | 29,320 samples | Balanced accuracy |
| FEVER | fact verification | Accuracy |
| MNLI | natural language inference | Accuracy |
| ANLI | adversarial NLI | Accuracy |
| PAWS | paraphrase detection | Accuracy |
| VitaminC | fact verification | Accuracy |
| TruthfulQA | truthfulness | Accuracy |
| FreshQA | temporal facts | Accuracy |
| RAGTruth | RAG grounding | F1 |
| E2E Pipeline | 300 traces | Precision/Recall/F1 |
| Latency | per-pair timing | ms/pair |
| GPU | batch throughput | pairs/sec |
| Streaming | token overhead | μs/token |
| Regression | all-of-above | non-regression gate |

Run: `python -m benchmarks.run_all`.

## Regeneration

```bash
# Full local validation
pytest tests/ -v --tb=short --cov=director_ai --cov-fail-under=90
cd backfire-kernel && cargo test --workspace
python -m benchmarks.regression_suite
python tools/preflight.py
```
