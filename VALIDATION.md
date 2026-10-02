# Validation

## Test Matrix

| Suite | Count | Scope |
|-------|------:|-------|
| Python unit/integration | 3545 | `pytest tests/` across 201 files |
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
| test | `pytest --cov` (3 Python versions) | Correctness + ≥97% coverage |
| test-extras | `pytest` with server/grpc extras | Integration with optional deps |
| security | `pip-audit` | Supply-chain vulnerabilities |
| sast | `bandit` + `semgrep` | OWASP + code patterns |
| fuzz | `pytest test_fuzz.py` (Hypothesis) | Property-based edge cases |
| benchmark | `benchmarks.regression_suite` | Performance non-regression |
| rust | `cargo fmt/clippy/test` | Rust backend correctness |
| sbom | `cyclonedx-py` | Software bill of materials |
| docker-smoke | `docker build` + health check | Container builds and starts |

## Coverage Policy

- **Minimum gate**: 97% coverage on `src/director_ai/`
- **Exclusions**: `server.py` (requires FastAPI), `grpc_server.py` (requires grpcio)
- **Enforcement**: `fail_under = 97` in `pyproject.toml` (`[tool.coverage.report]`)
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

## Monthly Adversarial Validation - 2026-04-29

This update uses `tools/live_red_team.py` against current public behaviour
datasets. Reports store counts, timings, source names, and short fingerprints;
they do not store raw prompt text.

Command:

```bash
python tools/live_red_team.py \
  --output /tmp/director-live-red-team-monthly.json \
  --max-cases-per-source 10 \
  --tiers input-sanitizer,output-injection,tier1-heuristic,tier2-rules \
  --timeout-s 20
```

Dataset sample:

| Source | Rows |
|--------|-----:|
| AdvBench harmful behaviours | 10 |
| HarmBench text test | 10 |
| JailbreakBench harmful behaviours | 10 |

Fast-tier result:

| Tier | Cases | Detected | Missed | Detection rate | Median latency |
|------|------:|---------:|-------:|---------------:|---------------:|
| input-sanitizer | 30 | 0 | 30 | 0.0% | 0.007 ms |
| output-injection | 30 | 0 | 30 | 0.0% | 0.035 ms |
| tier1-heuristic | 30 | 3 | 27 | 10.0% | 0.091 ms |
| tier2-rules | 30 | 30 | 0 | 100.0% | 0.038 ms |

Scope notes:

- This is a fast-tier live-fetch sample, not a claim for the heavy NLI tiers.
- The nightly workflow exercises the full scorer pyramid when optional
  dependencies are present.
- The first external security test is still pending an independent report.

## Regeneration

```bash
# Full local validation
pytest tests/ -v --tb=short --cov=director_ai --cov-fail-under=97
cd backfire-kernel && cargo test --workspace
python -m benchmarks.regression_suite
python tools/preflight.py
```

## Complete CPU CI partitions

The CI coordinator calls the responsibility workflows declared by
`tools/ci_workflow_policy.toml`. Each Python version collects the complete
`tests/` inventory in eight workers. Each test belongs to exactly one worker.
The full HaluEval benchmark preserves the first 200 rows of QA, summarization
and dialogue and both original responses: 1,200 model-backed reviews per
Python version. Every worker evaluates a contiguous 25-row range per task.
The existing 25-row QA smoke test remains in the suite.

The aggregate requires matching source SHA and Python lane, the same complete
collection, disjoint selected cases, native outcomes for every selected case,
and all original model-review identities from `benchmarks/halueval_ci_inputs.tsv`.
Missing models or data, substituted inputs, duplicate reviews and skipped
required model cases fail the aggregate. All eight coverage databases are
combined before the unchanged 97% gate. Downstream SBOM, fuzz, regression
benchmarks and push-only Docker validation wait for the complete Python run.

`Test (Python 3.11)` and `Lint & Format` remain the branch-protection names
through one aggregate job definition. Both require every applicable category.
Only declared PR skips for push-only Docker and main-only notification are
permitted. The notification retains the original individual job conclusions.

Run `python -m tools.audit_ci_workflow_modularity` and
`actionlint .github/workflows/ci*.yml` after workflow changes; both are part of
normal local preflight and hosted type checking. Install actionlint 1.7.12
before local preflight. Update versioned job hashes with intentional changes;
the guard refuses undeclared executable-body drift.
