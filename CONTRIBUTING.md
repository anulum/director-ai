<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- DIRECTOR-AI — CONTRIBUTING.md -->

# Contributing to Director-AI

Advanced/Labs uses are subject to the [permanent Use Limitation](docs-site/licensing.md); text and data mining and machine-learning uses require prior written permission.

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

The notebook profile tests run the native check with the pinned quality
interpreter. Set `DIRECTOR_QUALITY_PYTHON` to it before the full suite, as
described under [Required native quality check](#required-native-quality-check).

## Python-Only Contributor Path

For documentation, Python core, server, CLI, and test changes that do not touch
optional runtime surfaces, use:

```bash
pip install -e ".[dev]"
make python-only-check
```

This path does not require Rust, Go, Julia, Lean, or WASM toolchains. It runs
the fast Python preflight plus a focused smoke suite that covers import,
configuration, backend fallback, and this contributor-path contract. Optional
runtime work still uses the dedicated targets: `make test-rust`,
`make test-go`, `make test-julia`, `make test-lean`, or `make test-wasm`.

Advanced local runners can pin the interpreter and pass runner arguments
without changing the target:

```bash
make python-only-check PYTHON=.venv/bin/python PYTHON_ONLY_CHECK_ARGS="--no-tests"
make python-only-check PYTHON=.venv/bin/python PYTHON_ONLY_CHECK_ARGS="--print-plan"
```

## Reproduce document-ingest properties

The required CI fuzz job uses seed 0, the locked development dependencies and
the native wheel. Run the same document corpus from a development environment
with those dependencies installed.

```bash
python -m pytest tests/test_cli_ingest_properties.py \
  tests/test_doc_parser_properties.py tests/test_ingestion_pipeline_properties.py \
  --hypothesis-seed=0
```

The corpus has explicit example counts and input limits. It exercises real
document formats, sparse files above the CLI size limit, pipeline writes
and rollback through the in-memory vector store, and proofs reopened from
a local provenance ledger. Record the source commit, seed and minimised
Hypothesis example when reporting a failure.

## Native Rust documentation checks

The Rust quality reader checks authored declarations and binds recognised
macro recipes to their source imports. Use Rust 1.98.1 and the repository's
Python development environment. Fetch the locked reader dependencies once
before running the offline native checks.

```bash
cargo +1.98.1 fetch --locked --manifest-path tools/quality_ratchet_rust/Cargo.toml
DIRECTOR_QUALITY_CARGO="$(rustup which --toolchain 1.98.1 cargo)" \
DIRECTOR_QUALITY_RUSTC="$(rustup which --toolchain 1.98.1 rustc)" \
  python -m pytest tools/quality_ratchet_rust_tests
```

The documentation capture uses its pinned compiler and a populated registry
cache. Run it with a Cargo home and working directory whose Cargo configuration
lookup paths contain no `config` or `config.toml`. The capture reports a
refusal if one is present.

The required Rust job also audits the reader's own Cargo lock.

## Native Julia documentation checks

Use Julia 1.13.4 with the pinned declaration analyzer project. The native quality
job checks authored declarations and the output of the locked ArgParse and Turing
macros. The analyzer reads the tuner source and lock from the checkout. Run the
same reader tests locally with the analyzer's dependencies.

```bash
julia --startup-file=no --history-file=no --project=tools/quality_ratchet_julia -e 'import Pkg; Pkg.instantiate()'
JULIA_LOAD_PATH=@:@stdlib julia --startup-file=no --history-file=no --project=tools/quality_ratchet_julia tools/quality_ratchet_julia/test/runtests.jl
JULIA_LOAD_PATH=@:@stdlib julia --startup-file=no --history-file=no --project=tools/quality_ratchet_julia tools/quality_ratchet_julia/test/generated.jl
make quality-julia-native
```

## Makefile Targets

| Target | Command | Description |
|--------|---------|-------------|
| `make test` | `pytest tests/ -v --cov=director_ai --cov-fail-under=97` | Run tests with coverage (97% gate) |
| `make python-only-check` | `$(PYTHON) tools/python_only_check.py $(PYTHON_ONLY_CHECK_ARGS)` | Run contributor checks without optional runtime toolchains |
| `make test-rust` | `cargo test` in backfire-kernel | Run Rust tests |
| `make test-all` | test + test-rust | Both suites |
| `make lint` | `ruff format --check` + `ruff check` | Check style |
| `make fmt` | `ruff format` + `ruff check --fix` | Auto-fix style |
| `make bandit` | `bandit -r src/director_ai/` | SAST scan |
| `make preflight` | `python tools/preflight.py` | Full preflight gate |
| `make preflight-fast` | `preflight.py --no-tests` | Lint-only (~5s) |
| `make docs` | `mkdocs serve` | Local docs server |
| `make docs-build` | `mkdocs build --strict` | Build the Python/MkDocs reference |
| `make docs-polyglot` | Language-native generators | Build Rust, Go, TypeScript, Julia, Lean, and Protobuf references |
| `make docs-all` | docs-build + docs-polyglot | Build the complete documentation set |
| `make build` | `python -m build` | Build sdist + wheel |
| `make install-hooks` | `git config core.hooksPath .githooks` | Install local git hooks |
| `make clean` | Remove dist/, build/, __pycache__ | Clean build artifacts |

## Native Protobuf documentation checks

The Protobuf documentation tests use `protoc` 3.21.12 to inspect the source
descriptors of both checked-in schemas. In the development environment, install
the pinned descriptor reader and typing stubs, then run the native tests.

```bash
python -m pip install --require-hashes -r requirements/quality-protobuf.txt
python -m pytest tests/test_quality_ratchet_protobuf.py tests/test_quality_ratchet_protobuf_descriptors.py
```

Run `make docs-protobuf` to build the HTML reference for both schemas.

## Required native quality check

Run `make quality-ratchet` with the pinned Python, Go, TypeScript, Protobuf,
Lean, Rust and Julia tools installed. Use `QUALITY_RATCHET_ARGS` to pass the
toolchain paths or an explicit trusted `--base-ref`. Run the check with the
pinned quality interpreter. The default `--python` is the interpreter that runs
the check, so running it with the pinned interpreter selects it. Install it the
way CI does, outside the checkout:

```bash
export RUNNER_TEMP="$(mktemp -d)" GITHUB_ENV="$(mktemp)"
bash tools/install_quality_python.sh
set -a; . "$GITHUB_ENV"; set +a
make quality-ratchet PYTHON="$QUALITY_PYTHON"
```

The script downloads the pinned python-build-standalone 3.11.17 archive and
checks its size and SHA-256 before extracting it. It then creates a virtual
environment with the locked development, typing, lint and Protobuf
requirements, without a cache. `GITHUB_ENV` receives `QUALITY_PYTHON` and
`DIRECTOR_QUALITY_PYTHON`, both naming that environment's interpreter. The
check compares all configured profiles to the committed policy and baseline at
the trusted ref. The default trusted ref is `HEAD`. The check fails when that
ref lacks either file. For initial adoption, establish the reviewed policy and
baseline together at the trusted ref before enabling the required workflow.
Full preflight runs the same native check. CI first walks every new commit
against its own first parent: the pushed range, or a pull request from its
merge base. It then checks the tip against the pull request's exact base SHA,
or on a push against the pushed commit's first parent. Both aggregate checks
require the result, and the main-branch notification includes it.

A profile's tool identity changes only through a reviewed entry in the
append-only `tools/quality_ratchet_transitions.json`. The `configuration_change`
reason covers an edit of `pyproject.toml` that no tool of the three Python
profiles reads; today that is only the setuptools package exclusion. Such an
entry runs no canary. It records the TOML keys it changes, and the check
recomputes them. Every changed key must be on the reviewed list in
`tools/quality_ratchet_configuration.py`; any other key fails as an effective
change that needs canaries. The commit that adds the entry may change only that
file, the transitions file, the baseline's identity lines, the reviewed
packaging metadata and one placeholder `README.md` under an excluded package.
The walk checks that shape at that commit, and the tip check recomputes the
TOML diff against the trusted ref.
The native typing profiles use isolated Python interpreters and the repository's
configured import roots. Profile identity includes the compiler's resolved
Python package search paths.

Use `python -m tools.check_quality_ratchet --profile PROFILE capture
--output SNAPSHOT` to produce a diagnostic snapshot for one profile.
Use `--all capture` for a complete diagnostic bundle. Captures never
establish baseline authority.

The `notebook-types` profile type-checks Python code cells with MyPy in notebook
order and reports findings at their original cell and line. The native Python
compiler verifies notebook syntax, including top-level `await`. The check
statically accounts for the repository's literal `!pip` and `!director-ai`
commands. Other shell escapes, interactive magics and typing suppressions fail
the check. Notebook code cells and shell commands are never executed.
The projection reads cells with the checking interpreter's own tokenizer, and
Python 3.12 tokenises a leading `!` differently from 3.11. The profile identity
therefore binds the interpreter binary and its `tokenize` and `token` sources
for each platform, and only `linux-x64` is listed. The check refuses any other
build by name, both for the checking interpreter and for the one that runs
MyPy. The tests that run the notebook check launch the interpreter that
`DIRECTOR_QUALITY_PYTHON` names, and fail by name when it is unset.

With the Julia declaration analyzer project instantiated, run
`make quality-julia-native` to capture diagnostic evidence for the complete Julia
tuner source tree. The
report records source digests, native documentation attachments, ArgParse
option help and both methods generated by Turing for each model definition.
A parser or generator refusal leaves the report unpublished. The aggregate
gate runs the same native collector as `--profile julia-docs`. Pass `--julia`
the native Julia 1.13.4 binary. The declaration analyzer uses its own audited
Julia lock. Run the tuner package tests with the separate
`tools/julia_tuner` project.
The snapshot binds the executable, project, manifest and exact source bytes.
Native diagnostics on standard error or incomplete output fail the check.

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

## On module size

This project splits by **responsibility**, not line count. A single-responsibility
module of any length is fine; a multi-responsibility module of any length is not.
Some files are intentionally large because they are one cohesive thing — notably
`core/scoring/scorer.py` and `core/scoring/nli.py` (the scoring engines),
`guard.py` (the unified `ProductionGuard` facade),
`core/retrieval/vector_store/store.py` (the vector store), `_server_models.py`
(the API request/response schemas), and `core/runtime/streaming.py`. Please do not
flag these for size alone. Genuine multi-responsibility files are tracked for
splitting in the internal backlog.

## Production support contract

`src/director_ai/support_contract_catalog.py` classifies every export of
`director_ai.__all__` and every CLI command exactly once. Each classified
operation has three separate fields:

- the smallest installation that runs it: `core`, `core+pro` or `core+full`;
- its support: `production-qualified` or `experimental`;
- its activation: `default`, `opt-in` or `command`.

Add a new export or command to the experimental lists in the same change;
CI refuses an unclassified one. Promotion to `production-qualified` is a
reviewed change that cites the signed review records. Each setting that the
production profile turns on also names the operation that owns it.

```bash
PYTHONPATH=src:. python -m tools.support_contract_check --check
PYTHONPATH=src:. python -m tools.support_contract_check --write
```

`--write` regenerates `docs/_generated/production_support.md`. The package job
runs `scripts/probe_support_contract.py` on clean Core, Core+Pro and Core+Full
installations. It then checks that each claimed installation passed its probe
in that job, and that every receipt tested the commit's own tree
(`--receipts DIR --tree TREE`). The probe environments install the hash-locked
`requirements/ci-dev.txt` and the wheels with `--no-deps`, so the receipts
prove the wheels against that locked dependency set, not against a fresh
resolver install.

Only the Core distribution may ship the contract. An installed reader checks
that `director-ai`'s RECORD lists each contract file it imported, with a
matching sha256, and that no other distribution records it. Two limits hold
by design: a source or editable checkout, where nothing records the files, is
unverified and is never authority; and the check compares the file on disk
with its record at check time, not the bytecode that already ran.

## Pull Request Process

1. **Branch** from `main` with a descriptive name (`feature/`, `fix/`, `docs/`)
2. **Tests** must pass: `pytest tests/ -v`
3. **Lint** must pass: `ruff check && ruff format --check`
4. **Coverage**: add tests for new functionality
5. **Commits**: one logical change per commit, imperative mood, under 72 chars
6. **PR description**: summarise changes, link related issues

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
| `core/runtime/` | `streaming.py` | `StreamingKernel` | Opt-in streaming contradiction halt |
| `core/runtime/` | `async_streaming.py` | `AsyncStreamingKernel` | Non-blocking async streaming |
| `core/runtime/` | `batch.py` | `BatchProcessor` | Parallel candidate evaluation |
| `core/retrieval/` | `knowledge.py` | `GroundTruthStore` | In-memory fact store |
| `core/retrieval/` | `vector_store.py` | `VectorGroundTruthStore` | Vector store with pluggable backends |
| `core/safety/` | `policy.py` | `Policy`, `Violation` | YAML declarative policy engine |
| `core/safety/` | `audit.py` | `AuditLogger`, `AuditEntry` | Structured JSON audit trail |
| `core/safety/` | `sanitizer.py` | `InputSanitizer`, `SanitizeResult` | Prompt injection hardening (Stage 1: regex) |
| `core/safety/` | `injection.py` | `InjectionDetector` | Intent-grounded injection detection (Stage 2: NLI) |
| `core/scoring/` | `meta_confidence.py` | `compute_meta_confidence` | Verdict confidence estimation (v3.10.0) |
| `core/runtime/` | `contradiction_tracker.py` | `ContradictionTracker` | Cross-turn contradiction detection (v3.10.0) |
| `core/verification/` | `json_verifier.py` | `verify_json()` | JSON Schema + value grounding (v3.10.0) |
| `core/verification/` | `tool_call_verifier.py` | `verify_tool_call()` | Tool existence + fabrication detection (v3.10.0) |
| `core/verification/` | `code_verifier.py` | `verify_code()` | Python syntax + hallucinated API detection (v3.10.0) |
| `core/calibration/` | `feedback_store.py` | `FeedbackStore` | SQLite human correction store (v3.10.0) |
| `core/calibration/` | `online_calibrator.py` | `OnlineCalibrator` | Threshold calibration from feedback (v3.10.0) |
| `compliance/` | `audit_log.py` | `AuditLog` | Scored interaction audit trail (v3.10.0) |
| `compliance/` | `reporter.py` | `ComplianceReporter` | EU AI Act Article 15 reporting (v3.10.0) |
| `compliance/` | `drift_detector.py` | `DriftDetector` | Statistical drift detection (v3.10.0) |

### Integrations — `integrations/`

| Module | Class | Purpose |
|--------|-------|---------|
| `sdk_guard.py` | `guard()` | OpenAI/Anthropic SDK interceptor |
| `langchain.py` | `DirectorAIGuard` | LangChain Runnable guardrail |
| `llamaindex.py` | `DirectorAIPostprocessor` | LlamaIndex postprocessor |
| `langgraph.py` | `director_ai_node()` | LangGraph state node + conditional edge |
| `haystack.py` | `DirectorAIChecker` | Haystack 2.x component |
| `crewai.py` | `DirectorAITool` | CrewAI tool |
| `dspy.py` | `coherence_check()`, `director_assert()` | DSPy assertion + Instructor validation (v3.11.0) |
| `semantic_kernel.py` | `DirectorAIFilter` | Microsoft Semantic Kernel function invocation filter (v3.11.0) |

## Reporting Issues

- **Bugs**: include reproduction steps, Python version, and error output
- **Features**: describe the problem and proposed solution
- **Security**: see [SECURITY.md](SECURITY.md) — do not open public issues

## License

Director-AI is open core: the core is Apache-2.0 and the advanced & labs tier is
LicenseRef-BUSL-1.1-ANULUM-TDM. Contributions target the **Apache-2.0 core** and are accepted under
Apache-2.0. By contributing, you agree to license your contribution under the
same terms as the file it touches. See [licensing](docs-site/licensing.md) for
the full model.

## Complete CPU CI partitions

The CI coordinator calls the responsibility workflows declared by
`tools/ci_workflow_policy.toml`. Each Python version collects the complete
`tests/` inventory in eight workers, for 24 workers across Python 3.11–3.13.
Each test belongs to exactly one worker per Python version.
The full HaluEval benchmark preserves the first 200 rows of QA, summarization
and dialogue and both original responses: 1,200 model-backed reviews per
Python version. Each original row is a separate native test that reviews both
responses, allowing expensive model work to move between workers. Without
timing history, rows are assigned cyclically: each worker reviews 25 original
rows per task. Ordinary test modules remain together to preserve fixture
locality. The existing 25-row QA smoke test remains in the suite.

One immutable timing snapshot is prepared before the workers start. All workers
and their aggregate verify the same snapshot hash. Historical timings only
balance the current complete discovery: new tests enter automatically and
removed tests cannot be restored by a cache entry. Without history, scheduling
remains deterministic and complete; corrupt supplied history fails before tests.
Measured ordinary modules stay together, while individual HaluEval rows can move
between workers. The longest measured units are assigned first to the least
loaded worker.

The aggregate requires matching source SHA and Python lane, the same complete
collection, disjoint selected cases, native outcomes for every selected case,
and all original model-review identities from `benchmarks/halueval_ci_inputs.tsv`.
Downloaded artifact directories must match every expected worker index. Each
must contain its matching native report, JUnit and readable branch-coverage
databases with measured arcs; missing, extra or misidentified bundles fail
before any coverage combination.
Missing models or data, substituted inputs, duplicate reviews and skipped
required model cases fail the aggregate. Coverage from all eight indexed workers is
combined before the unchanged 97% gate. SBOM, fuzz, regression benchmarks
and push-only Docker validation run independently from the same checked-out
source. They install or build their own dependencies and do not consume the
Python test artefacts. Both required aggregate checks still require every
applicable category to succeed, including the complete Python run.

Workers retain native phase durations and JUnit alongside their coverage and
original review evidence. Only verified complete lanes produce new scheduling
weights; all three lanes and their coverage gates must succeed before a future
snapshot is cached. Zero measured unit durations receive a positive scheduling
floor of one nanosecond; original phase measurements remain in the worker
reports. Timing advice is neither cached scores nor proof that a test ran.
Failed, skipped or cancelled prerequisites fail the aggregate explicitly.

`Test (Python 3.11)` and `Lint & Format` remain the branch-protection names
through one aggregate job definition. Both require every applicable category.
Only declared PR skips for push-only Docker and main-only notification are
permitted. The notification retains the original individual job conclusions.

Run `python -m tools.audit_ci_workflow_modularity` and
`actionlint .github/workflows/ci*.yml` after workflow changes; both are part of
normal local preflight and hosted type checking. Install actionlint 1.7.12
before local preflight. Update versioned job hashes with intentional changes;
the guard refuses undeclared executable-body drift.
