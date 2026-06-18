# Documentation Integrity

This page records the repository-wide documentation contract used before public
claims, releases, and customer-facing changes.

## Sources Of Truth

- `pyproject.toml` owns the package version and optional dependency extras.
- `ROADMAP.md` owns public shipped/planned status.
- `docs/internal/AUDIT_INDEX.md` owns ignored internal audit reconciliation.
- `benchmarks/*_packet.toml` files own benchmark evidence readiness and claim
  boundaries.
- `mkdocs.yml` owns the published documentation navigation.

Do not mark roadmap or benchmark work complete from prose alone. A public claim
needs a code path, a validator or test, and a tracked evidence artefact.

## Required Local Checks

Run these before claiming documentation is current:

```bash
PYTHONPATH=src uv run --frozen python -m mkdocs build --strict
uv run --frozen ruff check README.md mkdocs.yml docs-site src/director_ai
git diff --check -- README.md mkdocs.yml docs-site src/director_ai
```

For API pages that use mkdocstrings, run a focused import smoke when a module
or public symbol changes:

```bash
PYTHONPATH=src uv run --frozen python - <<'PY'
import importlib

for name in (
    "director_ai",
    "director_ai.core.ingestion",
    "director_ai.enterprise",
    "director_ai.voice.demo",
):
    importlib.import_module(name)
print("doc_api_imports_ok")
PY
```

## Public API Documentation Rules

- Use API overview pages for primary supported surfaces, not every internal
  helper.
- Keep generated protobuf files and compatibility wrappers out of manual
  docstring cleanup unless their generation pipeline changes.
- Public dataclasses, enums, and wrappers exposed through mkdocstrings must
  explain what is safe to serialise and what stays out of tenant-safe metadata.
- Deployment and benchmark pages must say when an artefact is a fixture,
  replication packet, or unofficial smoke result.
- The CI docstring ratchet currently blocks pydocstyle `D` regressions for the
  strict-typed modules listed in `.github/workflows/ci.yml`: `core/_device.py`,
  `core/_heuristics.py`, `core/attribution`, `core/calibration`, `core/canary`,
  `core/config.py`,
  `core/scoring/scorer.py`, `core/containment`, `core/consensus`, `core/edge`,
  `core/eval_trace`, `core/evaluation`, `core/exceptions.py`,
  `core/evidence_packet`,
  `core/forecasting`, `core/ingestion`, `core/execution_rings`,
  `core/guard_control`, `core/irreversibility`, `core/mandatory.py`, `core/memory`,
  `core/meta_guard`, `core/ml_bom`, `core/output_integrity`,
  `core/output_trust`, `core/risk_threshold`, `core/safety_event.py`,
  `core/safety_protocol.py`, `core/self_healing`, `core/stats.py`,
  `core/sustainability`, `core/swarm_coherence`, `core/swarm_equilibrium`,
  `core/symbolic_chain`, `core/temporal_consistency`, `core/text_overlap.py`,
  `core/threat_intel`, `core/trace_safe`, and `core/types.py`.

## Current Known Boundaries

- Lite Scorer v2 evidence remains a staged pipeline until a trained student
  artefact, held-out evaluation, ONNX export, latency measurement, model card,
  benchmark-claim review, and evidence packet are recorded.
- The independent external security report remains open until a returned
  `security-validation/` evidence directory passes the validator.
- Repo-wide public docstring debt remains open. The live 2026-06-18 measurement
  for `ruff check src/director_ai --select D --statistics` counted 1,175
  pydocstyle findings after generated-file excludes. The guard-control ratchet
  reduced the live count to 1,171 findings in the same command; the
  execution-ring, irreversibility, safety-event, safety-protocol, stats, and
  text-overlap ratchet reduced it to 1,156 findings; the swarm, symbolic,
  temporal-consistency, threat-intel, and trace-safe ratchet reduced it to 1,115
  findings; the consensus, meta-guard, risk-threshold, and sustainability
  ratchet reduced it to 1,052 findings; the attribution, calibration, canary,
  eval-trace, and evidence-packet ratchet reduced it to 1,046 findings; the
  evaluation, forecasting, ingestion, memory, and self-healing ratchet reduced
  it to 1,033 findings; the core utility-file ratchet reduced it to 1,028
  findings. Keep generated protobuf files out of manual cleanup unless the
  generation pipeline changes. As of 2026-06-18, `core/_device.py`,
  `core/_heuristics.py`, `core/attribution`, `core/calibration`, `core/canary`,
  `core/containment`, `core/consensus`, `core/edge`, `core/eval_trace`,
  `core/evaluation`, `core/exceptions.py`, `core/evidence_packet`,
  `core/forecasting`, `core/ingestion`, `core/execution_rings`,
  `core/guard_control`, `core/irreversibility`, `core/mandatory.py`,
  `core/memory`, `core/meta_guard`, `core/ml_bom`, `core/output_integrity`,
  `core/output_trust`, `core/risk_threshold`, `core/safety_event.py`,
  `core/safety_protocol.py`, `core/self_healing`, `core/stats.py`,
  `core/sustainability`, `core/swarm_coherence`, `core/swarm_equilibrium`,
  `core/symbolic_chain`, `core/temporal_consistency`, `core/text_overlap.py`,
  `core/threat_intel`, `core/trace_safe`, and `core/types.py` have been added
  to the same blocking docstring ratchet as `core/config.py` and
  `core/scoring/scorer.py`.
- Repo-wide strict mypy remains open. The live 2026-06-18 measurement for
  `mypy --strict src/director_ai` found 621 errors in 93 files. The configured
  CI mypy command remains clean across 512 source files, so strict mode is being
  raised package by package until the global command can replace the moderate
  profile.
- The full-suite coverage gate is currently green. The live 2026-06-18 run of
  `pytest tests/ -q --tb=short --cov=director_ai --cov-report=term-missing --cov-report=xml --cov-fail-under=97`
  passed with 10,697 tests, 29 skips, and 97.50% total coverage. Remaining
  coverage debt to 100% is 753 missed statements and 563 partial branches.
- Long-range strategic context is tracked internally; `ROADMAP.md` is the
  public execution roadmap.
