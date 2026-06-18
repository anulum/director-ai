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
  strict-typed modules listed in `.github/workflows/ci.yml`: `core/config.py`,
  `core/scoring/scorer.py`, `core/containment`, `core/edge`, `core/ml_bom`,
  `core/output_integrity`, and `core/output_trust`.

## Current Known Boundaries

- Lite Scorer v2 evidence remains a staged pipeline until a trained student
  artefact, held-out evaluation, ONNX export, latency measurement, model card,
  benchmark-claim review, and evidence packet are recorded.
- The independent external security report remains open until a returned
  `security-validation/` evidence directory passes the validator.
- Repo-wide public docstring debt remains open and should resume after the
  active Vertex Lite Scorer v2 run is evaluated. The live 2026-05-18 audit
  counted 611 missing normal-source public docstrings across 219 files, plus 11
  generated/proto docstring gaps across 2 files. Keep generated protobuf files
  out of manual cleanup unless the generation pipeline changes. As of
  2026-06-18, `core/containment`, `core/edge`, `core/ml_bom`,
  `core/output_integrity`, and `core/output_trust` have been added to the same
  blocking docstring ratchet as `core/config.py` and `core/scoring/scorer.py`.
- Long-range strategic context is tracked internally; `ROADMAP.md` is the
  public execution roadmap.
