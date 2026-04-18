# Benchmark result schema — v1.0.0

Every JSON file the orchestrator writes under `benchmarks/results/`
carries the same top-level shape. A result that does not round-trip
through `benchmarks.orchestrator.RunReport.from_json` fails the
regression gate — the schema is the single source of truth, not
free-form dicts.

## Top-level shape

```json
{
  "schema_version": "1.0.0",
  "run_id": "uuid4",
  "timestamp_utc": "2026-04-18T05:12:03Z",
  "environment": { ... },
  "entries": [ { ... }, ... ],
  "notes": "suite=director_ai.default"
}
```

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `str` | Bumped major on backwards-incompatible change; regression engine refuses to compare across majors. |
| `run_id` | `str` (uuid4) | Uniquely identifies this run; useful when multiple reports live in the same directory. |
| `timestamp_utc` | `str` (ISO-8601) | Must end with `Z`. No timezone offsets. |
| `environment` | object | See below. |
| `entries` | array | Ordered list of suite-case results. |
| `notes` | `str` | Free-text operator note (suite name, context). |

## `environment` object

```json
{
  "git_commit": "877feea...",
  "git_dirty": false,
  "git_branch": "main",
  "package_version": "3.14.0",
  "python_version": "3.12.13",
  "platform": "Linux-6.17.0-20-generic-x86_64-with-glibc2.39",
  "cpu_model": "AMD Ryzen 7 PRO 5850U with Radeon Graphics",
  "cpu_count": 16,
  "ram_gb": 63.16,
  "gpu_model": "NVIDIA L4",
  "gpu_count": 1,
  "gpu_memory_gb": 22.49,
  "runner": "local"
}
```

| Field | Type | Notes |
|---|---|---|
| `git_commit` | `str` | Full SHA, empty only when not a git checkout (e.g. Vertex container running from a tarball). The regression engine refuses to compare two reports with different commits unless `--allow-commit-drift` is set. |
| `git_dirty` | `bool` | `true` when `git status --porcelain` returned any lines. A dirty baseline is a red flag — the reported numbers may not match the published commit. |
| `git_branch` | `str` | Current branch; `HEAD` when detached. |
| `package_version` | `str` | `director_ai.__version__`; empty if the package is not importable. |
| `python_version` | `str` | `sys.version.split()[0]`, e.g. `3.12.13`. |
| `platform` | `str` | `platform.platform()`. |
| `cpu_model` | `str` | `/proc/cpuinfo → model name` on Linux; `sysctl machdep.cpu.brand_string` on macOS. Empty string if unknown. |
| `cpu_count` | `int` | `os.cpu_count()`. |
| `ram_gb` | `float` | Total physical RAM in GiB (rounded to 2 d.p.); 0 when unknown. |
| `gpu_model` | `str` | From `nvidia-smi --query-gpu=name`; `""` when no CUDA device. |
| `gpu_count` | `int` | Number of devices `nvidia-smi` reported. |
| `gpu_memory_gb` | `float` | Sum across devices. |
| `runner` | `str` | One of `local` / `vertex` / `ci` / `remote`. `DIRECTOR_RUN_ENV` env var overrides the CLI default. |

## `entries[i]` object

```json
{
  "name": "rust_parity_safety",
  "kind": "smoke",
  "status": "passed",
  "metrics": [
    {"name": "pass_count", "value": 80, "unit": "count", "higher_is_better": true},
    {"name": "fail_count", "value": 0,  "unit": "count", "higher_is_better": false}
  ],
  "wall_clock_seconds": 1.42,
  "dataset_hash": "",
  "dataset_size": 80,
  "seed": 0,
  "notes": "pytest exit=0"
}
```

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Stable across runs; referenced by regression rules. |
| `kind` | `str` | `accuracy` / `latency` / `e2e` / `smoke`. Category drives the regression engine's interpretation. |
| `status` | `str` | `passed` / `failed` / `skipped` / `warned`. Regression rules evaluate only `passed`. |
| `metrics` | array | One entry per measured metric. |
| `wall_clock_seconds` | `float` | Excludes orchestrator overhead (import, environment capture). |
| `dataset_hash` | `str` | SHA-256 hex of the dataset bytes when applicable; `""` for cases that don't read a dataset. |
| `dataset_size` | `int` | Number of samples / examples / patterns evaluated. |
| `seed` | `int` | Random seed used, `0` when the case is deterministic. |
| `notes` | `str` | Free-text for operator; includes tracebacks on `failed`. |

## `entries[i].metrics[j]` object

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Canonical metric name (e.g. `balanced_accuracy`, `p99_latency_ms`). |
| `value` | `float` | Numeric value. Booleans are encoded as 0.0 / 1.0 with `unit="bool"`. |
| `unit` | `str` | Free-text unit for the report generator. |
| `higher_is_better` | `bool` | Drives regression direction. A polarity mismatch between baseline and current is itself a regression finding. |

## Regression report shape (`regression.json`)

```json
{
  "findings": [
    {
      "rule": { ... RegressionRule ... },
      "baseline_value": 0.758,
      "current_value":  0.724,
      "absolute_delta": -0.034,
      "relative_delta": -0.0449,
      "reason": "absolute delta -0.034 ratio exceeds budget +0.02",
      "severity": "high"
    }
  ],
  "skipped_rules": [
    {"rule": { ... }, "reason": "case 'e2e_halueval_qa' missing from one of the runs"}
  ]
}
```

The regression engine has **no statistical machinery** — comparisons
are deterministic per-rule tolerance checks. Adding confidence-interval
gates is an intentional future change; for now, tolerances are
hand-tuned per metric based on historical variance.

## Writing a new case

A case is a callable returning `benchmarks.orchestrator.CaseOutput`.
To add one:

1. Write the function in `benchmarks/orchestrator/cases.py` (or a
   domain-specific module such as `cases_accuracy.py`).
2. Register the `SuiteCase(name=..., kind=..., call=...)` in
   `default_cases()` or in a named sub-suite.
3. Add a matching `RegressionRule` in `default_rules()` if the
   metric should gate releases.
4. Document the new metric name + unit here if it is not obvious.

## Replicability checklist

Before publishing a baseline (`benchmarks/results/baseline.json`):

* [ ] `environment.git_dirty` is `false`.
* [ ] `environment.git_commit` matches the published tag.
* [ ] `environment.runner` matches the reference runner (typically `vertex`).
* [ ] No `entries[].status == "failed"`.
* [ ] `dataset_hash` set on every accuracy / e2e case.
* [ ] `seed` set on every non-deterministic case.
