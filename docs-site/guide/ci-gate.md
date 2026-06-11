# CI Quality Gate

Run the Director-AI guardrail as a **CI quality gate**: score a labelled eval set
on every pull request and fail the build when guard quality regresses. Same idea
as gating a test suite — except the thing under test is your LLM app's factual
behaviour.

## The dataset

A JSONL file, one case per line — a prompt, the response to judge, and the label
a correct guard should produce:

```json
{"prompt": "What is the capital of France?", "response": "Paris is the capital of France.", "expected": "approve"}
{"prompt": "What is the capital of France?", "response": "The capital of France is Berlin.", "expected": "reject"}
```

- `expected: "approve"` — a grounded answer the guard should let through.
- `expected: "reject"` — a hallucination the guard should catch.
- `id` is optional (defaults to the line number) and surfaces in the report.

## The command

```bash
director-ai ci-gate --dataset cases.jsonl --min-accuracy 0.9
```

It scores every case with your configured scorer, compares the approve/reject
decision to the label, prints a summary, and **exits non-zero when a threshold is
breached** — so CI blocks the merge.

| Flag | Meaning |
|------|---------|
| `--dataset PATH` | JSONL cases file (required) |
| `--min-accuracy R` | Minimum overall accuracy, 0–1 (default `0.9`) |
| `--min-catch-rate R` | Optional: minimum hallucination catch rate on `reject` cases |
| `--max-false-halt R` | Optional: maximum false-halt rate on `approve` cases |
| `--profile P` | Optional config profile (e.g. `medical`, `finance`) |
| `--output PATH` | Optional: write the JSON report for a CI artefact |

Exit codes: `0` pass, `1` threshold breached, `2` usage/data error.

!!! tip "Catch hallucinations, not just accuracy"
    `--min-accuracy` alone can be gamed by a guard that approves everything on a
    mostly-grounded set. Add `--min-catch-rate` to hold the guard's recall on the
    hallucination (`reject`) cases, and `--max-false-halt` to keep it from
    over-blocking grounded answers.

## The GitHub Action

The repository ships a composite action, so a workflow is a few lines:

```yaml
name: guardrail
on: [pull_request]

jobs:
  guardrail-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: anulum/director-ai@v1
        with:
          dataset: tests/guardrail_cases.jsonl
          min-accuracy: "0.9"
          min-catch-rate: "0.85"
          max-false-halt: "0.1"
```

Action inputs mirror the CLI flags, plus `extras` (the pip extra to install —
defaults to `nli` so real NLI scoring is available), `version` (a `director-ai`
version spec), `python-version`, and `output`.

!!! note "Heuristic vs model-backed"
    Without the `[nli]` extra and a knowledge base, scoring falls back to
    heuristics and will miss most hallucinations. The action installs `[nli]` by
    default; for grounded checks, ingest your facts first (see
    [KB ingestion](kb-ingestion.md)).

## The report

With `--output`, the gate writes a JSON report — counts, the metrics, the
breached thresholds, and per-case outcomes — suitable for upload as a CI
artefact or for trend tracking:

```json
{
  "total": 2, "correct": 2, "accuracy": 1.0,
  "catch_rate": 1.0, "false_halt_rate": 0.0,
  "passed": true, "failures": [],
  "outcomes": [{"case_id": "1", "expected": "approve", "predicted": "approve", "score": 0.98, "correct": true}]
}
```
