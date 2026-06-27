# PINT Replication Packet

The PINT replication packet prepares Director-AI for the Prompt Injection Test
evaluation workflow without claiming an official PINT score.

The upstream benchmark framework is public, but the held-back evaluation data
is controlled to reduce contamination and overfitting. Director-AI therefore
ships only a local adapter contract and synthetic smoke fixture until an
official run or approved data export is available.

## Files

| File | Purpose |
|------|---------|
| `benchmarks/pint_replication_packet.toml` | Upstream source, adapter contract, claim boundary |
| `benchmarks/pint_seed_cases.jsonl` | Synthetic seed cases for local smoke testing |
| `tools/validate_pint_replication_packet.py` | Schema and score-claim boundary gate |
| `tools/run_pint_seed_smoke.py` | Non-public seed-smoke runner for the production sanitizer path |
| `tools/run_pint_official_export.py` | Local export runner that records dataset hashes without approving public score claims |

## Boundary

`public_score_claim = false` is mandatory. The seed rows use
`source_type = "synthetic_seed"` and `benchmark_eligible = false`.

The validator rejects:

- seed packets marked as public score claims;
- synthetic seed rows marked benchmark eligible;
- official export rows marked benchmark eligible without separate private
  validation evidence;
- missing attack-category coverage;
- missing positive or benign hard-negative label coverage.

## Adapter Contract

The replication packet records `detector_contract = "text_to_boolean"`:

```text
input text -> true if prompt injection should be blocked, false otherwise
```

This matches Director-AI's current input-side sanitizer and output-side
injection detector integration points while avoiding any dependency on private
upstream test rows.

## Validation

```bash
uv run --frozen python tools/validate_pint_replication_packet.py .
```

The validator is covered through the production subprocess path in
`tests/test_pint_replication_packet_real_surface.py`, which runs the checked-in
packet and a temporary broken packet through the same CLI entry point operators
use in CI or release checks. The lower-level
`tests/test_pint_replication_packet.py` suite remains a branch guard for schema,
language-diversity, label, and claim-boundary failures.

The seed-smoke runner is covered through
`tests/test_pint_seed_smoke_runner_real_surface.py`, which executes
`tools/run_pint_seed_smoke.py` with the checked-in packet and a temporary output
path. The resulting JSON must keep `public_score_claim`, `official_pint_score`,
and `benchmark_eligible` false and must omit raw prompt text from per-case rows.

The official-export runner is covered through
`tests/test_pint_official_export_runner_real_surface.py`, which executes
`tools/run_pint_official_export.py` with a local PINT-format export and a
temporary output path. The result records the export path and SHA-256 digest,
keeps `public_score_claim` false, and omits raw prompt text from per-case rows.

Use this gate before adding PINT-related results to public benchmark tables.
Official score claims require a separate evidence packet with the upstream run
environment, raw outputs, metric mapping, and access approval.
