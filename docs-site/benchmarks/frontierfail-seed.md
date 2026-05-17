# FrontierFail Seed Packet

The FrontierFail packet is an intake and regression seed for production-failure
benchmarking. It is not an externally validated benchmark and must not be used
as a public production-failure score.

## Files

| File | Purpose |
|------|---------|
| `benchmarks/frontierfail_seed_packet.toml` | Packet metadata, claim boundary, required categories |
| `benchmarks/frontierfail_cases.jsonl` | Seed regression cases |
| `tools/validate_frontierfail_packet.py` | Schema, provenance, and benchmark-eligibility gate |

## Current Boundary

The seed cases use `source_type = "synthetic_regression"` and
`benchmark_eligible = false`. They exist to keep the taxonomy, row schema, and
downstream evaluation hooks stable while real sourced production cases are
collected and redacted.

Rows may become benchmark eligible only when they are sourced from a sanitized
production report or public incident with reviewable evidence. The validator
rejects synthetic rows marked as benchmark eligible.

## Required Categories

- `numeric_contradiction`
- `fabricated_policy`
- `unsupported_citation`
- `cross_turn_contradiction`
- `retrieval_misattribution`

## Validation

```bash
uv run --frozen python tools/validate_frontierfail_packet.py .
```

The gate fails when:

- the packet claims public benchmark eligibility;
- required categories have zero cases;
- synthetic seed rows are marked benchmark eligible;
- sourced benchmark-eligible rows lack reviewable evidence;
- required row fields are missing or empty;
- expected decisions fall outside the supported decision set.

This keeps FrontierFail useful for engineering regression work without blurring
the line between seed fixtures and independently validated production-failure
benchmarks.
