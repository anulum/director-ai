# FrontierFail Seed Packet

The FrontierFail packet is an intake and regression seed for production-failure
benchmarking. It is not an externally validated benchmark and must not be used
as a public production-failure score.

## Files

| File | Purpose |
|------|---------|
| `benchmarks/frontierfail_seed_packet.toml` | Packet metadata, claim boundary, required categories, and public-incident diversity floors |
| `benchmarks/frontierfail_cases.jsonl` | Synthetic regression cases plus sourced public-incident intake rows |
| `tools/validate_frontierfail_packet.py` | Schema, provenance, and benchmark-eligibility gate |

## Current Boundary

The packet contains deterministic `synthetic_regression` rows for engineering
regression coverage and sourced `public_incident` intake rows for early
production-failure taxonomy coverage. It is still not an externally validated
benchmark and must not be reported as a public FrontierFail score.

Rows may become benchmark eligible only when they are sourced from a sanitized
production report or public incident with reviewable evidence. The validator
rejects synthetic rows marked as benchmark eligible, requires public incidents
to include publisher, title, and access-date metadata, and enforces
category/domain/publisher/evidence-reference diversity for benchmark-eligible
public incidents. Duplicate public-incident evidence references are rejected.

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

The validator is covered through the production subprocess path in
`tests/test_frontierfail_packet_real_surface.py`, which runs the checked-in
packet and a temporary broken packet through the same CLI entry point operators
use in CI or release checks. The lower-level
`tests/test_frontierfail_packet.py` suite remains a branch guard for schema and
row-level failures.

The gate fails when:

- the packet claims public benchmark eligibility;
- required categories have zero cases;
- synthetic seed rows are marked benchmark eligible;
- sourced benchmark-eligible rows lack reviewable evidence;
- public incident rows lack publisher, title, or access-date metadata;
- public incident coverage misses the configured category, domain, publisher,
  or evidence-reference diversity floors;
- public incident evidence references are duplicated;
- required row fields are missing or empty;
- expected decisions fall outside the supported decision set.

This keeps FrontierFail useful for engineering regression work without blurring
the line between seed fixtures and independently validated production-failure
benchmarks.
