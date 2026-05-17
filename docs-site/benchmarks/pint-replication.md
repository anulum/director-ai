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

Use this gate before adding PINT-related results to public benchmark tables.
Official score claims require a separate evidence packet with the upstream run
environment, raw outputs, metric mapping, and access approval.
