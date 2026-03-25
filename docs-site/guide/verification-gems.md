# Verification Gems

Director-AI v3.10 includes three standalone verification modules that work
without NLI models. All are stdlib-only with zero external dependencies.

## Numeric Verification

Catches arithmetic errors, impossible dates, and probabilities outside [0, 100%].

```python
from director_ai import verify_numeric

result = verify_numeric(
    "Revenue grew 50% from $100 to $120. "
    "Founded in 2035."
)
print(result.valid)        # False
print(result.error_count)  # 1 (50% of 100 = 50, not 20)
for issue in result.issues:
    print(f"  {issue.issue_type}: {issue.description}")
```

**What it checks:**

- Percentage arithmetic ("grew 15% from X to Y" — is the math right?)
- Date ordering (birth < death, founding < present)
- Probability bounds (no negative or >100% probabilities)
- Order of magnitude (Earth population, speed of light)
- Internal consistency (same total referenced with different values)

### REST API

```bash
curl -X POST http://localhost:8080/v1/verify/numeric \
  -H "Content-Type: application/json" \
  -d '{"text": "Revenue grew 50% from $100 to $120."}'
```

Response:
```json
{
  "claims_found": 5,
  "issues": [{"issue_type": "arithmetic", "description": "...", "severity": "error", "context": "..."}],
  "valid": false,
  "error_count": 1,
  "warning_count": 0
}
```

## Reasoning Chain Verification

Detects non-sequiturs, circular reasoning, and unsupported leaps in
chain-of-thought responses.

```python
from director_ai import verify_reasoning_chain

result = verify_reasoning_chain(
    "Step 1: All birds can fly. "
    "Step 2: Penguins are birds. "
    "Step 3: Therefore, the economy is growing."
)
print(result.chain_valid)   # False
print(result.issues_found)  # 1
for v in result.verdicts:
    print(f"  Step {v.step_index}: {v.verdict} ({v.confidence:.2f})")
```

**Verdict types:** `supported`, `non_sequitur`, `unsupported_leap`, `circular`

### REST API

```bash
curl -X POST http://localhost:8080/v1/verify/reasoning \
  -H "Content-Type: application/json" \
  -d '{"text": "Step 1: A is true. Step 2: Therefore B."}'
```

## Temporal Freshness Scoring

Flags claims that may rely on stale knowledge — positions, statistics,
records, and "current" references.

```python
from director_ai import score_temporal_freshness

result = score_temporal_freshness("The CEO of Apple is Tim Cook.")
print(result.has_temporal_claims)     # True
print(result.overall_staleness_risk)  # 0.8 (positions change)
for claim in result.claims:
    print(f"  {claim.claim_type}: {claim.text} (risk: {claim.staleness_risk:.2f})")
```

**Claim types:** `position`, `statistic`, `record`, `current_reference`

### REST API

```bash
curl -X POST http://localhost:8080/v1/temporal-freshness \
  -H "Content-Type: application/json" \
  -d '{"text": "The CEO of Apple is Tim Cook."}'
```

Response:
```json
{
  "claims": [{"text": "CEO of Apple is Tim Cook", "claim_type": "position", "staleness_risk": 0.8, "reason": "..."}],
  "overall_staleness_risk": 0.8,
  "has_temporal_claims": true,
  "stale_claim_count": 1
}
```
