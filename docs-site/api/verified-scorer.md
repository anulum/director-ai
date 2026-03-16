# Verified Scorer

Sentence-level multi-signal fact verification. Decomposes response into claims, matches each to the best source sentence, and checks 5 independent signals.

## Signals

| # | Signal | What It Detects |
|---|--------|-----------------|
| 1 | **NLI entailment** | Semantic contradiction between claim and source |
| 2 | **Entity consistency** | Named entity mismatches (Paris ≠ Berlin) |
| 3 | **Numerical consistency** | Number mismatches ($99 ≠ $49, 30 days ≠ 60 days) |
| 4 | **Negation detection** | Polarity flips ("supports" vs "does not support") |
| 5 | **Traceability** | Fabricated content not present in source |

## Usage

```python
from director_ai.core.verified_scorer import VerifiedScorer

vs = VerifiedScorer(nli_scorer=nli)  # or None for heuristic-only
result = vs.verify(response="The plan costs $99.", source="Pricing: $49/month.")

print(result.approved)           # False
print(result.confidence)         # "high"
print(result.contradicted_count) # 1
for claim in result.claims:
    print(f"  [{claim.verdict}] {claim.claim}")
    print(f"    Source: {claim.matched_source}")
    print(f"    NLI: {claim.nli_divergence:.2f}, Numbers: {claim.numerical_match}")
```

## REST API

```bash
curl -X POST http://localhost:8080/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is the price?", "response": "The plan costs $99/month."}'
```

## Verdicts

| Verdict | Meaning | Confidence Basis |
|---------|---------|-----------------|
| `supported` | Claim is consistent with source | 2+ signals agree |
| `contradicted` | Claim conflicts with source | 2+ signals agree |
| `fabricated` | Claim content not traceable to source | Traceability < 15% |
| `unverifiable` | Insufficient signal agreement | Signals disagree |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nli_scorer` | `None` | NLI model for primary signal. `None` = heuristic only. |
| `nli_threshold` | `0.65` | NLI divergence above this = contradiction |
| `support_threshold` | `0.35` | NLI divergence below this = supported |
| `min_confidence` | `0.4` | Below this, verdict is unverifiable |
