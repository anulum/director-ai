# Types & Dataclasses

Shared data types returned by scorer, agent, and streaming methods. All are frozen or standard dataclasses.

## CoherenceScore

The primary return type from `CoherenceScorer.review()` and `score()`.

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Composite coherence score (0.0–1.0) |
| `approved` | `bool` | Whether `score >= threshold` |
| `h_logical` | `float` | Logical divergence (NLI contradiction probability) |
| `h_factual` | `float` | Factual divergence (RAG deviation) |
| `warning` | `bool` | `True` if score is between `threshold` and `soft_limit` |
| `evidence` | `ScoringEvidence \| None` | Retrieved evidence and scoring details |
| `strict_mode_rejected` | `bool` | `True` if rejected because NLI was unavailable in strict mode |
| `cross_turn_divergence` | `float \| None` | Cross-turn NLI score (set when session context exists) |

```python
approved, score = scorer.review(query, response)

print(f"Score: {score.score:.3f}")
print(f"H_logical: {score.h_logical:.3f}")
print(f"H_factual: {score.h_factual:.3f}")
print(f"Warning: {score.warning}")

if score.evidence:
    for chunk in score.evidence.chunks:
        print(f"  Source: {chunk.text[:80]}")
```

---

## ReviewResult {: #reviewresult }

Return type from `CoherenceAgent.process()`.

| Field | Type | Description |
|-------|------|-------------|
| `output` | `str` | Best approved response (or fallback) |
| `coherence` | `CoherenceScore \| None` | Coherence score of the response |
| `halted` | `bool` | Whether safety kernel halted |
| `candidates_evaluated` | `int` | Number of candidates generated |
| `fallback_used` | `bool` | Whether a fallback was activated |
| `halt_evidence` | `HaltEvidence \| None` | Structured halt reason |

---

## ScoringEvidence {: #scoringevidence }

Evidence collected during scoring — retrieved KB chunks, NLI details, and attribution.

| Field | Type | Description |
|-------|------|-------------|
| `chunks` | `list[EvidenceChunk]` | Top-K retrieved chunks |
| `nli_premise` | `str` | NLI premise text used |
| `nli_hypothesis` | `str` | NLI hypothesis text used |
| `nli_score` | `float` | Raw NLI divergence score |
| `chunk_scores` | `list[float] \| None` | Per-chunk NLI scores |
| `premise_chunk_count` | `int` | Number of premise chunks |
| `hypothesis_chunk_count` | `int` | Number of hypothesis chunks |
| `attributions` | `list[ClaimAttribution] \| None` | Per-claim source attribution |
| `token_count` | `int` | NLI token consumption |
| `estimated_cost_usd` | `float` | Estimated NLI inference cost |
| `claim_coverage` | `float \| None` | Fraction of claims supported by source |
| `per_claim_divergences` | `list[float] \| None` | Per-claim divergence scores |
| `claims` | `list[str] \| None` | Decomposed atomic claims |

---

## EvidenceChunk {: #evidencechunk }

A single chunk of retrieved evidence.

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Chunk text content |
| `distance` | `float` | Similarity distance (lower = more relevant) |
| `source` | `str \| None` | Source identifier |

---

## ClaimAttribution {: #claimattribution }

Maps a summary claim to its source sentence.

| Field | Type | Description |
|-------|------|-------------|
| `claim` | `str` | The atomic claim |
| `claim_index` | `int` | Index of the claim in the decomposed list |
| `source_sentence` | `str` | Best-matching source sentence |
| `source_index` | `int` | Index of the source sentence |
| `divergence` | `float` | NLI divergence score (lower = better support) |
| `supported` | `bool` | Whether the claim is supported |

---

## HaltEvidence {: #haltevidence }

Structured halt reason with evidence chunks.

| Field | Type | Description |
|-------|------|-------------|
| `reason` | `str` | Halt mechanism that triggered |
| `last_score` | `float` | Coherence score at halt point |
| `evidence_chunks` | `list[EvidenceChunk]` | Contradicting chunks |
| `suggested_action` | `str` | Recommended action (e.g., "retry with KB context") |

```python
if session.halt_evidence_structured:
    ev = session.halt_evidence_structured
    print(f"Reason: {ev.reason}")
    print(f"Score: {ev.last_score:.3f}")
    for chunk in ev.evidence_chunks:
        print(f"  {chunk.text[:80]} (distance={chunk.distance:.3f})")
```

---

## Full API

::: director_ai.core.types.CoherenceScore

::: director_ai.core.types.ReviewResult

::: director_ai.core.types.ScoringEvidence

::: director_ai.core.types.EvidenceChunk

::: director_ai.core.types.ClaimAttribution

::: director_ai.core.types.HaltEvidence
