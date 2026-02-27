# Evidence & Fallback

## Evidence Return

Every `CoherenceScore` carries evidence explaining the scoring decision:

```python
approved, score = scorer.review(query, response)

if score.evidence:
    print(f"NLI score: {score.evidence.nli_score:.3f}")
    print(f"Premise: {score.evidence.nli_premise[:100]}")
    print(f"Hypothesis: {score.evidence.nli_hypothesis[:100]}")
    for chunk in score.evidence.chunks:
        print(f"  [{chunk.distance:.3f}] {chunk.text[:80]}")
```

### ScoringEvidence Fields

| Field | Type | Description |
|-------|------|-------------|
| `chunks` | list[EvidenceChunk] | Top-K retrieved context chunks |
| `nli_premise` | str | Context fed to NLI model |
| `nli_hypothesis` | str | Response being scored |
| `nli_score` | float | Raw NLI divergence (0=entailment, 1=contradiction) |

## Fallback Modes

When all candidates are rejected, Director-AI can recover instead of hard-stopping:

### Retrieval Fallback
Returns verified context from the knowledge base:

```python
agent = CoherenceAgent(fallback="retrieval")
result = agent.process("What is the refund policy?")
# result.output = "Based on verified sources: Refunds within 30 days..."
# result.fallback_used = True
```

### Disclaimer Fallback
Prepends a confidence warning to the best-rejected candidate:

```python
agent = CoherenceAgent(fallback="disclaimer")
result = agent.process("What is the refund policy?")
# result.output = "Note: This response could not be fully verified. ..."
# result.fallback_used = True
```

## Soft Warning Zone

Scores between `threshold` and `soft_limit` pass but get flagged:

```python
scorer = CoherenceScorer(threshold=0.5, soft_limit=0.65)
approved, score = scorer.review(query, response)

if score.warning:
    # Inject disclaimer prefix in streaming
    output = f"[Confidence: moderate] {response}"
```
