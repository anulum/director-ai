# Medical Domain Cookbook

## Complete Working Example

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()  # empty — populate with your KB
store.add("aspirin children", "Aspirin should not be given to children under 16 due to Reye's syndrome risk.")
store.add("blood pressure", "Normal blood pressure is below 120/80 mmHg.")
store.add("diabetes diagnosis", "Type 2 diabetes is diagnosed when fasting glucose exceeds 126 mg/dL.")

scorer = CoherenceScorer(threshold=0.30, ground_truth_store=store)

# Correct claim → approved
approved, score = scorer.review("Is aspirin safe for children?",
    "Aspirin should not be given to children under 16 due to Reye's syndrome risk.")
print(f"Correct: approved={approved}, score={score.score:.2f}")

# Incorrect claim → rejected
approved, score = scorer.review("Is aspirin safe for children?",
    "Aspirin is perfectly safe for children of all ages.")
print(f"Wrong:   approved={approved}, score={score.score:.2f}")
if score.evidence:
    for chunk in score.evidence.chunks:
        print(f"  Evidence: {chunk.text}")
```

## Configuration

```python
scorer = CoherenceScorer(
    threshold=0.30,    # Measured on PubMedQA (F1=59.9% at this threshold)
    soft_limit=0.35,   # Conservative warning zone
    use_nli=True,
    nli_model="lytang/MiniCheck-DeBERTa-L",
    ground_truth_store=store,
)
```

## Knowledge Base Setup

```python
store = VectorGroundTruthStore()
store.ingest([
    "Aspirin should not be given to children under 16 due to Reye's syndrome risk.",
    "Normal blood pressure is below 120/80 mmHg.",
    "Type 2 diabetes is diagnosed when fasting glucose exceeds 126 mg/dL.",
])
```

## Safety Pattern

```python
from director_ai import CoherenceAgent

agent = CoherenceAgent(
    use_nli=True,
    fallback="retrieval",  # Always fall back to verified sources
)

# Add disclaimer for all medical content
agent.disclaimer_prefix = "[Medical information — verify with a healthcare provider] "
```

## Risk Reduction (Illustrative Estimates)

!!! warning "These are illustrative industry estimates, not measured results from Director-AI deployments."

| Metric | Without guardrail (industry baseline) | With Director-AI (threshold=0.30) |
|--------|--------------------|---------------------------------|
| Hallucinated dosage/contraindication rate | 8–15% (model-dependent, per published LLM medical benchmarks) | < 1% estimated with verified KB (not yet validated on clinical data) |
| Clinician review time per AI response | 45 sec (read + verify manually) | 10 sec (review evidence chunk) — estimated |
| Unsafe recommendation reach (before catch) | 100% of users | 0% (mid-stream halt) — estimated |

**Cost model (illustrative, not measured):** At 500 medical queries/day, reducing manual review from 45s to 10s would save ~4.8 clinician-hours/day (~$175K/year at $150/hr). These figures are planning estimates based on industry rates, not measured deployment data. Validate on your own workload before budgeting.

## Key Considerations

- **Never use `fallback=None`** for medical — always provide verified sources
- **Log all rejections** with full evidence for clinical review
- **Tune threshold on your data** — CoherenceScorer scores cluster 0.25–0.35; start at 0.30 and adjust based on your domain's FPR
- **Regular KB updates** — medical guidelines change frequently
