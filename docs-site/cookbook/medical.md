# Medical Domain Cookbook

## Complete Working Example

```python
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("aspirin children", "Aspirin should not be given to children under 16 due to Reye's syndrome risk.")
store.add("blood pressure", "Normal blood pressure is below 120/80 mmHg.")
store.add("diabetes diagnosis", "Type 2 diabetes is diagnosed when fasting glucose exceeds 126 mg/dL.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

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
    threshold=0.75,    # High threshold for medical safety
    soft_limit=0.85,   # Conservative warning zone
    use_nli=True,
    nli_model="lytang/MiniCheck-DeBERTa-L",
    ground_truth_store=store,
)
```

## Knowledge Base Setup

```python
store = VectorGroundTruthStore(auto_index=False)
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

## Key Considerations

- **Never use `fallback=None`** for medical — always provide verified sources
- **Log all rejections** with full evidence for clinical review
- **Threshold 0.75+** — false negatives in medical context are dangerous
- **Regular KB updates** — medical guidelines change frequently
