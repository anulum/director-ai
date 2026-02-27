# Medical Domain Cookbook

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
