# BatchProcessor

Concurrent batch processing for `CoherenceScorer` and `CoherenceAgent`. Processes multiple prompts in parallel with configurable concurrency and progress tracking.

## Usage

```python
from director_ai import CoherenceAgent
from director_ai.core.batch import BatchProcessor

agent = CoherenceAgent(use_nli=True)
processor = BatchProcessor(agent, max_concurrency=8)

results = processor.process_batch([
    "What is the capital of France?",
    "What is the speed of light?",
    "When was Python released?",
])

for result in results:
    print(f"{result.query}: approved={result.approved}, score={result.score:.3f}")
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `CoherenceAgent` | required | Agent instance for processing |
| `max_concurrency` | `int` | `4` | Maximum parallel requests |

## Methods

### process_batch()

```python
results = processor.process_batch(
    prompts: list[str],
) -> list[BatchResult]
```

### process_batch_async()

```python
results = await processor.process_batch_async(
    prompts: list[str],
    max_concurrency: int = 8,
) -> list[BatchResult]
```

## BatchResult

| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Original prompt |
| `response` | `str` | Best response (or fallback) |
| `approved` | `bool` | Whether response passed coherence |
| `score` | `float` | Coherence score |
| `evidence` | `ScoringEvidence \| None` | Retrieved evidence |

## Scorer-Level Batching

For bulk NLI inference without the agent orchestration layer, use `CoherenceScorer.review_batch()` directly:

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(threshold=0.6, use_nli=True)

items = [
    ("What is 2+2?", "The answer is 4."),
    ("Capital of France?", "Paris is in Germany."),
]
results = scorer.review_batch(items)
```

This runs 2 GPU kernel calls total instead of 2×N.

## Full API

::: director_ai.core.batch.BatchProcessor
