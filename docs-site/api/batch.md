# BatchProcessor

Concurrent batch processing for `CoherenceScorer` and `CoherenceAgent`. Processes multiple prompts in parallel with configurable concurrency and progress tracking.

## Usage

```python
from director_ai import CoherenceAgent
from director_ai.core.batch import BatchProcessor

agent = CoherenceAgent(use_nli=True)
processor = BatchProcessor(agent, max_concurrency=8)

result = processor.process_batch([
    "What is the capital of France?",
    "What is the speed of light?",
    "When was Python released?",
])

print(f"Succeeded: {result.succeeded}/{result.total}")
print(f"Duration: {result.duration_seconds:.1f}s")
for r in result.results:
    print(f"  output={r.output}, halted={r.halted}")
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `CoherenceAgent \| CoherenceScorer` | required | Backend instance for processing |
| `max_concurrency` | `int` | `4` | Maximum parallel workers |
| `item_timeout` | `float` | `60.0` | Per-item timeout in seconds |

## Methods

### process_batch()

```python
result = processor.process_batch(
    prompts: list[str],
    tenant_id: str = "",
) -> BatchResult
```

### process_batch_async()

```python
result = await processor.process_batch_async(
    prompts: list[str],
    tenant_id: str = "",
    max_concurrency: int | None = None,
) -> BatchResult
```

## BatchResult

| Field | Type | Description |
|-------|------|-------------|
| `results` | `list[ReviewResult \| tuple[bool, CoherenceScore]]` | Per-item results |
| `errors` | `list[tuple[int, str]]` | Failed items: `(index, reason)` |
| `total` | `int` | Total items submitted |
| `succeeded` | `int` | Items that completed |
| `failed` | `int` | Items that errored |
| `duration_seconds` | `float` | Wall-clock time for the batch |

## Scorer-Level Batching

For scoring prompt/response pairs without the agent orchestration layer, use `CoherenceScorer.review_batch()`:

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(threshold=0.3, use_nli=True)

items = [
    ("What is 2+2?", "The answer is 4."),
    ("Capital of France?", "Paris is in Germany."),
]
results = scorer.review_batch(items)
for approved, cs in results:
    print(f"approved={approved}, score={cs.score:.3f}")
```

`review_batch()` batches logical and factual NLI through `score_batch()` (2 GPU forward passes total) when NLI is available. Dialogue items fall back to sequential `review()`. Without NLI, all items are scored sequentially via heuristics.

## Full API

::: director_ai.core.runtime.batch.BatchProcessor
