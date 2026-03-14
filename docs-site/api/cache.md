# Score Cache

Thread-safe LRU + TTL cache for coherence scores. Avoids redundant NLI inference on repeated prompt/response pairs. Reduces GPU cost by 60-80% in streaming workloads.

## Usage

Pass `cache_size` to `CoherenceScorer` to enable transparent caching:

```python
from director_ai import CoherenceScorer

scorer = CoherenceScorer(
    cache_size=2048,  # max entries
    cache_ttl=300.0,  # 5-minute TTL
    use_nli=True,
)

# First call: NLI inference (~15ms)
approved, score = scorer.review("What is 2+2?", "4.")

# Second call with same inputs: cache hit (<0.1ms)
approved, score = scorer.review("What is 2+2?", "4.")

# Monitor cache performance
print(f"Hit rate: {scorer.cache.hit_rate:.1%}")
print(f"Size: {scorer.cache.size}")
```

## ScoreCache

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_size` | `int` | `1024` | Maximum cached entries |
| `ttl_seconds` | `float` | `300.0` | Time-to-live per entry |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `hit_rate` | `float` | Ratio of hits to total lookups |
| `size` | `int` | Current number of cached entries |
| `hits` | `int` | Total cache hits |
| `misses` | `int` | Total cache misses |

### Methods

- `get(key) -> CoherenceScore | None` — retrieve cached score
- `put(key, score)` — store a score
- `invalidate(key)` — remove a specific entry
- `clear()` — flush all entries

## Cache Key

The cache key is derived from `(query, response)` text content. TTL-expired and LRU-evicted entries are cleaned lazily on access.

## Full API

::: director_ai.core.cache.ScoreCache
