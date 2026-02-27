# Metrics & Observability

## Built-in Metrics

```python
from director_ai.core.metrics import metrics

# Increment counters
metrics.inc("reviews_total")
metrics.inc("halts_total", label="hard_limit")

# Observe histograms
metrics.observe("coherence_score", 0.87)
metrics.observe("review_duration_seconds", 0.042)

# Set gauges
metrics.gauge_set("nli_model_loaded", 1.0)

# Timer context manager
with metrics.timer("review_duration_seconds"):
    approved, score = scorer.review(query, response)
```

## Prometheus Export

```python
# Returns Prometheus text exposition format
print(metrics.prometheus_format())
```

```
director_ai_reviews_total 1523
director_ai_reviews_approved 1401
director_ai_reviews_rejected 122
director_ai_halts_total{reason="hard_limit"} 89
director_ai_halts_total{reason="window_avg"} 33
director_ai_coherence_score_bucket{le="0.5"} 122
director_ai_coherence_score_bucket{le="0.6"} 245
director_ai_coherence_score_count 1523
director_ai_coherence_score_sum 1298.45
director_ai_review_duration_seconds_bucket{le="0.05"} 1200
director_ai_review_duration_seconds_count 1523
director_ai_active_requests 3
```

## FastAPI Integration

```python
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from director_ai.core.metrics import metrics

app = FastAPI()

@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    return metrics.prometheus_format()
```

## JSON Metrics

```python
@app.get("/metrics/json")
def json_metrics():
    return metrics.get_metrics()
```
