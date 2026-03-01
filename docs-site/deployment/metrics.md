# Metrics & Observability

Director-AI ships a zero-dependency Prometheus-compatible metrics collector.
All metrics use the `director_ai_` prefix.

## Metric Reference

### Counters

| Metric | Labels | Description |
|--------|--------|-------------|
| `reviews_total` | — | Total review requests processed |
| `reviews_approved` | — | Reviews that passed coherence threshold |
| `reviews_rejected` | — | Reviews that failed coherence threshold |
| `halts_total` | `reason` | Safety kernel halt events |
| `http_requests_total` | `method`, `endpoint`, `status` | HTTP requests by method/endpoint/status |

### Histograms

| Metric | Buckets | Description |
|--------|---------|-------------|
| `coherence_score` | 0.1–1.0 (step 0.1) | Coherence score distribution |
| `review_duration_seconds` | 0.01–10s | End-to-end review latency |
| `batch_size` | 1–1000 | Batch request sizes |
| `nli_inference_seconds` | 0.005–5s | Single NLI inference latency |
| `factual_retrieval_seconds` | 0.001–1s | RAG retrieval latency |
| `chunked_nli_seconds` | 0.01–30s | Chunked NLI scoring latency |
| `nli_premise_chunks` | 1–20 | Premise chunk count per scoring call |
| `nli_hypothesis_chunks` | 1–20 | Hypothesis chunk count per scoring call |
| `http_request_duration_seconds` | 0.005–10s | HTTP request duration |

### Gauges

| Metric | Description |
|--------|-------------|
| `active_requests` | In-flight requests |
| `nli_model_loaded` | 1 if NLI model is loaded |

## Prometheus Endpoint

```
GET /v1/metrics/prometheus
```

Output includes `# HELP` and `# TYPE` headers per metric family, `le="+Inf"` overflow bucket on histograms, and labeled counter lines.

## Kubernetes Scrape Config

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/v1/metrics/prometheus"
```

## Grafana PromQL Examples

```promql
# Request rate (5m window)
rate(director_ai_http_requests_total[5m])

# p99 review latency
histogram_quantile(0.99, rate(director_ai_review_duration_seconds_bucket[5m]))

# Error rate by endpoint
sum(rate(director_ai_http_requests_total{status=~"5.."}[5m]))
  / sum(rate(director_ai_http_requests_total[5m]))

# Coherence score distribution
histogram_quantile(0.5, rate(director_ai_coherence_score_bucket[5m]))

# Average premise chunks per call
rate(director_ai_nli_premise_chunks_sum[5m])
  / rate(director_ai_nli_premise_chunks_count[5m])
```

## Docker Compose Verification

```bash
# Start the server
docker compose up -d director-ai

# Verify Prometheus output
curl -s http://localhost:8080/v1/metrics/prometheus | head -20
# Expected: lines starting with # HELP, # TYPE, director_ai_*
```

## JSON Metrics

```
GET /v1/metrics
```

Returns all counters, histograms (count/total/mean/p50/p90/p99), and gauges as JSON.

## Python API

```python
from director_ai.core.metrics import metrics

metrics.inc("reviews_total")
metrics.inc("halts_total", label="hard_limit")
metrics.inc_labeled("http_requests_total", {"method": "GET", "status": "200"})
metrics.observe("coherence_score", 0.87)
metrics.gauge_set("nli_model_loaded", 1.0)

with metrics.timer("review_duration_seconds"):
    approved, score = scorer.review(query, response)
```
