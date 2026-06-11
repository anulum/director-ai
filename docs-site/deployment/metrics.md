<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-Class AI — Metrics and observability reference
-->

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
| `feedback_total` | `outcome` | Human feedback events used for calibration and false-positive tracking |
| `retune_recommendations_total` | none | Retune recommendations emitted by safety operations |
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

HTTP request metrics use route templates for `endpoint` labels, such as
`/v1/sessions/{session_id}` and `/v1/tenants/{tenant_id}/facts`, not raw path
values. This keeps Prometheus cardinality bounded and prevents tenant, session,
or document identifiers from entering metric labels. Authentication failures are
also counted with the same route-template label contract.

### Gauges

| Metric | Description |
|--------|-------------|
| `active_requests` | In-flight requests |
| `nli_model_loaded` | 1 if NLI model is loaded |
| `kb_stale_sources` | Knowledge sources that need refresh or review |
| `retune_recommended` | 1 when recent feedback indicates retuning is due |

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
metrics.inc_labeled("feedback_total", {"outcome": "false_positive"})
metrics.inc("retune_recommendations_total")
metrics.inc_labeled("http_requests_total", {"method": "GET", "status": "200"})
metrics.observe("coherence_score", 0.87)
metrics.gauge_set("nli_model_loaded", 1.0)
metrics.gauge_set("kb_stale_sources", 2)
metrics.gauge_set("retune_recommended", 1)

with metrics.timer("review_duration_seconds"):
    approved, score = scorer.review(query, response)
```

## Sustainability Policy Events

The sustainability scoring adapter returns tenant-safe `GuardDecision` and
`SafetyEvent` payloads rather than raw metrics. Deployments that mirror these
events into Prometheus, OpenTelemetry, or a billing warehouse should export only
aggregate fields:

- decision, reason, and policy id
- request-count and total-unit summaries
- energy, carbon, and cost estimates with provenance
- threshold alert names
- hardware profile id

Do not export raw prompts, completions, media, credentials, API keys, or access
tokens through sustainability telemetry. Hardware profile values must be marked
as measured, configured, or projected so operators can distinguish instrumented
energy measurements from planning estimates.
