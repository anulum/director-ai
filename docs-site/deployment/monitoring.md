# Monitoring & Observability

Director-AI exposes Prometheus metrics at `/v1/metrics/prometheus`. This guide wires up Prometheus + Grafana alongside the application.

## Docker Compose

```yaml
services:
  director-ai:
    image: ghcr.io/anulum/director-ai:latest
    ports:
      - "8080:8080"
    environment:
      - DIRECTOR_METRICS_ENABLED=true

  prometheus:
    image: prom/prometheus:v2.53.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - director-ai

  grafana:
    image: grafana/grafana:11.1.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
```

## Prometheus Scrape Config

```yaml
# prometheus.yml
scrape_configs:
  - job_name: director-ai
    scrape_interval: 15s
    metrics_path: /v1/metrics/prometheus
    static_configs:
      - targets: ["director-ai:8080"]
```

## Key Metrics

| Metric | Type | What it tells you |
|--------|------|-------------------|
| `director_ai_reviews_total` | counter | Total review throughput |
| `director_ai_coherence_score` | histogram | Score distribution across requests |
| `director_ai_review_duration_seconds` | histogram | End-to-end review latency |
| `director_ai_halts_total` | counter | Safety kernel halt events (labeled by `reason`) |
| `director_ai_active_requests` | gauge | Current in-flight requests |
| `director_ai_nli_inference_seconds` | histogram | NLI model inference latency |

See [Metrics Reference](metrics.md) for the full list.

## Alert Rules

```yaml
# prometheus-alerts.yml
groups:
  - name: director-ai
    rules:
      - alert: HighP99Latency
        expr: histogram_quantile(0.99, rate(director_ai_review_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p99 review latency exceeds 2s"

      - alert: HighHaltRate
        expr: rate(director_ai_halts_total[5m]) / rate(director_ai_reviews_total[5m]) > 0.10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Halt rate exceeds 10% of reviews"

      - alert: HighErrorRate
        expr: >
          sum(rate(director_ai_http_requests_total{status=~"5.."}[5m]))
          / sum(rate(director_ai_http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "HTTP 5xx error rate exceeds 1%"
```

Load alerts in `prometheus.yml`:

```yaml
rule_files:
  - prometheus-alerts.yml
```

## Grafana Dashboard

1. Open `http://localhost:3000` (admin / admin).
2. Add data source: Prometheus at `http://prometheus:9090`.
3. Import or create a dashboard with these panels:

**Row 1 — Throughput:**

- Reviews/sec: `rate(director_ai_reviews_total[5m])`
- Halt rate: `rate(director_ai_halts_total[5m]) / rate(director_ai_reviews_total[5m])`

**Row 2 — Latency:**

- p50: `histogram_quantile(0.50, rate(director_ai_review_duration_seconds_bucket[5m]))`
- p99: `histogram_quantile(0.99, rate(director_ai_review_duration_seconds_bucket[5m]))`

**Row 3 — Score Distribution:**

- Median coherence: `histogram_quantile(0.50, rate(director_ai_coherence_score_bucket[5m]))`
- Rejection rate: `rate(director_ai_reviews_rejected[5m]) / rate(director_ai_reviews_total[5m])`

**Row 4 — Infrastructure:**

- Active requests: `director_ai_active_requests`
- NLI model loaded: `director_ai_nli_model_loaded`
