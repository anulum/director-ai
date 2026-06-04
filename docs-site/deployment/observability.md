# Observability Pack

*Added in v3.11.0*

Pre-built Grafana dashboards and Prometheus alert rules live at
`deploy/observability/`.

## Grafana Dashboard

Import `deploy/observability/grafana-dashboard.json` into Grafana.

For halt operations, import
`deploy/observability/safety-ops-grafana-dashboard.json`.

### Panels (9)

| Panel | Type | Metric |
|-------|------|--------|
| Reviews / minute | timeseries | `rate(director_reviews_total[1m])` |
| Hallucination rate | timeseries | `rate(director_hallucinations_total[1m]) / rate(director_reviews_total[1m])` |
| Review latency p50/p95/p99 | timeseries | `histogram_quantile(0.50/0.95/0.99, ...)` |
| Coherence score distribution | histogram | `director_coherence_score` |
| Active streaming sessions | stat | `director_streaming_active_sessions` |
| Streaming halts / min | stat | `rate(director_streaming_halts_total[1m])` |
| KB queries / min | stat | `rate(director_knowledge_queries_total[1m])` |
| Error rate | stat | `rate(director_errors_total[5m])` |
| Drift score (7-day) | timeseries | `director_drift_score` |

## Prometheus Alerts

Add `deploy/observability/prometheus-alerts.yml` to your Prometheus configuration.

For halt-rate, false-positive-rate, stale-knowledge, and retune alerts, also add
`deploy/observability/safety-ops-prometheus-rules.yml`.

### Alert Rules (6)

| Alert | Condition | Severity |
|-------|-----------|----------|
| `HighHallucinationRate` | > 15% for 5 min | warning |
| `ReviewLatencyHigh` | p95 > 500ms for 5 min | warning |
| `StreamingHaltSpike` | > 10 halts/min for 2 min | critical |
| `DriftDetected` | drift score > 0.2 for 15 min | warning |
| `ErrorRateHigh` | > 5 errors/min for 5 min | critical |
| `KBQueryFailures` | > 5% failure rate for 5 min | warning |

## Setup

### Prometheus

Ensure Director-AI exposes metrics at `/v1/metrics/prometheus` (enabled via
`DIRECTOR_METRICS_ENABLED=true`):

```bash
director-ai serve --port 8080
# Metrics at http://localhost:8080/v1/metrics/prometheus
```

When API-key auth is enabled, scrape with `Authorization: Bearer <api-key>` or
the `X-API-Key` header.

### Grafana

1. Add your Prometheus as a data source
2. Import → Upload JSON → select `grafana-dashboard.json`
3. Select the Prometheus data source

## Safety Operations Mixin

The safety operations mixin tracks:

- halt rate: `director_ai_halts_total / director_ai_reviews_total`
- false-positive feedback rate: `director_ai_feedback_total{outcome="false_positive"} / director_ai_halts_total`
- stale knowledge: `director_ai_kb_stale_sources`
- retune guidance: `director_ai_retune_recommended` and
  `director_ai_retune_recommendations_total`

It pairs with `director-ai safety-dashboard` for recent evidence review and
labelled-feedback retuning.

## Tenant-Safe Operations Report

The dashboard parser can also produce a machine-readable operations packet for
review boards and deployment gates:

```python
from director_ai.ui import (
    ComplianceExportRef,
    TrustControl,
    build_observability_operations_report,
)

events_jsonl = open("safety_events.jsonl", encoding="utf-8").read()
report = build_observability_operations_report(
    events_jsonl,
    controls=[
        TrustControl(
            control="External security test",
            status="warning",
            evidence_ref="security/external-review-ticket.md",
        ),
    ],
    compliance_exports=[
        ComplianceExportRef(
            standard="EU AI Act Article 15",
            name="30-day technical documentation",
            status="available",
            evidence_ref="reports/article15-current.md",
        ),
    ],
)

payload = report.to_dict()
markdown = report.to_markdown()
```

The packet joins per-tenant halt rates, contradiction-source forensics, recent
halt evidence, rolling drift alerts, readiness controls, and compliance-export
references. It is tenant-safe by construction: raw prompts, responses, customer
identifiers, feedback payloads, and compliance artefact contents are not
serialised.
