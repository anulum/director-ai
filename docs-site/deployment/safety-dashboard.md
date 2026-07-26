<!--
SPDX-License-Identifier: Apache-2.0
Commercial licence available
Concepts 1996-2026 Miroslav Sotek. All rights reserved.
Code 2020-2026 Miroslav Sotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
Director-AI - safety operations dashboard
-->

# Safety Operations Dashboard

The safety operations dashboard turns tenant-safe `SafetyEvent` JSONL and
calibration feedback into an immediate halt-rate view.

It is intended for the first response to drift, stale knowledge, and repeated
false-positive halts:

- per-tenant event count, halt count, halt rate, false-positive count, and alert
  state
- top contradiction sources across recent halts
- recent halt evidence with score, reason, source, and suggested operator action
- a ready-to-run retune command for recent labelled feedback
- a one-click retune action in the wizard that turns labelled feedback into a
  tuned profile overlay with threshold guidance and confidence notes

## Launch The UI

```bash
director-ai safety-dashboard
```

The same panel is also available inside:

```bash
director-ai wizard
```

Open the **Safety Ops** tab, paste `SafetyEvent` JSONL, paste optional feedback
JSONL, and render the tables. Use **Retune from Feedback** when the feedback
rows include `prompt`, `response`, and a human verdict.

## Text Mode

Use text mode when running over SSH or in CI:

```bash
director-ai safety-dashboard \
  --text \
  --events safety_events.jsonl \
  --feedback recent_feedback.jsonl
```

The command prints tenant halt rates, top contradiction sources, recent evidence,
and the retune command:

```bash
director-ai tune --dataset recent_feedback.jsonl --output director-ai-tuned.yaml
```

## Trust Console Report

The same tenant-safe parser can build a customer-facing Trust Console report for
security reviews, pilots, and procurement questionnaires. The report includes
aggregate halt metrics, tenant alert state, recent evidence references, and
operator-supplied readiness controls. It never serialises raw prompt text,
response text, customer identifiers, media payloads, or feedback payloads.

```python
from director_ai.ui import TrustControl, build_trust_console_report

events_jsonl = open("safety_events.jsonl", encoding="utf-8").read()
report = build_trust_console_report(
    events_jsonl,
    controls=[
        TrustControl(
            control="PII redaction",
            status="passed",
            evidence_ref="docs/BENCHMARKS.md#pii-redaction",
            owner="security",
            updated_at="2026-05-17",
        ),
        TrustControl(
            control="Article 15 report template",
            status="passed",
            evidence_ref="docs-site/guide/compliance-reporting.md",
        ),
    ],
    generated_at="2026-05-17T12:05:00Z",
)

payload = report.to_dict()
markdown = report.to_markdown()
assert payload["privacy"]["raw_event_text_included"] is False
```

Control statuses are `passed`, `warning`, `failing`, or `not_applicable`.
Any failing control marks the report risk level as `critical`; tenant alert
states or warning controls mark it as `attention_required`.

## Observability Operations Report

Use the operations report when the deployment gate needs one tenant-safe packet
that combines halt forensics, drift alerts, readiness controls, and compliance
export references:

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
            control="Trace retention",
            status="passed",
            evidence_ref="runbooks/trace-retention.md",
        ),
    ],
    compliance_exports=[
        ComplianceExportRef(
            standard="EU AI Act Article 15",
            name="30-day operations export",
            status="available",
            evidence_ref="reports/article15-current.md",
        ),
    ],
    drift_alert_threshold=0.10,
)

payload = report.to_dict()
markdown = report.to_markdown()
assert payload["privacy"]["raw_event_text_included"] is False
```

The drift alert uses the first and second halves of each tenant's recent event
stream as baseline and current windows. It requires enough samples in both
windows, ignores feedback rows for halt-rate drift, and reports mild,
moderate, or severe drift based on rate change.

## Alert Thresholds

The defaults are intentionally conservative:

- halt-rate alert: `0.15`
- false-positive alert: `0.05`
- operations drift alert: `0.10`
- minimum drift window: `2` events per baseline/current window

Override them when a deployment has a known review cadence:

```bash
director-ai safety-dashboard \
  --text \
  --events safety_events.jsonl \
  --halt-alert-threshold 0.25 \
  --false-positive-alert-threshold 0.10
```

Dashboard, Trust Console, and observability operations builders fail closed when
an alert threshold is not a finite rate in `[0, 1]`. Operations reports also
reject non-positive drift windows before parsing events, which prevents
misconfigured dashboards from emitting misleading drift packets.

## Prometheus And Grafana Mixin

The deployable safety-operations mixin lives under `deploy/observability/`:

- `safety-ops-prometheus-rules.yml`
- `safety-ops-grafana-dashboard.json`

Load the rules from Prometheus:

```yaml
rule_files:
  - safety-ops-prometheus-rules.yml
```

Import `safety-ops-grafana-dashboard.json` into Grafana and select the same
Prometheus data source that scrapes `/v1/metrics/prometheus`.

The mixin uses these metric families:

| Metric | Purpose |
|--------|---------|
| `director_ai_halts_total` | Halt-rate numerator |
| `director_ai_reviews_total` | Halt-rate denominator |
| `director_ai_feedback_total{outcome="false_positive"}` | False-positive feedback |
| `director_ai_kb_stale_sources` | Stale source count |
| `director_ai_retune_recommended` | Retune recommendation state |
| `director_ai_retune_recommendations_total` | Retune recommendation count |

The alert rules include safe denominator guards for low-traffic services, so a
single halt during startup does not divide by zero.

## Input Shape

Each event should be one tenant-safe JSON object per line. Native
`SafetyEvent.to_dict()` records work directly:

```json
{"event_id":"e1","tenant_id":"tenant-a","policy_decision":"halt","halt_reason":"contradiction","observed_score":0.22,"trace_attribution":{"fact_source":"kb://physics"},"tenant_safe_explanation":"Refresh the cited fact."}
```

Feedback rows can use the calibration format:

```json
{"event_id":"e1","tenant_id":"tenant-a","guardrail_approved":false,"human_approved":true,"source":"kb://physics"}
```

Rows where the guard rejected an answer but the human accepted it count as
false positives for the tenant.

Rows used for one-click retuning also need the reviewed prompt and response:

```json
{"prompt":"What is the refund window?","response":"Refunds are available for 30 days.","guardrail_approved":true,"human_approved":true,"domain":"support"}
```

The UI requires at least four labelled rows before it emits an overlay. Mixed
approved and rejected examples produce more reliable threshold guidance; single-class
feedback is allowed but marked provisional.
