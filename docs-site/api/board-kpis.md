# Board-Level Guardrail KPIs

A guardrail product is steered by a handful of numbers, not by another dashboard
of raw metrics. This layer is those numbers and a board-facing way to read them:

- [`compute_kpis`](#data-layer) derives the KPIs from the same reviewer-labelled
  decisions the [active-labelling cockpit](labelling-cockpit.md) produces, plus
  operational counters the host already tracks. It is deterministic and
  tenant-safe (aggregates only).
- [`kpi_report`](#presentation-layer) classifies each KPI against operating
  targets (`ok` / `watch` / `alert`) and renders a Markdown or plain-text
  summary.
- The [`director-ai kpis`](#cli-export) command is the export front-end over both.

## Data layer

`compute_kpis` takes the reviewer-labelled `LabelItem`s and turns them into one
`KpiReport`:

| KPI | Meaning |
|-----|---------|
| `labelled_total` | Number of decisions a reviewer has labelled. |
| `halt_rate` | Fraction of labelled decisions the guard halted. |
| `halt_precision` | Of the halts, the fraction that were real hallucinations. |
| `false_positive_rate` | Of grounded answers, the fraction wrongly halted. |
| `per_domain_false_positive_rate` | The same FPR, split by domain. |
| `p95_scoring_latency_ms` | 95th-percentile end-to-end scoring latency. |
| `tenant_boundary_violations` | Counter passed through verbatim. |
| `unsigned_kb_writes_rejected` | Counter passed through verbatim. |
| `security_exception_debt` | Counter passed through verbatim. |

Metrics with no supporting data (no halts, no grounded items, no latency
samples) are `None` rather than a fabricated zero.

```python
from director_ai.core.labelling_cockpit import LabelItem
from director_ai.core.observability import compute_kpis

items = [
    LabelItem("a", 0.9, guard_approved=False, domain="legal", label="hallucination"),
    LabelItem("b", 0.2, guard_approved=True, domain="legal", label="grounded"),
]
report = compute_kpis(
    items,
    latency_ms_samples=[10.0, 20.0, 30.0],
    security_exception_debt=1,
)
```

## Presentation layer

`KpiTargets` holds the operating thresholds. A metric is `alert` once it crosses
its target, `watch` once it enters the shoulder below the target
(`watch_fraction` of the way there), and `ok` otherwise. `None` metrics render as
`n/a`.

```python
from director_ai.core.observability import (
    KpiTargets, kpi_statuses, overall_status, render_markdown, render_text,
)

targets = KpiTargets(max_false_positive_rate=0.10, min_halt_precision=0.80)
print(overall_status(report, targets))   # worst per-metric status
print(render_text(report, targets=targets))
print(render_markdown(report, targets=targets))
```

`kpi_statuses(report, targets)` returns the per-metric status map (including one
entry per domain, keyed `false_positive_rate[<domain>]`); `overall_status`
collapses it to the worst of `alert` > `watch` > `ok`.

## CLI export

`director-ai kpis` reads a JSON bundle and prints the report in `text`
(default), `markdown`, or `json`:

```bash
director-ai kpis --input kpis.json --format markdown
```

The bundle is a JSON object:

```json
{
  "items": [
    {"item_id": "a", "score": 0.9, "guard_approved": false,
     "domain": "legal", "label": "hallucination"},
    {"item_id": "b", "score": 0.2, "guard_approved": true,
     "domain": "legal", "label": "grounded"}
  ],
  "latency_ms_samples": [10.0, 20.0, 30.0],
  "tenant_boundary_violations": 0,
  "unsigned_kb_writes_rejected": 2,
  "security_exception_debt": 1,
  "targets": {"max_false_positive_rate": 0.10, "min_halt_precision": 0.80}
}
```

Only the documented `LabelItem` fields are read from each record; `targets` is an
optional overlay over the defaults. The `json` format emits the report, the
per-metric status map, and the overall status together, suitable for a web
dashboard to consume. No raw prompt/response text is included — the export is
tenant-safe by construction.
