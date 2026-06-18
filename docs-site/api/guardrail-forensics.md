# Guardrail Forensics

The KPI layer says whether the guardrail is healthy. The forensics layer explains
reviewed misses without exposing raw prompt, response, or evidence text.

`build_forensics_report()` consumes tenant-safe eval records, usually from
[`eval_trace`](eval-trace.md), joined with reviewer labels:

```python
from director_ai.core.observability import build_forensics_report

records = [
    {
        "director.eval.answer_id": "case-1",
        "director.eval.approved": True,
        "director.eval.score": 0.82,
        "director.eval.threshold": 0.60,
        "director.eval.scorer": "nli",
        "director.eval.model": "customer-model",
        "director.eval.evidence_count": 0,
        "label": "hallucination",
    }
]

report = build_forensics_report(records)
```

The report classifies each reviewed case as:

| Outcome | Meaning |
|---|---|
| `false_negative` | Reviewer labelled a hallucination that the guard allowed. |
| `false_positive` | Reviewer labelled a grounded answer that the guard halted. |
| `correct_halt` | Reviewer confirmed a halted hallucination. |
| `correct_allow` | Reviewer confirmed an allowed grounded answer. |
| `unlabelled_allow` / `unlabelled_halt` | Eval record has no reviewer label yet. |

For every case it records the scorer, model, model revision when supplied,
domain, threshold margin, knowledge-state summary, reason, and recommended
operator action. Examples include `refresh_or_add_governed_facts`,
`add_counterexample_and_recalibrate_scorer`, and
`review_retrieval_source_mapping`.

## CLI

`director-ai forensics` reads either a JSON array of records or an object with a
`records` array:

```bash
director-ai forensics --input eval_records.json --format markdown
```

The `json` output includes:

- top-level miss counts;
- misses grouped by scorer, model, and domain;
- per-case action recommendations;
- a privacy block confirming that raw prompt, response, and evidence text are
  not included.

This is the core file/export surface. The richer safety dashboard remains the
UI/operations packet around halt rates, drift alerts, controls, and compliance
exports.

## API

::: director_ai.core.observability.forensics.ForensicsCase

::: director_ai.core.observability.forensics.ForensicsReport

::: director_ai.core.observability.forensics.build_forensics_report

::: director_ai.core.observability.forensics.render_forensics_markdown

::: director_ai.core.observability.forensics.render_forensics_text
