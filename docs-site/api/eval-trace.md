# OpenTelemetry Eval Trace Standard

Director's guard decision is most useful when it lands in the evaluation tracer a
team already runs. The eval trace standard emits each guard decision as an
**OpenTelemetry span** carrying a stable attribute schema — Director's own
`director.eval.*` fields plus the `gen_ai.*` semantic-convention keys — so an
OTLP-native eval tracer (Phoenix / Arize) ingests it directly, and a
metadata-based tracer (LangSmith, Ragas) consumes the same attribute dict.

The attributes are tenant-safe and OTel-primitive: no raw prompt, answer, or
chunk text, only ids, counts, scores, and labels.

## The attribute schema (`director.eval.v1`)

| Attribute | Meaning |
|---|---|
| `director.eval.schema_version` | schema version (`director.eval.v1`) |
| `director.eval.decision` | `allow` / `halt` |
| `director.eval.approved` | guard approval (bool) |
| `director.eval.score` / `director.eval.threshold` | coherence score and threshold |
| `director.eval.evidence_count` | evidence chunks the decision rested on |
| `director.eval.unsupported_claims` | unsupported claims (from the Answer BOM) |
| `director.eval.model` / `director.eval.scorer` | model and scorer ids |
| `director.eval.tenant_id` / `director.eval.domain` | tenant and domain |
| `director.eval.answer_id` | links the span to the Answer BOM |
| `gen_ai.system` / `gen_ai.operation.name` / `gen_ai.request.model` | OTLP gen-AI conventions |

## Emitting a decision

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard.from_profile("finance")
result = guard.check(prompt, response)

# Emit an OTel span and get the record back for non-OTLP tracers
record = guard.eval_trace(result, model="gpt-4o", tenant_id="acme", domain="finance")
```

`record` is the attribute dict; the span is emitted under the
`director_ai.eval.guard_decision` name. Configure an exporter once with
`setup_otel()` (see the [OpenTelemetry bridge](#)) and the spans flow to your
collector.

For direct control:

```python
from director_ai.core.eval_trace import (
    guard_decision_attributes,
    record_guard_decision,
)

attrs = guard_decision_attributes(
    decision="halt", approved=False, score=0.4, threshold=0.6, model="gpt-4o"
)
with record_guard_decision(attrs) as span:
    ...   # span carries the attributes; a no-op sink when the SDK is absent
```

## Ingesting downstream

- **Phoenix / Arize** ingest OTLP spans directly — point an OTLP exporter at the
  collector and the `gen_ai.*` + `director.eval.*` attributes appear on each
  span.
- **LangSmith / Ragas** take metadata records — pass the dict returned by
  `eval_trace` / `eval_record_from_guard` as the run metadata.

The emitter is a no-op when the OpenTelemetry SDK is not installed, so calling
`eval_trace` is always safe.
