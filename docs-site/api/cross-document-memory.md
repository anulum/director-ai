# Cross-Document Consistency Memory

::: director_ai.core.memory.consistency.CrossDocumentConsistencyMemory

::: director_ai.core.memory.consistency.CrossDocumentConsistencyReport

::: director_ai.core.memory.consistency.CrossDocumentConflict

::: director_ai.core.memory.consistency.StoredDocument

## Tenant and Privacy Model

`CrossDocumentConsistencyMemory` is a SQLite-backed memory for checking whether
a new generated document contradicts earlier documents from the same tenant.
It is intended for long-running agents, support bots, and regulated workflows
where yesterday's answer must stay consistent with today's answer.

- all reads and comparisons are tenant-scoped
- tenant ids are validated against `^[A-Za-z0-9_-]{1,64}$`
- blocked reports do not persist the incoming document
- report serialization omits raw text unless `include_text=True`
- `delete_tenant()` removes retained tenant documents for right-to-delete
  workflows
- `max_documents_per_tenant` enforces bounded retention

```python
from director_ai.core import CrossDocumentConsistencyMemory

memory = CrossDocumentConsistencyMemory(
    "consistency.sqlite",
    score_fn=contradiction_score,
    contradiction_threshold=0.85,
)

report = memory.record_document(
    tenant_id="tenant-a",
    document_id="answer-2026-05-13",
    text=response,
)
if report.blocked:
    escalate(report.to_dict())
```

Use an NLI, verified-scorer, or domain verifier as `score_fn(previous, incoming)`.
The memory layer owns storage, tenant isolation, reporting, and retention; it
does not pretend to be a contradiction model by itself.
