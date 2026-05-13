# Human Review Queue

`HumanReviewQueue` is a durable human-in-the-loop gate for halted outputs and
post-halt correction proposals. It is separate from `ReviewQueue`, which is only
the continuous batching queue for scorer throughput.

The queue stores pending cases, reviewer decisions, release decisions, and retry
requests in SQLite. Candidate text is never included in default audit payloads;
callers must explicitly request it when they are inside the tenant boundary.

```python
from director_ai import HumanReviewQueue

queue = HumanReviewQueue("review.db")

case = queue.enqueue_case(
    candidate_text="Corrected answer that still needs review.",
    evidence_refs=("kb://policy-42", "trace://halt-17"),
    tenant_id="tenant-a",
    request_id="req-123",
    source_kind="halt",
    reason="coherence halt",
)

queue.decide(
    case.case_id,
    reviewer_id="reviewer-1",
    action="approve",
    reason="source verified",
)

released_text = queue.release(
    case.case_id,
    reviewer_id="reviewer-1",
    release_id="release-20260513-001",
)
```

## Correction Proposals

Use `enqueue_correction_proposal()` to connect the queue to
`CorrectionLoop`. The correction loop still supplies verifier consensus and an
unreleased candidate; the human review queue supplies durable reviewer approval
and release audit.

```python
case = queue.enqueue_correction_proposal(
    proposal,
    tenant_id="tenant-a",
    request_id="req-123",
    reason="post-halt correction proposal",
)
```

## State Transitions

| Current status | Allowed action | New status | Release text available |
|----------------|----------------|------------|------------------------|
| `pending` | `approve` | `approved` | No |
| `approved` | `release()` | `released` | Yes |
| `pending` | `reject` | `rejected` | No |
| `pending` | `request_retry` | `retry_requested` | No |

Rejected, retry-requested, and released cases cannot be re-decided. `release()`
requires an approved case, a reviewer id, and a release id.

## Retry Gate

`retry_payload()` is available only after a reviewer requests retry. It returns
tenant-safe routing context and evidence references, never candidate text.

```python
queue.decide(
    case.case_id,
    reviewer_id="reviewer-1",
    action="request_retry",
    reason="needs fresher source",
    metadata={"retry_budget": 1},
)

payload = queue.retry_payload(case.case_id)
```

## Tenant-Safe Audit

`HumanReviewCase.to_dict()` excludes `candidate_text` by default:

```python
audit_payload = case.to_dict()
assert "candidate_text" not in audit_payload
```

Use `include_candidate=True` only inside a tenant-controlled review surface.

## Full API

::: director_ai.core.runtime.human_review.HumanReviewQueue

::: director_ai.core.runtime.human_review.HumanReviewCase

::: director_ai.core.runtime.human_review.HumanReviewDecision
