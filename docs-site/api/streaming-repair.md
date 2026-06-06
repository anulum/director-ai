# Streaming Repair

The streaming kernel halts a stream when coherence drops and discards the unsafe
tail. That is safe but blunt — one bad clause throws away the rest of a good
answer. **Streaming repair** turns the halt into a corrective pass: it pauses,
finds the unsupported clause, retrieves corrective evidence, rewrites *only* that
clause (or redacts it when no rewrite path is configured), and resumes with the
supported clauses intact — emitting a tenant-safe repair event for each fix.

## How it works

1. **Split.** The answer is split into clauses that rejoin losslessly — the
   sentence terminators and the whitespace between sentences are preserved, so
   the repaired answer differs from the original only where a clause was changed.
2. **Score.** Each clause is scored for support against the knowledge base.
3. **Repair.** A clause below the threshold is rewritten from retrieved
   corrective evidence when a rewrite path and evidence are available; otherwise
   it is redacted. Supported clauses are kept verbatim.
4. **Resume.** The clauses are reassembled into the corrected answer.
5. **Report.** A `RepairResult` carries the corrected text, a per-clause action
   (`keep` / `rewrite` / `redact`), and a `SafetyEvent` per fix.

Scoring, retrieval, and rewriting are injected, so the repairer is decoupled
from any particular scorer, store, or model.

```python
from director_ai.core.streaming_repair import StreamingRepairer

def score(clause: str) -> float:
    ...   # support in [0, 1] for one clause

def retrieve(clause: str):
    return [{"id": "vector:ceo", "text": "The CEO is Jane Doe."}]

def rewrite(clause: str, evidence: list[str]) -> str:
    return "The CEO is Jane Doe."

repairer = StreamingRepairer(
    score, threshold=0.6, retrieve_fn=retrieve, rewrite_fn=rewrite
)
result = repairer.repair("The CEO is a robot. Contact support.", tenant_id="acme")

print(result.repaired)        # True
print(result.repaired_text)   # "The CEO is Jane Doe. Contact support."
for event in result.events:   # one tenant-safe warn event per fix
    ...
```

When no `rewrite_fn` is supplied, or retrieval finds no corrective evidence, or
the rewrite comes back blank, the clause is redacted with a clearly-labelled
placeholder rather than left in place.

## Through the guard

`ProductionGuard.repair_stream()` wires the repairer to the guard's scorer (each
clause scored against the knowledge base via `review`) and store (corrective
evidence via `retrieve_context_with_chunks`):

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard.from_profile("finance")
guard.load_facts({"refund": "Refunds are available within 30 days."})

result = guard.repair_stream(
    "What is the refund window?",
    "Refunds close after 30 days. Late refunds get a bonus.",
    tenant_id="acme",
    rewrite_fn=my_llm_rewrite,   # optional; redacts when omitted
)
```

## Metrics

- `streaming_repair_clauses_total` — clauses scored.
- `streaming_repair_actions_total{action}` — actions taken, labelled
  `keep` / `rewrite` / `redact`.
