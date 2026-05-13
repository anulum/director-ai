# Conflict-Aware Knowledge Checks

`ConflictAwareKnowledgeGuard` validates proposed facts before they enter a
retrieval store. It is designed for user facts, signed facts, passport claims,
and curated KB updates where contradictory inserts must be stopped before they
become retrievable context.

The guard works with `GroundTruthStore` and `VectorGroundTruthStore`:

- same-key value changes are blocked by default
- explicit contradiction references in metadata are blocked before write
- optional semantic contradiction scoring can warn or block against the current
  store contents
- reports use keys, references, and hashes only; raw fact text is not serialized
- unsafe metadata keys are stripped before a permitted write

```python
from director_ai import ConflictAwareKnowledgeGuard, GroundTruthStore, KnowledgeFact

store = GroundTruthStore()
store.add_fact("refund_policy", "Refunds are available within 30 days.")

guard = ConflictAwareKnowledgeGuard(store)
result = guard.add_fact(
    KnowledgeFact(
        key="refund_policy",
        value="Refunds are never available.",
        metadata={"claim_id": "claim-refund"},
    )
)

assert result.decision == "block"
assert store.facts["refund_policy"] == "Refunds are available within 30 days."
```

## Semantic Scoring

Pass `score_fn(existing, incoming) -> float` when the deployment has a calibrated
NLI, rule, or domain verifier for contradiction scoring. Scores must be finite
and in `[0, 1]`. Values above `warn_threshold` produce `warn`; values above
`block_threshold` produce `block`.

```python
guard = ConflictAwareKnowledgeGuard(
    store,
    score_fn=contradiction_score,
    warn_threshold=0.6,
    block_threshold=0.9,
)
```

## Explicit References

Metadata can declare a contradiction target by key, claim id, signed-fact id,
passport-claim id, source id, or external id:

```python
result = guard.add_fact(
    KnowledgeFact(
        key="policy_v2",
        value="Refunds are unavailable after delivery confirmation.",
        metadata={"contradicts": "claim-refund-v1"},
    )
)
```

The guard returns `block` when the target is already present. For vector stores,
this happens before `VectorGroundTruthStore.add_fact()`, so the fact is not
embedded and the store's post-write conflict ledger is not polluted by rejected
input.

## Full API

::: director_ai.core.retrieval.conflict_guard.KnowledgeFact

::: director_ai.core.retrieval.conflict_guard.KnowledgeConflict

::: director_ai.core.retrieval.conflict_guard.KnowledgeConflictCheck

::: director_ai.core.retrieval.conflict_guard.ConflictAwareKnowledgeGuard
