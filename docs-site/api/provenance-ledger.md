# Operational Provenance Ledger

`KnowledgeProvenanceLedger` records the lifecycle of the knowledge base
itself. Where the per-response provenance verifier proves the citation set of
one answer, the ledger proves *how the knowledge base reached its current
state*: every ingest, update, and delete is appended as a signed, ordered
event. It answers the two questions an auditor asks — *where did this chunk
come from* and *what happened to this document* — and detects after-the-fact
tampering of either the stored content or the mutation history.

Two integrity layers compose:

- **Content commitment.** Each event carries a Merkle root over the SHA-256
  digests of the chunks it admitted or retired. Editing a stored chunk later
  changes the digest, so the inclusion proof returned by `provenance_of` no
  longer folds to the recorded root.
- **HMAC chain.** Each event is folded into an HMAC-signed chain keyed on a
  digest of the event's full semantic payload. Reordering, deleting, or
  editing any field of any event breaks `verify()`.

Events persist as one JSON object per line. The ledger reloads and verifies
the file on construction, so a process restart resumes the exact chain and a
tampered file is rejected before any new event is appended.

## Wiring into ingestion

`DocumentIngestionPipeline` takes an optional `ledger`. When supplied, every
mutation is recorded automatically; when omitted, ingestion behaves exactly as
before.

```python
from director_ai.core.ingestion import DocumentIngestionPipeline
from director_ai.core.provenance import KnowledgeProvenanceLedger
from director_ai.core.retrieval.vector_store import VectorGroundTruthStore

ledger = KnowledgeProvenanceLedger(secret=secret_key, path="kb-provenance.jsonl")
pipeline = DocumentIngestionPipeline(store=VectorGroundTruthStore(), ledger=ledger)

result = pipeline.ingest_text(
    "Refunds are available within 30 days.",
    doc_id="refunds",
    source="refunds.md",
)

provenance = ledger.provenance_of(result.chunk_ids[0])
assert provenance.verified            # inclusion proof folds to the event root
assert provenance.source == "refunds.md"
assert ledger.verify() == (True, None)  # chain intact
```

An `update_text` that changes content appends an `update` event that admits the
new chunks and retires the previous revision's chunks; `provenance_of` returns
`None` for a retired chunk. A `delete` appends a `delete` event bound to the
exact chunk set it removed. An `update_text` with unchanged content appends no
event.

## Querying provenance

```python
# Full per-document history in chain order.
for event in ledger.history_for("refunds"):
    print(event.index, event.event_type, event.source, event.timestamp)

# Origin of one chunk, with a self-contained inclusion proof.
prov = ledger.provenance_of("refunds:chunk:0")
if prov is not None and prov.verified:
    print(prov.doc_id, prov.event_type, prov.proof.root.hex())
```

## Tamper detection

`verify()` re-derives the chain over the persisted events and returns
`(ok, first_bad_index)`. Construction raises `LedgerTamperError` when the
persisted file fails this check — whether a field was edited, the events were
reordered, or the file was signed with a different secret.

```python
ok, first_bad = ledger.verify()
if not ok:
    raise RuntimeError(f"provenance ledger compromised at event {first_bad}")
```

## Self-updating supersession

A self-updating knowledge base recognises when a new document replaces older
material. `KnowledgeSupersessionPolicy` turns three signals — an explicit
`supersedes` hint, a same-source revision, or a caller-supplied per-document
contradiction score — into a reviewable `SupersessionDecision`. The policy is
side-effect free; every non-empty decision is gated on human approval by
default, and auto-promotion is opt-in and only fires when every candidate
clears a high score bar.

```python
from director_ai.core.provenance import KnowledgeSupersessionPolicy

policy = KnowledgeSupersessionPolicy()
decision = policy.evaluate(
    incoming_doc_id="refunds_v2",
    incoming_source="refunds.md",
    tenant_id="acme",
    existing=pipeline.registry.list_for_tenant("acme"),
    contradiction_scores={"refunds_v1": 0.88},  # from an NLI/similarity verifier
)
# decision.action == "recommend"; decision.requires_human_approval is True
```

`DocumentIngestionPipeline.apply_supersession` executes an approved decision:
it retires each superseded document's chunks from the store and registry and
records a single ledger `supersede` event linking them to the incoming
document. A decision that still needs review is refused unless `approved=True`.

```python
result = pipeline.apply_supersession(decision, approved=True)
# result.superseded_doc_ids == ("refunds_v1",)
# the retired chunks now resolve to None via ledger.provenance_of(...)
# ledger.history_for("refunds_v2") contains a "supersede" event
```

## Online credibility from feedback

`SourceCredibility` already tracks a decaying trust score per source and already
feeds `ProvenanceVerifier`'s composite trust score. `CredibilityFeedbackLoop`
supplies the missing online-learning step: it folds human approvals and
rejections into that tracker, so a source whose cited facts keep getting
rejected drifts down while a consistently-approved source drifts up. Share the
tracker with the verifier and later responses are scored by what earlier
feedback taught.

```python
from director_ai.core.provenance import (
    CredibilityFeedbackLoop,
    ProvenanceChain,
    ProvenanceVerifier,
    SourceCredibility,
)

credibility = SourceCredibility()
loop = CredibilityFeedbackLoop(credibility=credibility)
verifier = ProvenanceVerifier(chain=ProvenanceChain(secret=secret), credibility=credibility)

# A human rejects a response citing "blog-x"; its credibility drops, and the
# verifier's trust score for the next "blog-x" citation drops with it.
loop.observe(source_ids=["blog-x"], human_approved=False)
```

The same credibility can re-rank retrieval candidates. `rerank` blends each
chunk's relevance (from its distance) with its source credibility; a weight of
0 keeps the pure relevance order, a weight of 1 ranks purely by credibility.

```python
ranked = loop.rerank(evidence_chunks, credibility_weight=0.4)
```

Stored corrections replay through the loop when the caller can resolve which
sources each response cited:

```python
loop.ingest_corrections(
    feedback_store.get_corrections(),
    source_resolver=lambda correction: sources_cited_by(correction.review_id),
)
```

## Counterfactual contradiction explanations

Grounding a claim is not only about finding support — it is about surfacing the
evidence that *refutes* it. `ContradictionExplainer` scores each retrieved
passage against a claim and returns a human-readable account of the
contradictions: *this claim contradicts the passage from source X because the
passage states "…" (contradiction 0.91)*.

The contradiction signal is injected, like `ConflictAwareKnowledgeGuard`'s
`score_fn`: the caller supplies `scorer(passage, claim) -> probability` backed
by the NLI scorer in `director_ai.core.scoring.nli`, a rule engine, or a domain
model. Keeping the model out of the explainer makes its selection-and-rationale
logic deterministic and testable on its own.

```python
from director_ai.core.causal_verifier import ContradictionExplainer

explainer = ContradictionExplainer(scorer=nli_contradiction_probability, threshold=0.5)
report = explainer.explain(claim, retrieved_chunks)
if report.has_contradiction:
    print(report.best.rationale)
    # "This claim contradicts the passage from policy.md because the passage
    #  states: "Refunds are never available." (contradiction 0.92)."
```

`report.contradictions` is ordered strongest-first; each entry keeps the
originating `chunk_index` and `chunk_source` so the contradiction can be traced
back to its retrieved passage.

## Content commitment

The Merkle commitment is available directly for callers that bind their own
content sets. `commit_root` and `prove_inclusion` use the Rust kernel
(`backfire_kernel.rust_merkle_*`) with a bit-identical pure-Python reference,
so an `InclusionProof` verifies regardless of which path produced it.

```python
from director_ai.core.provenance import commit_root, prove_inclusion

root = commit_root(leaf_digests)
proof = prove_inclusion(leaf_digests, index=2)
assert proof.verify()
assert proof.root == root
```

## Full API

::: director_ai.core.provenance.ledger.LedgerEvent

::: director_ai.core.provenance.ledger.ChunkProvenance

::: director_ai.core.provenance.ledger.KnowledgeProvenanceLedger

::: director_ai.core.provenance.ledger.LedgerTamperError

::: director_ai.core.provenance.content_commitment.InclusionProof

::: director_ai.core.provenance.content_commitment.commit_root

::: director_ai.core.provenance.content_commitment.prove_inclusion

::: director_ai.core.provenance.supersession.SupersessionCandidate

::: director_ai.core.provenance.supersession.SupersessionDecision

::: director_ai.core.provenance.supersession.KnowledgeSupersessionPolicy

::: director_ai.core.ingestion.pipeline.SupersessionResult

::: director_ai.core.provenance.credibility_feedback.CredibilityFeedbackLoop

::: director_ai.core.causal_verifier.contradiction_explainer.ContradictionExplanation

::: director_ai.core.causal_verifier.contradiction_explainer.ContradictionReport

::: director_ai.core.causal_verifier.contradiction_explainer.ContradictionExplainer
