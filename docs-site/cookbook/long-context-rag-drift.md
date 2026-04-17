# Long-Context RAG Drift

Long responses drift. The first two paragraphs cite the source
faithfully; paragraph seven introduces a number that sounds
plausible and has no basis in the retrieved passages. NLI on the full
response averages the signal out and reports a comfortable coherence
score. This cookbook shows how to catch that middle-of-response
drift with chunked NLI and the retrieval strategies that make the
catch cheaper.

## Minimal reproduction

```python
from director_ai.core import CoherenceScorer
from director_ai.core.retrieval.vector_store import (
    HybridBackend,
    SentenceTransformerBackend,
    VectorGroundTruthStore,
)

store = VectorGroundTruthStore(
    backend=HybridBackend(base=SentenceTransformerBackend()),
)
store.ingest(
    [
        "ANULUM reported CHF 4.2M in revenue for FY2025 (audited).",
        "Marbach SG headcount at year-end 2025 was 12 employees.",
        "Director-AI v3.14 was released on 14 April 2026.",
    ],
)

scorer = CoherenceScorer(
    threshold=0.6,
    ground_truth_store=store,
    use_nli=True,
    chunked_nli=True,           # score each sentence, then aggregate
    chunked_aggregation="trimmed_mean",
)

response = (
    "ANULUM reported CHF 4.2M in revenue for FY2025. "
    "The company, based in Marbach SG, employs 12 people. "
    "Director-AI v3.14 shipped on 14 April 2026 with six new RAG "
    "backends. "
    "Market analysts project CHF 11M in FY2026 revenue. "  # drift
    "The headcount will double by Q3 next year."           # drift
)
approved, score = scorer.review(
    prompt="Summarise ANULUM's 2025 performance and outlook.",
    action=response,
)
print(f"approved={approved} score={score.score:.3f} "
      f"n_chunks={len(score.per_chunk)}")
for i, chunk_score in enumerate(score.per_chunk):
    print(f"  sentence {i}: {chunk_score:.3f}")
```

The output prints per-sentence scores. Sentences 0–2 land near `0.9`;
sentences 3–4 are below `0.3`. With
`chunked_aggregation="trimmed_mean"` (the v3.10 default) the bottom
25% of sentences drop out of the average, so a single bad sentence
does not pull the whole response below threshold — but the per-chunk
trace still flags it for the caller.

## Why chunked NLI beats whole-document NLI

| Signal | Whole-document | Chunked (trimmed mean) |
| --- | --- | --- |
| Per-sentence visibility | none | yes |
| Sensitivity to one bad sentence | averaged out | preserved |
| Token budget | fits 4k model | 512 per chunk, no truncation |
| Latency | one NLI call | N parallel calls |

Chunked NLI uses the same 0.4B DeBERTa FactCG model as the default
scorer, so the accuracy of each per-sentence decision matches the
AggreFact leaderboard entry.

## Retrieval strategies that reduce drift at source

Most drift comes from retrieval that *almost* works — a passage with
the right keywords but the wrong number. Four of the six retrieval
backends directly attack that failure mode:

```python
from director_ai.core.config import DirectorConfig

cfg = DirectorConfig(
    use_nli=True,
    threshold=0.6,
    # Parent-child: index small chunks for precision, return full
    # parent chunks for NLI context.
    parent_child_enabled=True,
    parent_child_parent_chunks=3,
    # HyDE: LLM generates a hypothetical answer, embed that and
    # retrieve against it.
    hyde_enabled=True,
    # Query decomposition: split "and"/"or" queries, retrieve per
    # part, fuse with Reciprocal Rank Fusion.
    query_decomposition_enabled=True,
    # Contextual compression: drop sentences that don't match the
    # query before passing to the scorer.
    contextual_compression_enabled=True,
)
store = cfg.build_store()
```

Every backend is a decorator over any `VectorBackend` — stack them
without changing the scorer. Repository references:

* `src/director_ai/core/retrieval/parent_child.py`
* `src/director_ai/core/retrieval/hyde.py`
* `src/director_ai/core/retrieval/query_decomposition.py`
* `src/director_ai/core/retrieval/contextual_compression.py`

## When NLI alone is not enough

Long technical responses (code, legal prose, multi-step proofs)
benefit from two additional signals on top of chunked NLI:

1. **`VerifiedScorer.verify(atomic=True)`** — atomic claim extraction
   then per-claim NLI with evidence attribution. Slower (4-6× latency)
   but surfaces *which* sentence failed and *which* source contradicted
   it.
2. **Cross-turn contradiction tracking** — on multi-turn threads,
   `ConversationSession` tracks whether the current answer contradicts
   a previous one. `contradiction_index` rises; halts trigger when it
   crosses `contradiction_index_halt` in `DirectorConfig`.

## Operating notes

* `chunked_aggregation` defaults to `"trimmed_mean"` with
  `trim_fraction=0.25`. Lower to `0.10` if your domain tolerates
  occasional hedging sentences (legal disclaimers, dialogue turns)
  without inflating false positives.
* Per-sentence scores are stored under `score.per_chunk`. Log them
  to your observability stack — the trace is the single most useful
  artefact for tuning thresholds on a new domain.
* The `HybridBackend` is the default for `VectorGroundTruthStore.grounded()`
  because sparse BM25 catches the key-term-wrong-number case that
  dense embeddings can miss.
* Long documents (> 16k tokens) may need `onnxruntime-gpu` with at
  least 16 GB VRAM to keep NLI latency bounded; CPU inference
  multiplies per-chunk latency by the sentence count.

## See also

* [`docs/BENCHMARKS.md`](https://github.com/anulum/director-ai/blob/main/docs/BENCHMARKS.md)
  — AggreFact per-dataset scores and latency matrix.
* Streaming-halt guide (planned under `guide/streaming-halt.md`) —
  catching drift *during* generation, not post-hoc.
