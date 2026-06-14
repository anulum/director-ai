# Pre-generation hallucination forecasting

> **Status: heuristic, dependency-free, and store-agnostic.** The forecaster
> scores a prompt *before* generation from three transparent signals. It is a
> triage aid, not a guarantee — it tells you where to spend retrieval, routing,
> or review budget, while the streaming halt and NLI scorer remain the
> response-side ground truth.

Every other guard in Director-AI scores the *response*. The
`HallucinationForecaster` runs one step earlier: given the incoming prompt — and,
optionally, the knowledge base it will be grounded against — it estimates how
likely the answer is to hallucinate, so a caller can pre-emptively retrieve more
context, route to a stronger model, or ask for human review instead of paying for
a generation that is likely to be halted.

## The three signals

| Signal | What it measures | Source |
|---|---|---|
| **ambiguity** | Under-specification — too few content words, vague terms (`something`, `stuff`), no concrete anchor (a digit or proper noun), several stacked questions | the prompt text |
| **kb_coverage** | Best lexical overlap between the prompt and the facts the store would retrieve for it; low coverage means the answer cannot be grounded | `GroundTruthStore.retrieve_context` |
| **pattern_history** | The observed hallucination rate of past answers whose prompt shared this one's coarse shape (leading word, length bucket, anchored/vague) | online feedback |

The three combine into a convex risk in `[0, 1]`. When no knowledge base is
supplied the coverage term falls back to a configurable prior; when the prompt's
shape has never been seen the history term is dropped and its weight
redistributed, so the score never depends on a signal it does not have.

A risk below `ground_threshold` (default 0.34) recommends **proceed**; below
`review_threshold` (default 0.66) recommends **ground** (retrieve / augment);
above it recommends **human_review**. The result also carries a short rationale.

## Usage

```python
from director_ai.guard import ProductionGuard

guard = ProductionGuard()
guard._store.add("capital of France", "The capital of France is Paris.")

result = guard.forecast("What is the capital of France?")
print(result.risk, result.recommendation, result.rationale)
# 0.36 'ground' ('well-specified and grounded',)
```

Feed observed outcomes back so the pattern signal learns online:

```python
guard.forecast_history.record("What is the capital of Atlantis?", hallucinated=True)
```

The forecaster and its history persist for the life of the guard, so the pattern
signal accumulates across requests.

## Polyglot backend

The lexical overlap behind `kb_coverage` runs through the Rust
`backfire_kernel.rust_word_overlap` extension when it is installed, and a
pure-Python Jaccard fallback otherwise. The two are **bit-for-bit identical** —
both case-fold, split on whitespace, and retain punctuation on the token — so the
dispatch is purely a speed choice and the forecaster runs everywhere. See
[Rust Acceleration](rust-acceleration.md).

## Measured separation

On the bundled labelled set (`python -m benchmarks.hallucination_forecast`) the
forecaster cleanly orders risky prompts above safe ones:

| Metric | Value |
|---|---|
| Mean risk, well-specified + grounded prompts | 0.43 |
| Mean risk, under-specified / ungrounded prompts | 0.67 |
| Pairwise ranking accuracy (AUROC equivalent) | 0.72 |
| Recommendation-band accuracy | 0.90 |
| Rust ↔ Python parity | exact |

These numbers come from the committed benchmark and
`benchmarks/results/hallucination_forecast.json`; reproduce them with the command
above. The separation is real but modest — the keyword store's loose matching
caps coverage precision, and a `VectorGroundTruthStore` sharpens it. We publish
the keyword-store numbers rather than the best case.
