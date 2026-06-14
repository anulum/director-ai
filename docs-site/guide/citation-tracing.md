# Citation tracing

> **Status: core (Apache-2.0), deterministic, opt-in in reasoning checks.** Pure
> character-offset interval mapping over the citation parser — no scoring, no
> inference, no tolerance.

A grounded answer cites its sources; an ungrounded one asserts and moves on.
Citation tracing links the two: it segments the response body into claim
sentences, attaches each inline citation to the claim it sits in, and reports
which claims carry a citation and which do not.

## How it works

1. Citations are parsed by [`citation_grounding`](structured-verification.md) —
   DOIs, arXiv ids, URLs, numeric markers resolved through the reference list, and
   author-year forms — each with a character offset.
2. The body (everything before the reference section) is segmented into claim
   sentences, each with its offset span.
3. Every citation is attached to the sentence whose span contains its offset.
4. `coverage` is the fraction of claim sentences carrying a citation;
   `uncited` lists the rest.

## Usage

```python
from director_ai.core import trace_citations

result = trace_citations(
    "Transformers came in 2017 [1]. They scale well.\n\n"
    "References:\n[1] https://arxiv.org/abs/1706.03762\n"
)
result.coverage                       # 0.5
[c.index for c in result.uncited]     # [1]  ("They scale well.")
result.claims[0].citations[0].identifier   # "1706.03762"
```

It is wired into the reasoning-chain verifier (opt-in, since most chains carry no
citations):

```python
from director_ai.core import verify_reasoning_chain

result = verify_reasoning_chain(answer, check_citations=True)
result.citation_trace.coverage
```

The trace is reported on `citation_trace` and does not affect `chain_valid` —
not every sentence needs a citation, so it is an advisory coverage signal, not a
validity gate.

## Measured

`python -m benchmarks.citation_tracing`:

| Metric | Value |
|---|---|
| Attachment accuracy (labelled snippets, n=5) | 1.00 |
| Throughput | ~42k traces/s |

Attachment is exact because it is deterministic offset mapping, not a learned
judgement. Numbers come from the committed benchmark and
`benchmarks/results/citation_tracing.json`. Unlike the lexical and arithmetic
checks this module has no Rust kernel — there is no hot numeric path to
accelerate, only interval arithmetic — so it runs identically everywhere.
