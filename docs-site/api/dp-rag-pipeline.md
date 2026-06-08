# Unified DP-RAG Pipeline

::: director_ai.core.dp_rag.pipeline.DPRagPipeline

::: director_ai.core.dp_rag.decoding.DPTokenDecoder

::: director_ai.core.dp_rag.decoding.DPTokenChoice

## Boundary

A RAG answer leaks the private retrieval corpus through three stages, not one:
retrieval ranking, next-token decoding, and any released coherence score.
[`DifferentiallyPrivateRetrieval`](dp-rag-retrieval.md) meters retrieval alone;
`DPRagPipeline` charges **all three stages against one per-tenant `(ε)`
accountant**, so the budget reflects the whole pipeline.

| Stage | Mechanism | Guarantee |
|-------|-----------|-----------|
| `rank` | Laplace noise on similarity scores | `ε`-DP ranking |
| `decode` | exponential mechanism (Gumbel-max) on logits | `ε`-DP token selection |
| `release_score` | Laplace noise on the coherence score | `ε`-DP score |

Every stage is pure `ε`-DP, so the loss composes additively. A stage that would
push a tenant past `max_epsilon` is refused with `DPBudgetExceededError` **before**
any noise is drawn or budget charged; per-stage charges are logged (stage + `ε` +
tenant) so a tenant can audit where the budget went without any raw query, logit,
or score crossing the boundary.

```python
from director_ai.core.dp_rag import DPRagPipeline, ScoredItem

pipe = DPRagPipeline(max_epsilon=10.0)

ranking = pipe.rank(
    [ScoredItem("doc-1", 0.91), ScoredItem("doc-2", 0.44)],
    tenant_id="tenant-a",
    epsilon=2.0,
)
choice = pipe.decode(next_token_logits, tenant_id="tenant-a", epsilon=3.0)
released = pipe.release_score(coherence, tenant_id="tenant-a", epsilon=1.0)

print(pipe.spent("tenant-a"), pipe.remaining("tenant-a"))  # 6.0  4.0
```

`ProductionGuard.dp_rag_pipeline()` builds one with the guard's defaults.

## Decoding mechanism

`DPTokenDecoder` selects the next token with the **exponential mechanism**: token
`i` with logit `u_i` is released with probability `∝ exp(ε · u_i / (2Δ))`, where
`Δ` is the L∞ sensitivity of the logits to one record of the conditioning
context. This is implemented with the equivalent Gumbel-max trick — adding
`Gumbel(0, 2Δ/ε)` noise to each logit and taking the argmax (McSherry & Talwar,
FOCS 2007). The selection is pure `ε`-DP in the conditioning data.

## Privacy-utility tradeoff

`benchmarks/dp_rag_privacy_utility.py` measures utility against `ε`: top-`k`
overlap for retrieval and top-1 agreement for decoding, averaged over seeded
trials. Utility rises monotonically with `ε` toward the no-privacy baseline of
`1.0`, so a deployment can pick `ε` from the measured curve. For Gaussian-noise
pipelines, account composition with the
[Rényi-DP accountant](rdp-accountant.md) instead of the pure-`ε` accountant.
