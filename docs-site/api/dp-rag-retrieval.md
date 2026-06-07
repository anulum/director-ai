# Differentially Private RAG Retrieval

Rank retrieval candidates **differentially privately** and meter every query
against a per-tenant privacy budget. Retrieval similarity scores are a side
channel — repeatedly ranking against a tenant's knowledge base can reveal which
documents it holds — so calibrated Laplace noise is added to the scores before
ranking, and each query is charged against a per-tenant `(ε, δ)` budget. A query
that would exceed the budget is refused before any noise is spent.

## Quick start

```python
from director_ai import ProductionGuard
from director_ai.core.config import DirectorConfig
from director_ai.core.dp_rag import ScoredItem, DPBudgetExceededError

dp = ProductionGuard(DirectorConfig()).dp_retrieval   # per-tenant budget, cap 10.0

candidates = [ScoredItem("doc-1", 0.91), ScoredItem("doc-2", 0.55), ScoredItem("doc-3", 0.12)]

ranking = dp.rank(candidates, tenant_id="acme", epsilon=0.5)
for item in ranking.items:        # ordered by the DP-noised score
    print(item.item_id, item.score)
print(ranking.epsilon_spent, ranking.epsilon_remaining)

try:
    dp.rank(candidates, tenant_id="acme", epsilon=100.0)
except DPBudgetExceededError:
    pass   # refused — would exceed the tenant's remaining budget
```

`rank()` returns a `PrivateRanking` with the noised, re-ordered items and the
`epsilon_spent` / `epsilon_remaining` for the tenant. `to_dict()` is tenant-safe
(item ids, noised scores, and budget only).

## Privacy budget

- The per-query `ε` is **adaptive** — the caller chooses it per request; the
  cumulative per-tenant cap is enforced.
- `remaining(tenant_id)` / `spent(tenant_id)` report the budget state.
- Budgets are **per tenant** — one tenant's queries never draw down another's.
- A query that would push a tenant past `max_epsilon` raises
  `DPBudgetExceededError` **without** consuming any budget.

```python
from director_ai.core.dp_rag import DifferentiallyPrivateRetrieval

dp = DifferentiallyPrivateRetrieval(max_epsilon=4.0, sensitivity=1.0, seed=None)
```

`sensitivity` is the L1 sensitivity of the similarity score (default 1.0 for
cosine-like scores in `[0, 1]`). `seed` is for reproducible noise in tests;
production uses system entropy (`None`), and successive queries draw independent
noise.

## Accounting model and scope

The accounting is `(ε, δ)` differential privacy via the shared
`PrivacyAccountant` (basic/advanced composition). Full Rényi-DP (RDP) accounting
is a future refinement — the shared accountant documents RDP/zCDP as out of
scope. This module makes the *retrieval ranking* differentially private and
budgeted; DP-aware decoding (logit noise) and DP-aware NLI scoring are separate
extensions on the same budget.

## Notes

- Builds on `core/federated_privacy` (`LaplaceMechanism`, `PrivacyAccountant`).
- `guard.dp_retrieval` persists across calls so the per-tenant budget
  accumulates; construct `DifferentiallyPrivateRetrieval` directly for a custom
  cap, sensitivity, or seed.
