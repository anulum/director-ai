# Coverage exclusions — tracked branch-edge baseline

Status of the CI coverage gate and the documented remainder between the
measured figure and a literal 100 % line+branch score.

## Measurement source

- CI workflow run `29051155653` (commit `287d1a7d0346bc42d09025f6f1697be0e7010540`),
  job "Test (Python 3.12)": **99.80 %** total coverage — 44 117 statements,
  41 missed, 13 348 branches, 73 partial. Gate: `--cov-fail-under=97`.
- The CI `test` job installs the core + dev dependency set only. Optional
  extras (chromadb, crewai, z3, faiss, …) are exercised by the dedicated
  `Test extras (…)` matrix jobs, which do not contribute to this figure.

## Missed statements: resolved (2026-07-10)

All 41 missed statements from the measurement above were addressed in one
wave — dedicated tests in the owning per-module test surfaces for 15 modules
(Chroma embedding adapter, config validation bounds, licence signature
malformation, tuner-loader guards, FAISS `ivf_nlist` validation, LangGraph
node validation, guard z3 factory error path, closed audit-log contract,
Python sentence-splitter fallback, and others), plus removal of two
structurally unreachable `union == 0` guards in
`core/scoring/lexical_signals.py` (both sit behind emptiness checks that
already guarantee a non-empty union).

**CI-confirmed** (run `29059290946`, commit `96a1ded2`): 44 114 statements,
**0 missed**, 52 partial branch edges, total **99.91 %**. The per-file edge
table below matches that run exactly.

## Remaining exclusions: partial branch edges

The remainder are partial branch edges (`x->y` arcs never taken) in files
whose statements are fully covered. They fall into three categories:

| Category | Meaning |
| --- | --- |
| loop-edge | Backward arc (`y < x`): a loop's final iteration never leaves via that arc (e.g. `for … if match: return` exhausting without the early exit in one direction). |
| skip-edge | Forward arc: a defensive condition whose false path is only reachable under object states no public API produces today. |
| extras-gated | Arc taken only when an optional extra is present/absent in a combination the core CI job does not produce; the behaviour itself is covered by the extras matrix jobs or the floor job. |

Baseline as of run `29059290946` (52 branch edges, per file). The
`core/scoring/scorer.py` rows were re-mapped locally after the WCB-1
decomposition into `_divergence.py` and `_review_pipeline.py` (same
edges at their new locations; two `_heuristic_factual` edges closed by
the new mixin test surface — local full run: 44 178 statements,
0 missed, 47 partial):

| File | Edges | Category |
| --- | --- | --- |
| core/citation_grounding/citations.py | 112->117 | skip-edge |
| core/containment/detector.py | 230->227 | loop-edge |
| core/cyber_physical/hook.py | 138->140 | skip-edge |
| core/defense_genome/registry.py | 140->144 | skip-edge |
| core/federated_privacy/accountant.py | 200->208 | skip-edge |
| core/federated_privacy/rdp_accountant.py | 216->224 | skip-edge |
| core/financial_services/banking_policy.py | 319->317 | loop-edge |
| core/knowledge_graph/graph.py | 198->191 | loop-edge |
| core/license.py | 385->389 | skip-edge |
| core/meta_guard/adjuster.py | 160->166 | skip-edge |
| core/meta_guard/analyzer.py | 244->240 | loop-edge |
| core/multimodal_guard/encoders.py | 194->202 | extras-gated |
| core/multimodal_guard/verifier.py | 229->237 | extras-gated |
| core/runtime/correction.py | 276->278, 278->280, 280->282 | skip-edge |
| core/safety/injection.py | 519->524 | skip-edge |
| core/scoring/_nli_export.py | 139->147 | extras-gated |
| core/scoring/lexical_signals.py | 113->115, 119->121 | loop-edge |
| core/scoring/_divergence.py | 359->370, 365->370, 404->424 | skip-edge |
| core/scoring/_review_pipeline.py | 395->408, 403->408, 439->443, 620->625 | skip-edge |
| core/scoring/scorer.py | 497->500 | skip-edge |
| core/self_evolving/calibration.py | 63->61 | loop-edge |
| core/symbolic_chain/prover.py | 115->114 | loop-edge |
| core/tenant.py | 243->242 | loop-edge |
| core/trace_safe/oracle.py | 137->134 | loop-edge |
| core/verification/code_verifier.py | 102->104, 106->94 | loop-edge |
| core/verification/tool_call_verifier.py | 124->109, 171->185 | loop-edge |
| finetune_api.py | 77->165, 243->242, 331->337, 366->380 | extras-gated |
| integrations/crewai_swarm.py | 79->86 | extras-gated |
| integrations/inference_server_hooks.py | 203->206, 383->386 | extras-gated |
| integrations/langchain.py | 78->81 | extras-gated |
| integrations/langgraph.py | 104->107 | skip-edge |
| routers/process.py | 67->72, 173->182, 279->248 | skip-edge |
| routers/scoring.py | 65->70, 238->243 | skip-edge |

## Ratchet rule

This baseline may only shrink. A change that introduces a **new** partial
branch edge (or a new missed statement) must either add the covering test in
the same commit or add a justified row here. Burning down the edges above is
part of the `pragma: no cover` audit lane (WCD-1 in the internal backlog);
several loop-edges will require restructuring rather than tests and should be
evaluated for readability before chasing the metric.
