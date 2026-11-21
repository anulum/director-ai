<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI — AtlasKV retrieval comparison results -->
# AtlasKV-style retrieval comparison: results

**The candidate was not accepted.** Entity-relation keys with two-level
pruning (arm C) found supporting evidence less often than the maintained
hybrid retrieval (arm A), so the gain rule failed. The run counts as a valid
negative result.

## What was compared

Three arms answered the same questions on one generated corpus, with one
encoder (`sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243`), one
`top_k` of 5 and one process. Nothing was trained.

- **A, hybrid:** the maintained hybrid backend over the evidence text.
- **B, keys:** flat entity-relation keys.
- **C, keys with pruning:** an index of entity names picks the best `g`
  entity groups, and only their keys are ranked. C falls back to B when the
  best group's similarity is below `tau` or the groups yield fewer than
  `top_k` items.

The corpus generator (`benchmarks/atlaskv_corpus.py`) and the harness
(`benchmarks/atlaskv_retrieval_bench.py`) were frozen before the test split
was run. Every artefact records their SHA-256 digests (`1b7d862e…` and
`1667aa73…`), the template digest (`6ead978a…`) and the digest of the frozen
study plan (`e96c44a3…ae785`). The acceptance rules were fixed in that plan
before any test number existed.

## Parameter choice on the dev split

The dev split (20 % of entities) chose `g = 2` and `tau = 0.4` by C's
supporting recall. On dev, `tau` changed nothing for any `g` (91.72 % at
`g = 2`, 91.50 % at 4 and 8), and the harness rule for ties (smaller `g`,
then larger `tau`) chose 0.4. `atlaskv_retrieval_dev.json` holds the tuning
only; its numbers are not results.

## Test split (80 % of entities, 1,920 questions)

| Arm | R_sup@5 | R_con@5 | p95 latency | False approvals | Fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: |
| A, hybrid | 96.13 % | 92.22 % | 136.6 ms | 0.06 % | — |
| B, keys | 81.72 % | 85.03 % | 45.9 ms | 0.00 % | — |
| C, keys with pruning | 92.67 % | 95.21 % | 80.5 ms | 0.24 % | 4.95 % |

R_sup@5 is the share of questions with a supporting item among the first
five, and R_con@5 the same for contradicting items. Intervals are 95 %
paired bootstraps over entity groups, resampling each entity with its
similar-name partner.

| Rule (C against A) | Outcome |
| --- | --- |
| Gain: R_sup@5 of C minus A is at least +5.0 pp, and its interval's lower bound is above zero | **Fails:** the difference is −3.45 pp, interval [−4.71, −2.25] |
| Contradictions: R_con@5 of C is at least A's minus 1.0 pp (point comparison) | Passes: 95.21 % against 92.22 % |
| False approvals: the interval's upper bound is at most 0.5 pp | Passes: [−0.06, 0.46] |
| Boundaries: no foreign-tenant item, no retracted fact, exact source and snapshot | Passes, including the 84 snapshot-swap questions |
| Cost: p95 at most 1.5 × A, RAM and index at most 2 × A | Passes |
| H2, C against B: lower p95 and selection time, with R_sup@5 and R_con@5 down by at most 1.0 pp | **Fails:** C's p95 (80.5 ms) is above B's (45.9 ms), though both recalls rose |

C's false approvals were four questions of the class whose false value
belongs to the shorter-named member of a similar-name pair. A's single false
approval was a contradicted fact.

## Why C fell back

The harness records only whether C fell back. `benchmarks/atlaskv_fallback_diagnostic.py`
answers the same questions again with the same index and splits each
fallback by condition. It also checks, for every question, that its
conditions agree with the fallback the index returns.

| Split | Questions | Fallbacks | Similarity below `tau` only | Too few items only | Both |
| --- | ---: | ---: | ---: | ---: | ---: |
| test | 1,920 | 95 | 95 | 0 | 0 |
| dev | 480 | 9 | 9 | 0 | 0 |

Every fallback came from the similarity threshold. With `g = 2`, the groups
always held at least five items.

## The frozen harness and the type-clean harness

The comparison and diagnostic runs used the frozen harness bytes, SHA-256
`1667aa73…`. They are kept verbatim as
`frozen_sources/atlaskv_retrieval_bench.py.txt`, so both artefacts stay tied
to the bytes that produced them.

The committed `benchmarks/atlaskv_retrieval_bench.py` differs from them in
typing only, so that it passes the repository's strict type gate. It adds an
encoder `Protocol` and four `typing.cast` calls and removes four
`# type: ignore` comments. Its digest is `ed0163ee…`. The test split was not
run again: a second run after the result is known would be a second look at
it.

That the type fix changed no behaviour was checked on the dev split, which
had already been spent on tuning. Three dev runs used one host, one
environment, the same arguments and 8 threads each: the frozen harness twice,
as a control, and the type-clean harness once. `behaviour_check/` holds the
three documents and `comparison.json`:

- **Between the two frozen runs,** 23 of 155 fields differ. All of them are
  measurements: build and rebuild times, latency percentiles, traced and
  resident memory, and wall time.
- **Between a frozen run and the type-clean run,** the same 23 fields differ,
  plus `sources.harness_sha256`. Every metric, interval, class count, the
  selection grid and the decision are identical.
- **The original dev run,** with default threads, also agrees with the frozen
  runs outside those measurement fields. The only other difference is the
  study plan digest it was given: it ran before the plan was frozen.

| File | SHA-256 |
| --- | --- |
| `frozen_sources/atlaskv_retrieval_bench.py.txt` | `1667aa737fab3621ec1dc9ec592201177bd4a7fdc17970121899aecd5559e97e` |
| `behaviour_check/dev_frozen_harness_run_a.json` | `bf77b8a588e6e7584641c21eb5e83b664e4bdeb5368f38046f013ceaa83ed16c` |
| `behaviour_check/dev_frozen_harness_run_b.json` | `4f8789b241f3aea0a45de4e249ebb2496e63222f23d255debf3e41e1544bb50c` |
| `behaviour_check/dev_type_clean_harness.json` | `ec3766145aa1344cf94409e53074a9d6f3f6cabd2f81a8cc778a05c7033ab664` |
| `behaviour_check/comparison.json` | `dbec1c259a32f48813461f085aa4867bb7c271587c709f3e185b8e33d91717cb` |

## Files

| File | SHA-256 |
| --- | --- |
| `atlaskv_retrieval_test.json` | `6a61fd3883a040c99c09de6a7bbb2a83706e21f3853cbeacd2bbf00badc73051` |
| `atlaskv_retrieval_dev.json` | `337dfad35c6d72da81f19b1fa1fa52c05f98209e1558327d59912d6a6eb3343c` |
| `atlaskv_fallback_test.json` | `283f22367ce47a7fac177dddcd84f4368b14ae7b22a165656e6e967706980b09` |
| `atlaskv_fallback_dev.json` | `325fbf2b68c4c1da202567ba183c9d95a978303f01617969cbc0d0db48bb8b4e` |
| `lab_server_venv_pip_freeze.txt` | `47a9bbd9169bf6005eea7ec0a4ed7b98a025fc5d2d0d6915942cf7c126feca51` |

Both comparison runs and both diagnostic runs used one host (Intel Xeon
E5-2640, Python 3.12.3, numpy 2.4.6) and the same environment, listed in
`lab_server_venv_pip_freeze.txt`.

## Reproducing

```bash
python -m benchmarks.atlaskv_retrieval_bench --split dev --output dev.json
python -m benchmarks.atlaskv_retrieval_bench --split test --g 2 --tau 0.4 --output test.json
python -m benchmarks.atlaskv_fallback_diagnostic --split test --g 2 --tau 0.4 --output fallback.json
```

Latency depends on the host; the other numbers depend only on the corpus
seed, the encoder revision and the frozen sources.
