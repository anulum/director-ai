# `benchmarks/scores/` — schema and naming convention

Every `*.json` file in this directory is a **cached score dump of one NLI model
evaluated on the AggreFact benchmark** (the LLM-AggreFact test suite). They are
committed so the comparison tables in `benchmarks/comparison/` and the docs-site
leaderboard are reproducible without re-running the models.

## What the filename means — read this before citing a file

The filename encodes the **model variant**, *not* an evaluation dataset:

- `base.json` — the base `FactCG-DeBERTa-v3-Large` model, un-fine-tuned.
- `factcg-<corpus>.json` — the same FactCG model **fine-tuned on `<corpus>`**
  (see the matching `tools/run_<corpus>_training.py`), then scored on AggreFact.

The `<corpus>` suffix is the **training corpus of the model**. It is **NOT** the
evaluation dataset. Every file — whatever its suffix — is scored on the *same*
11 AggreFact subsets below.

> **Do not misread the suffix as the eval set.** `factcg-fever.json` is the
> FEVER-*trained* FactCG model's scores **on AggreFact**; it is **not** FEVER
> evaluation data, and it contains no FEVER split. The same holds for
> `factcg-snli.json`, `factcg-wanli.json`, `factcg-boolq.json`, etc. To read
> results *on* FEVER, use `benchmarks/fever_eval.py`, not this directory.

## JSON schema

Each file is a JSON object keyed by **exactly** these 11 AggreFact subsets, in
any order:

| Subset | Task type (adaptive-threshold routing) |
|---|---|
| `AggreFact-CNN` | summarization |
| `AggreFact-XSum` | summarization |
| `TofuEval-MediaS` | summarization |
| `TofuEval-MeetB` | summarization |
| `Wice` | fact_check |
| `Reveal` | fact_check |
| `ClaimVerify` | fact_check |
| `FactCheck-GPT` | fact_check |
| `ExpertQA` | qa |
| `Lfqa` | qa |
| `RAGTruth` | rag |

The canonical subset → task-type mapping lives in
`benchmarks/threshold_analysis.py::TASK_TYPE_MAP` and is used for adaptive
thresholding.

Each subset maps to a **list of `[label, score]` pairs**, one per test example:

- `label` — the gold binary consistency label, `0` (unsupported) or `1`
  (supported).
- `score` — the model's predicted probability of "supported", a float in
  `[0, 1]`.

```json
{
  "AggreFact-CNN": [[1, 0.7365], [0, 0.0802], ...],
  "RAGTruth":      [[1, 0.9313], ...]
}
```

Balanced accuracy per subset (and the macro-average headline, e.g. the 75.6 %
in `benchmarks/comparison/COMPETITOR_COMPARISON.md`) is computed from these
pairs at the operating threshold.

## Contract

`tests/test_factcg_scores_schema.py` enforces this document: every
`benchmarks/scores/*.json` must be an object keyed by exactly the 11 subsets
above, with `[label, score]` pairs (`label ∈ {0, 1}`, `0.0 ≤ score ≤ 1.0`), and
this file must list every subset. A new dump with a different shape, or a subset
renamed here without updating the data, fails the test.
