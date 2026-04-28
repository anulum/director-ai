<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Managed training sweeps

Managed fine-tuning sweeps use the same `director-ai train sweep`
interface as customer-triggered jobs. The sweep command creates an
auditable matrix, submits each scenario as a Vertex AI CustomJob, and
stores model artefacts plus `training_result.json` under the scenario
output prefix.

## Dataset artefacts

The 2026-04-28 sweep dataset is stored in:

```text
gs://gotm-director-ai-training/labels/sweeps/20260428/
```

| Split | Rows | Purpose |
|---|---:|---|
| `managed_eval_1000_20260428.jsonl` | 1,000 | Shared balanced eval set |
| `managed_train_natural_100_20260428.jsonl` | 100 | Low-data curve |
| `managed_train_natural_500_20260428.jsonl` | 500 | Smoke-compatible curve |
| `managed_train_natural_1000_20260428.jsonl` | 1,000 | Main small-data curve |
| `managed_train_natural_5000_20260428.jsonl` | 5,000 | Medium-data reserve |
| `managed_train_natural_full_20260428.jsonl` | 29,420 | Full train split |
| `managed_train_balanced_1000_20260428.jsonl` | 1,000 | Label-balance scenario |
| `managed_train_hardneg_1000_20260428.jsonl` | 1,000 | Hard-negative enriched scenario |

Source rows came from the existing
`gs://gotm-director-ai-training/labels/distil_labels_v3_flipped.json`
label file. Conversion maps `hard_label` to the fine-tune `label`
field and uses deterministic seed `20260428`.

## Model set

The sweep covers the current stable model plus experimental choices:

| Alias | Model id | Status |
|---|---|---|
| `factcg-deberta-v3-large` | `yaxili96/FactCG-DeBERTa-v3-Large` | stable |
| `deberta-v3-large-nli` | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` | experimental |
| `roberta-large-mnli` | `roberta-large-mnli` | experimental |
| `deberta-v3-small` | `microsoft/deberta-v3-small` | experimental |
| `distilroberta-base` | `distilroberta-base` | experimental |

Experimental models require `--allow-experimental-model`. Unknown model
ids remain rejected unless the caller explicitly opts into the
experimental path.

## Submitted sweeps on 2026-04-28

| Sweep id | Jobs | Hardware | Purpose | Output prefix |
|---|---:|---|---|---|
| `managed-size-budget-20260428` | 30 | T4 / `n1-standard-8` | 5 models × 100/500/1,000 natural rows × 1/3 epochs | `gs://gotm-director-ai-training/managed-training/sweeps/managed-size-budget-20260428/` |
| `managed-label-strategy-20260428` | 10 | L4 / `g2-standard-8` | 5 models × balanced/hard-negative 1,000 rows × 1 epoch | `gs://gotm-director-ai-training/managed-training/sweeps/managed-label-strategy-20260428/` |
| `managed-full-e1-20260428` | 5 | T4 / `n1-standard-8` | 5 models × full 29,420-row train split × 1 epoch | `gs://gotm-director-ai-training/managed-training/sweeps/managed-full-e1-20260428/` |
| `managed-label-strategy-retry-20260428` | 3 | T4 / `n1-standard-8` | Retry failed balanced 1,000-row label-strategy jobs | `gs://gotm-director-ai-training/managed-training/sweeps/managed-label-strategy-retry-20260428/` |

All 45 jobs were accepted by Vertex AI. The first post-submit status check
reported 2 `RUNNING` jobs and 43 `PENDING` jobs. Later same-session checks
reported 12 `SUCCEEDED`, 2 `RUNNING`, and 31 `PENDING`, then 22 `SUCCEEDED`,
2 `RUNNING`, 20 `PENDING`, and 1 `FAILED` at `2026-04-28T19:30:11+02:00`.
A later refresh reported 32 `SUCCEEDED`, 2 `RUNNING`, 8 `PENDING`, and
3 `FAILED` at `2026-04-28T23:08:01+02:00`.

Failed jobs:

| Job | Ended | Note |
|---|---|---|
| `director-ai-managed-sweep-balanced1000-deberta-v3-large-nli-e1-b1` | `2026-04-28T17:07:03Z` | Vertex AI reported insufficient resources in `europe-west4`; retry candidate |
| `director-ai-managed-sweep-balanced1000-deberta-v3-small-e1-b1` | `2026-04-28T17:47:19Z` | Vertex AI internal retryable error |
| `director-ai-managed-sweep-balanced1000-distilroberta-base-e1-b1` | `2026-04-28T19:20:21Z` | Vertex AI reported insufficient resources in `europe-west4`; retry candidate |

Running jobs at the same refresh were
`director-ai-managed-sweep-naturalfull-factcg-deberta-v3-large-e1-b1` and
`director-ai-managed-sweep-naturalfull-deberta-v3-large-nli-e1-b1`.

A narrow retry sweep for the three failed balanced scenarios was submitted at
`2026-04-28T23:12:48+02:00` on T4 hardware. The three retry CustomJobs were
accepted and initially reported `JOB_STATE_PENDING`:

| Scenario | Job id |
|---|---|
| `balanced1000-deberta-v3-large-nli-e1-b1` | `projects/533499266196/locations/europe-west4/customJobs/9096172697733300224` |
| `balanced1000-deberta-v3-small-e1-b1` | `projects/533499266196/locations/europe-west4/customJobs/6484084913858412544` |
| `balanced1000-distilroberta-base-e1-b1` | `projects/533499266196/locations/europe-west4/customJobs/4750199057320771584` |

## Re-run commands

Size and epoch budget sweep:

```bash
director-ai train sweep --execute \
  --sweep-id managed-size-budget-20260428 \
  --project gotm-director-ai \
  --region europe-west4 \
  --image europe-west4-docker.pkg.dev/gotm-director-ai/director-ai-training/director-ai-benchmarks:925837e-managed-training-model-choice-20260428 \
  --output-prefix gs://gotm-director-ai-training/managed-training/sweeps/managed-size-budget-20260428 \
  --train-set natural100=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_train_natural_100_20260428.jsonl \
  --train-set natural500=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_train_natural_500_20260428.jsonl \
  --train-set natural1000=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_train_natural_1000_20260428.jsonl \
  --eval-set natural100=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl \
  --eval-set natural500=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl \
  --eval-set natural1000=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl \
  --model factcg-deberta-v3-large \
  --model deberta-v3-large-nli \
  --model roberta-large-mnli \
  --model deberta-v3-small \
  --model distilroberta-base \
  --epochs 1 \
  --epochs 3 \
  --batch-size 1 \
  --lr 1e-5 \
  --machine n1-standard-8 \
  --gpu NVIDIA_T4 \
  --gpu-count 1 \
  --timeout-min 120 \
  --allow-experimental-model \
  --limit 30
```

Label-strategy sweep:

```bash
director-ai train sweep --execute \
  --sweep-id managed-label-strategy-20260428 \
  --project gotm-director-ai \
  --region europe-west4 \
  --image europe-west4-docker.pkg.dev/gotm-director-ai/director-ai-training/director-ai-benchmarks:925837e-managed-training-model-choice-20260428 \
  --output-prefix gs://gotm-director-ai-training/managed-training/sweeps/managed-label-strategy-20260428 \
  --train-set balanced1000=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_train_balanced_1000_20260428.jsonl \
  --train-set hardneg1000=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_train_hardneg_1000_20260428.jsonl \
  --eval-set balanced1000=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl \
  --eval-set hardneg1000=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl \
  --model factcg-deberta-v3-large \
  --model deberta-v3-large-nli \
  --model roberta-large-mnli \
  --model deberta-v3-small \
  --model distilroberta-base \
  --epochs 1 \
  --batch-size 1 \
  --lr 1e-5 \
  --machine g2-standard-8 \
  --gpu NVIDIA_L4 \
  --gpu-count 1 \
  --timeout-min 120 \
  --allow-experimental-model \
  --limit 10
```

Full-data one-epoch sweep:

```bash
director-ai train sweep --execute \
  --sweep-id managed-full-e1-20260428 \
  --project gotm-director-ai \
  --region europe-west4 \
  --image europe-west4-docker.pkg.dev/gotm-director-ai/director-ai-training/director-ai-benchmarks:925837e-managed-training-model-choice-20260428 \
  --output-prefix gs://gotm-director-ai-training/managed-training/sweeps/managed-full-e1-20260428 \
  --train-set naturalfull=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_train_natural_full_20260428.jsonl \
  --eval-set naturalfull=gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl \
  --model factcg-deberta-v3-large \
  --model deberta-v3-large-nli \
  --model roberta-large-mnli \
  --model deberta-v3-small \
  --model distilroberta-base \
  --epochs 1 \
  --batch-size 1 \
  --lr 1e-5 \
  --machine n1-standard-8 \
  --gpu NVIDIA_T4 \
  --gpu-count 1 \
  --timeout-min 720 \
  --allow-experimental-model \
  --limit 5
```

## Result collection

Each scenario writes:

```text
<scenario-output-prefix>/training_result.json
<scenario-output-prefix>/* model artefacts
```

Collect results after jobs finish:

```bash
director-ai train harvest \
  --prefix-uri gs://gotm-director-ai-training/managed-training/sweeps/managed-size-budget-20260428
```

The harvester emits JSON sorted by `best_balanced_accuracy` descending, with
`best`, `result_count`, and per-scenario artifact URIs. It also works on local
directories for dry-run or downloaded artifacts:

```bash
director-ai train harvest --prefix-uri ./managed-training/sweeps/smoke
```

Then compare trained artefacts with:

```bash
director-ai train benchmark-models \
  --general-uri gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl \
  --model factcg-deberta-v3-large=<artifact-path>
```

Promotion rule: no experimental model becomes a default unless it wins on
the shared eval set and passes the anti-regression benchmark against the
current stable model.

## Current completed metrics

The refresh at `2026-04-28T23:08:01+02:00` harvested 30 completed
`size-budget` scenarios, 2 completed `label-strategy` scenarios, and no
completed `full-e1` scenarios. Balanced accuracy is measured on the shared
1,000-row eval set.

| Scenario | Balanced accuracy |
|---|---:|
| `size-budget/natural1000-deberta-v3-large-nli-e3-b1` | `0.759` |
| `label-strategy/balanced1000-roberta-large-mnli-e1-b1` | `0.733` |
| `size-budget/natural500-deberta-v3-large-nli-e3-b1` | `0.729` |
| `size-budget/natural1000-roberta-large-mnli-e3-b1` | `0.720` |
| `size-budget/natural1000-factcg-deberta-v3-large-e3-b1` | `0.716` |
| `label-strategy/balanced1000-factcg-deberta-v3-large-e1-b1` | `0.714` |
| `size-budget/natural500-factcg-deberta-v3-large-e3-b1` | `0.712` |
| `size-budget/natural500-factcg-deberta-v3-large-e1-b1` | `0.708` |
| `size-budget/natural1000-deberta-v3-large-nli-e1-b1` | `0.707` |
| `size-budget/natural1000-factcg-deberta-v3-large-e1-b1` | `0.705` |
| `size-budget/natural500-roberta-large-mnli-e3-b1` | `0.697` |
| `size-budget/natural100-factcg-deberta-v3-large-e3-b1` | `0.694` |
| `size-budget/natural1000-roberta-large-mnli-e1-b1` | `0.671` |
| `size-budget/natural500-deberta-v3-large-nli-e1-b1` | `0.668` |
| `size-budget/natural1000-distilroberta-base-e3-b1` | `0.661` |
| `size-budget/natural1000-deberta-v3-small-e3-b1` | `0.658` |
| `size-budget/natural100-deberta-v3-large-nli-e3-b1` | `0.652` |
| `size-budget/natural500-distilroberta-base-e3-b1` | `0.621` |
| `size-budget/natural500-deberta-v3-small-e3-b1` | `0.574` |
| `size-budget/natural100-factcg-deberta-v3-large-e1-b1` | `0.570` |
| `size-budget/natural500-roberta-large-mnli-e1-b1` | `0.560` |
| `size-budget/natural100-roberta-large-mnli-e3-b1` | `0.525` |
| `size-budget/natural100-roberta-large-mnli-e1-b1` | `0.502` |
| `size-budget/natural100-deberta-v3-large-nli-e1-b1` | `0.500` |
| `size-budget/natural100-deberta-v3-small-e1-b1` | `0.500` |
| `size-budget/natural100-deberta-v3-small-e3-b1` | `0.500` |
| `size-budget/natural100-distilroberta-base-e1-b1` | `0.500` |
| `size-budget/natural100-distilroberta-base-e3-b1` | `0.500` |
| `size-budget/natural500-deberta-v3-small-e1-b1` | `0.500` |
| `size-budget/natural500-distilroberta-base-e1-b1` | `0.500` |

Current best candidates:

| Rank | Scenario | Balanced accuracy | F1 | Final loss |
|---:|---|---:|---:|---:|
| 1 | `size-budget/natural1000-deberta-v3-large-nli-e3-b1` | `0.759` | `0.790` | `0.532` |
| 2 | `label-strategy/balanced1000-roberta-large-mnli-e1-b1` | `0.733` | `0.749` | `1.178` |
| 3 | `size-budget/natural500-deberta-v3-large-nli-e3-b1` | `0.729` | `0.771` | `0.542` |
| 4 | `size-budget/natural1000-roberta-large-mnli-e3-b1` | `0.720` | `0.761` | `0.636` |
| 5 | `size-budget/natural1000-factcg-deberta-v3-large-e3-b1` | `0.716` | `0.763` | `0.608` |

Do not promote from this snapshot alone. The current best candidate is
`size-budget/natural1000-deberta-v3-large-nli-e3-b1`, but promotion remains
blocked until the running full-data jobs finish and the candidate passes the
anti-regression benchmark against the current stable model.
