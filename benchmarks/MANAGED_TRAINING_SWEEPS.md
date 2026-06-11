<!--
SPDX-License-Identifier: Apache-2.0
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

All 45 original jobs were accepted by Vertex AI. Final status on
2026-04-29: the 30 size/epoch-budget jobs succeeded, the 5 full-data
jobs succeeded, and 6 of 10 original label-strategy jobs succeeded by
Vertex job state. The 3 balanced label-strategy failures were retried on
T4 hardware and all 3 retry jobs succeeded. The hard-negative DeBERTa
v3-large NLI job ended as `FAILED` in Vertex but wrote a usable
`training_result.json`; the hard-negative FactCG job ended as `FAILED`
and did not write a harvested result.

Failed original label-strategy jobs:

| Job | Ended | Note |
|---|---|---|
| `director-ai-managed-sweep-balanced1000-deberta-v3-large-nli-e1-b1` | `2026-04-28T17:07:03Z` | Vertex AI reported insufficient resources in `europe-west4`; retry candidate |
| `director-ai-managed-sweep-balanced1000-deberta-v3-small-e1-b1` | `2026-04-28T17:47:19Z` | Vertex AI internal retryable error |
| `director-ai-managed-sweep-balanced1000-distilroberta-base-e1-b1` | `2026-04-28T19:20:21Z` | Vertex AI reported insufficient resources in `europe-west4`; retry candidate |
| `director-ai-managed-sweep-hardneg1000-deberta-v3-large-nli-e1-b1` | `2026-04-28T21:45:04Z` | Vertex job failed after writing `training_result.json`; harvested metric retained |
| `director-ai-managed-sweep-hardneg1000-factcg-deberta-v3-large-e1-b1` | `2026-04-28T21:45:09Z` | Vertex job failed and no harvested result was present |

A narrow retry sweep for the three failed balanced scenarios was submitted at
`2026-04-28T23:12:48+02:00` on T4 hardware. The three retry CustomJobs
succeeded:

| Scenario | Job id |
|---|---|
| `balanced1000-deberta-v3-large-nli-e1-b1` | `projects/533499266196/locations/europe-west4/customJobs/9096172697733300224` |
| `balanced1000-deberta-v3-small-e1-b1` | `projects/533499266196/locations/europe-west4/customJobs/6484084913858412544` |
| `balanced1000-distilroberta-base-e1-b1` | `projects/533499266196/locations/europe-west4/customJobs/4750199057320771584` |

Harvested result counts:

| Prefix | Harvested results | Best scenario | Best BA |
|---|---:|---|---:|
| `managed-size-budget-20260428` | 30 | `natural1000-deberta-v3-large-nli-e3-b1` | `0.759` |
| `managed-label-strategy-20260428` | 6 | `hardneg1000-deberta-v3-large-nli-e1-b1` | `0.738` |
| `managed-label-strategy-retry-20260428` | 3 | `balanced1000-deberta-v3-large-nli-e1-b1` | `0.752` |
| `managed-full-e1-20260428` | 5 | `naturalfull-deberta-v3-large-nli-e1-b1` | `0.800` |

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

Then compare trained artefacts on Vertex AI with:

```bash
MODEL_SPECS='{"naturalfull-deberta-v3-large-nli":"gs://gotm-director-ai-training/managed-training/sweeps/managed-full-e1-20260428/naturalfull-deberta-v3-large-nli-e1-b1"}' \
GENERAL_URI='gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl' \
benchmarks/run_vertex_model_benchmark.sh --suffix full-e1-YYYYMMDD
```

The local CLI can still compare downloaded artefacts for debugging:

```bash
director-ai train benchmark-models \
  --general-uri gs://gotm-director-ai-training/labels/sweeps/20260428/managed_eval_1000_20260428.jsonl \
  --model factcg-deberta-v3-large=<artifact-path>
```

Promotion rule: no experimental model becomes a default unless it wins on
the shared eval set and passes the anti-regression benchmark against the
current stable model.

## Current completed metrics

The 2026-04-29 harvest contains 44 result artefacts across the original and
retry sweeps. Balanced accuracy is measured on the shared 1,000-row eval set.
The full-data DeBERTa v3-large NLI run is the training-eval winner. The
separate Vertex model-choice anti-regression benchmark below selects the
full-data FactCG artefact as the best deployable candidate on the same
1,000-row general benchmark gate.

| Rank | Sweep | Scenario | Train rows | Epochs | BA | F1 | Final loss |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | full-e1 | `naturalfull-deberta-v3-large-nli-e1-b1` | 29,420 | 1 | `0.800` | `0.8291` | `0.6987` |
| 2 | full-e1 | `naturalfull-roberta-large-mnli-e1-b1` | 29,420 | 1 | `0.765` | `0.7889` | `0.8625` |
| 3 | size-budget | `natural1000-deberta-v3-large-nli-e3-b1` | 1,000 | 3 | `0.759` | `0.7895` | `0.5315` |
| 4 | label-strategy retry | `balanced1000-deberta-v3-large-nli-e1-b1` | 1,000 | 1 | `0.752` | `0.7578` | `1.0907` |
| 5 | full-e1 | `naturalfull-deberta-v3-small-e1-b1` | 29,420 | 1 | `0.752` | `0.7888` | `0.8070` |
| 6 | full-e1 | `naturalfull-factcg-deberta-v3-large-e1-b1` | 29,420 | 1 | `0.752` | `0.7916` | `0.7530` |
| 7 | label-strategy | `hardneg1000-deberta-v3-large-nli-e1-b1` | 1,000 | 1 | `0.738` | `0.7189` | `1.0325` |
| 8 | label-strategy | `balanced1000-roberta-large-mnli-e1-b1` | 1,000 | 1 | `0.733` | `0.7488` | `1.1784` |
| 9 | size-budget | `natural500-deberta-v3-large-nli-e3-b1` | 500 | 3 | `0.729` | `0.7713` | `0.5420` |
| 10 | full-e1 | `naturalfull-distilroberta-base-e1-b1` | 29,420 | 1 | `0.722` | `0.7574` | `0.8418` |

## Vertex model-choice benchmark

The first full-data Vertex model-choice benchmark attempt used image
`director-ai-benchmarks:81e2a2e` and wrote:

```text
gs://gotm-director-ai-training/managed-training/benchmarks/20260429T1300-81e2a2e-full-e1-20260429/model_benchmark_report.json
```

That CustomJob succeeded as infrastructure, but every model result was an
import error. A diagnostic Vertex job showed the root cause was an
incompatible inherited `torchvision` package in the PyTorch base image:
`operator torchvision::nms does not exist`. The benchmark and distillation
Dockerfiles now uninstall inherited `torchvision` after the hash-pinned
dependency install.

Two corrected Vertex runs followed:

| Image | Job id | Status | Result |
|---|---|---|---|
| `director-ai-benchmarks:83aac35` | `3857430799938748416` | succeeded | Valid scores, but custom artefact aliases collapsed to `custom-experimental` in the summary |
| `director-ai-benchmarks:e9859f8` | `7361231310032994304` | succeeded | Final clean report with readable artefact aliases |

The final report is stored at:

```text
gs://gotm-director-ai-training/managed-training/benchmarks/20260429T1409-e9859f8-full-e1-20260429-alias/model_benchmark_report.json
```

Final Vertex anti-regression results:

| Rank | Artefact alias | General BA | General F1 | Regression pp | Recommendation |
|---:|---|---:|---:|---:|---|
| 1 | `naturalfull-factcg-deberta-v3-large` | `0.752` | `0.7916` | `-0.6` | `deploy` |
| 2 | `naturalfull-deberta-v3-small` | `0.747` | `0.7843` | `-1.1` | `deploy` |
| 3 | `naturalfull-deberta-v3-large-nli` | `0.740` | `0.7822` | `-1.8` | `deploy` |
| 4 | `naturalfull-distilroberta-base` | `0.719` | `0.7604` | `-3.9` | `deploy_domain_only` |
| 5 | `naturalfull-roberta-large-mnli` | `0.706` | `0.7529` | `-5.2` | `deploy_domain_only` |

Promotion decision from these runs: do not promote the experimental DeBERTa
v3-large NLI full-data artefact as the default scorer despite its `0.800`
training-eval BA. The anti-regression gate chooses the full-data FactCG
artefact as the best deployable candidate (`0.752` general BA, `-0.6pp`
against the `0.758` baseline).
