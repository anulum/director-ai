<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Benchmark Cache Schema

This file documents cache files used by the public benchmark
reproduction manifest at `benchmarks/public_accuracy_manifest.toml`.

## AggreFact Score Cache

Produced by:

```bash
python -m benchmarks.aggrefact_eval \
  --model yaxili96/FactCG-DeBERTa-v3-Large \
  --save-scores benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json
```

Required top-level fields:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | string | Cache schema marker from the runner |
| `model` | string | Model identifier used for scoring |
| `backend` | string | Inference backend name |
| `samples` | integer | Number of scored examples |
| `scores` | array[number] | Entailment/support score per example |
| `labels` | array[integer] | Ground-truth label per example |
| `predictions` | array[integer] | Prediction per example at stored thresholds |
| `datasets_per_sample` | array[string] | AggreFact subset per example |
| `latencies_per_sample` | array[number] | Inference latency per example in seconds |
| `per_dataset` | object | Per-subset sample counts and balanced accuracy |
| `per_dataset_thresholds` | object | Tuned threshold per subset |

`scores`, `labels`, `predictions`, `datasets_per_sample`, and
`latencies_per_sample` must have equal length. The leaderboard metric is
`per_dataset_mean_balanced_accuracy_at_global_threshold`.

## HaluEval Data Cache

Produced by `benchmarks.halueval_eval` and reused by `benchmarks.e2e_eval`.

| Path | Format | Meaning |
|---|---|---|
| `benchmarks/.cache/halueval_qa.parquet` | parquet | QA task rows |
| `benchmarks/.cache/halueval_summarization.parquet` | parquet | Task key used by upstream dataset |
| `benchmarks/.cache/halueval_dialogue.parquet` | parquet | Dialogue task rows |
| `benchmarks/.cache/halueval_results.json` | JSON | Aggregate metrics from direct HaluEval runner |

Task result JSON shape:

| Field | Type | Meaning |
|---|---|---|
| `benchmark` | string | Benchmark name |
| `overall` | object | Aggregate TP/FP/TN/FN and derived metrics |
| `per_task` | object | Same metrics split by task |

## End-to-End Result Cache

Produced by:

```bash
python -m benchmarks.e2e_eval --nli --output-json benchmarks/results/e2e_guardrail.json
```

Required fields include `total`, `threshold`, `soft_limit`, `tp`, `fp`,
`tn`, `fn`, `catch_rate`, `false_positive_rate`, `precision`, `f1`,
`accuracy`, `evidence_coverage`, `avg_latency_ms`, `p95_latency_ms`, and
`per_task`.

## Local Judge Cache

Produced by `benchmarks.run_judge_benchmark`.

| Field | Type | Meaning |
|---|---|---|
| `nli_only` | object | End-to-end baseline metrics |
| `local_judge` | object | End-to-end local judge metrics |
| `total_elapsed_s` | number | Runner wall time |
| `hw` | object | Hardware fingerprint captured by the runner |

## HalluBench Internal Validation Cache

Produced by:

```bash
HF_TOKEN=<accepted-access-token> python -m benchmarks.hallubench_eval \
  --predictions-jsonl validation/hallubench_predictions.jsonl \
  --output-json benchmarks/results/hallubench_internal_validation.json
```

HalluBench is gated and licensed for non-commercial, no-derivatives use. The
result cache must not contain raw images, questions, ground-truth answers, or
model predictions. It may contain sample metadata, aggregate metrics, and
SHA-256 hashes of references and predictions for audit matching.

Required top-level fields:

| Field | Type | Meaning |
|---|---|---|
| `benchmark` | string | Benchmark name, expected `HalluBench` |
| `schema_version` | string | Result schema marker from the runner |
| `benchmark_evidence` | boolean | Always `false` until external review promotes a score |
| `claim_boundary` | string | Publication boundary for the gated benchmark |
| `dataset` | object | Dataset id, source, split, access, license, and raw-data flag |
| `model_id` | string | Operator-supplied model or system identifier |
| `parameters` | object | Filters and metric thresholds used for the run |
| `overall` | object | Aggregate totals and derived metrics |
| `by_application` | object | Metrics split by `emergency` and `urban` |
| `by_task_type` | object | Metrics split by HalluBench task taxonomy |
| `by_output_form` | object | Metrics split by `short` and `long` answer forms |
| `by_temporal` | object | Metrics split by temporal-pair and single-image rows |
| `image_modalities` | object | Count of referenced modalities such as RGB or SAR |
| `elapsed_seconds` | number | Runner wall time |
| `per_sample` | array[object] | Metadata, metric row, and reference/prediction hashes |

Each metric summary includes `total`, `passed`, `failed`,
`missing_predictions`, `accuracy`, `exact_match_rate`, `numeric_match_rate`,
and `mean_token_f1`. Per-sample rows include `ground_truth_sha256` and may
include `prediction_sha256`; they must not include `question`, `ground_truth`,
or `prediction` fields.

## Streaming False-Halt Cache

Produced by `benchmarks.streaming_false_halt_bench`.

| Field | Type | Meaning |
|---|---|---|
| `benchmark` | string | Benchmark name |
| `nli` | boolean | Whether NLI scoring was enabled |
| `total_passages` | integer | Number of known-good passages |
| `false_halts` | integer | Count of wrong halts |
| `false_halt_rate` | number | `false_halts / total_passages` |
| `halt_quality` | object | TP/FP/TN/FN, halt precision, halt recall, false-halt rate, token-of-halt accuracy, median halt latency |
| `avg_coherence` | number | Mean coherence over all passages |
| `avg_latency_ms` | number | Mean passage latency |
| `per_passage` | array[object] | Per-fixture outcome rows |
| `bad_passages` | array[object] | Labelled contradiction smoke rows with expected and observed halt token indices |

## Retrieval Quality Cache

Produced by `benchmarks.retrieval_bench`.

| Field | Type | Meaning |
|---|---|---|
| `backend` | string | Vector backend under test |
| `total_facts` | integer | Indexed facts, including distractors |
| `total_queries` | integer | Retrieval queries in the synthetic set |
| `hit_at_1` | number | Fraction with a relevant fact ranked first |
| `hit_at_3` | number | Fraction with a relevant fact in top 3 |
| `precision_at_3` | number | Mean relevant fraction among returned top-3 results |
| `latency_ms_avg` | number | Mean retrieval latency per query |
| `downstream_scoring` | object | Supported/unsupported factual-scoring probe using retrieved context |
| `per_query` | array[object] | Per-query retrieval rows |

`downstream_scoring` includes `scoring_threshold`, `total_cases`,
`scoring_accuracy`, `supported_accept_rate`, `unsupported_reject_rate`,
mean supported/unsupported factual divergence, mean scoring latency, and
`per_case` rows. Keep these scoring metrics separate from raw retrieval
rank metrics; they measure how retrieval quality propagates into the
guardrail decision path.
