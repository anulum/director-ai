<!--
SPDX-License-Identifier: AGPL-3.0-or-later
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

## Streaming False-Halt Cache

Produced by `benchmarks.streaming_false_halt_bench`.

| Field | Type | Meaning |
|---|---|---|
| `benchmark` | string | Benchmark name |
| `nli` | boolean | Whether NLI scoring was enabled |
| `total_passages` | integer | Number of known-good passages |
| `false_halts` | integer | Count of wrong halts |
| `false_halt_rate` | number | `false_halts / total_passages` |
| `avg_coherence` | number | Mean coherence over all passages |
| `avg_latency_ms` | number | Mean passage latency |
| `per_passage` | array[object] | Per-fixture outcome rows |
