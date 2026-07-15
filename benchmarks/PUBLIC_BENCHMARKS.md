<!--
SPDX-License-Identifier: Apache-2.0
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Public Benchmark Reproduction

This page is the public index for accuracy and halt-rate tables that
appear in `README.md` and `benchmarks/BENCHMARK_REPORT.md`.

The machine-readable source of truth is
`benchmarks/public_accuracy_manifest.toml`. The manifest lists:

- public table id and source file;
- runner file(s);
- dataset id(s);
- cache/result artefact path(s);
- reproduction command(s);
- metric names used by the table.

The JSON result shape for orchestrated runs remains documented in
`benchmarks/results/SCHEMA.md`. Cache files used by direct benchmark
runners are documented in `benchmarks/CACHE_SCHEMA.md`.

External accuracy review packet:
`benchmarks/EXTERNAL_VALIDATION_PACKET.md`.

## Datasets

| Dataset id | Source | Access | Local cache |
|---|---|---|---|
| `llm_aggrefact` | `lytang/LLM-AggreFact`, test split | Gated dataset access; set `HF_TOKEN` or log in locally | score cache in `benchmarks/results/` |
| `halueval` | `pminervini/HaluEval` task files | Public parquet downloads | `benchmarks/.cache/halueval_*.parquet` |
| `truthfulqa` | TruthfulQA CSV cache | Public CSV cache | `benchmarks/.cache/TruthfulQA.csv` |
| `hallubench` | `AuwAuwAuw/HalluBench`, train split | Gated Hugging Face dataset; requires accepted access; CC BY-NC-ND 4.0 | none; raw images, questions, answers, and predictions must stay outside the repo |
| `streaming_good_passages` | In-repo fixture list | Tracked source fixture | none |
| `beir_nfcorpus` | BEIR NFCorpus zip (test split) | Public zip downloads | `benchmarks/data/beir/nfcorpus/` (not committed) |
| `beir_scifact` | BEIR SciFact zip (test split) | Public zip downloads | `benchmarks/data/beir/scifact/` (not committed) |
| `streaming_bad_passages` | In-repo labelled contradiction smoke set | Tracked source fixture | none |
| `latency_fixture_pairs` | In-repo fixture list | Tracked source fixture | none |

## Public Tables

| Table id | Public file | Runner(s) | Primary result artefact |
|---|---|---|---|
| `readme_scoring_pyramid` | `README.md` | `aggrefact_eval.py`, `latency_bench.py`, `streaming_false_halt_bench.py` | `benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json` |
| `readme_aggrefact_leaderboard` | `README.md` | `aggrefact_eval.py` | `benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json` |
| `readme_routed_local_judge` | `README.md` | `gemma_aggrefact_routed.py` | `benchmarks/results/gemma_e4b_q6_routed.json` |
| `benchmark_report_e2e_halueval` | `benchmarks/BENCHMARK_REPORT.md` | `e2e_eval.py`, `halueval_eval.py` | `benchmarks/results/e2e_guardrail.json` |
| `benchmark_report_local_judge` | `benchmarks/BENCHMARK_REPORT.md` | `run_judge_benchmark.py` | `benchmarks/results/judge_bench_summary_200.json` |
| `benchmark_report_hallubench_internal` | `benchmarks/PUBLIC_BENCHMARKS.md` | `hallubench_eval.py` | `benchmarks/results/hallubench_internal_validation.json` |
| `benchmark_report_streaming_false_halt` | `benchmarks/BENCHMARK_REPORT.md` | `streaming_false_halt_bench.py` | `benchmarks/results/streaming_false_halt_heuristic.json` |

## Benchmark Mode Cards

These cards define which public numbers may be compared. A catch-rate card is
not interchangeable with an accuracy card, and a judge-assisted card is not a
pure NLI result.

| Card id | Mode family | Public metric | Boundary |
|---|---|---|---|
| `heuristic_streaming_false_halt` | heuristic | 4.4% false-halt rate; 33.3% halt recall on labelled smoke set | False-halt rate and small smoke-set diagnostics only; not a customer-domain hallucination catch-rate claim. |
| `pure_nli_aggrefact_global` | pure NLI | 75.8% per-dataset mean BA at threshold 0.46 (75.6% at the leaderboard's threshold 0.50) | Global-threshold FactCG result; separate from tuned thresholds. |
| `tuned_threshold_aggrefact` | tuned-threshold NLI | 77.76% per-dataset mean BA | Threshold replay only; not default runtime behaviour. |
| `pure_nli_halueval_e2e` | pure NLI | 46.7% catch, 56.9% precision | End-to-end HaluEval guardrail mode; not an AggreFact row. |
| `hybrid_remote_judge_halueval` | hybrid judge | 90.7% catch, 64.0% FPR | Judge-assisted HaluEval mode; never merge with pure NLI. |
| `local_judge_halueval` | local judge | 33.8% catch, 3.7% FPR, 90.2% precision (200/task, threshold 0.5); QA subset 84.5% catch, 95.5% precision | Local classifier mode; report apart from remote-judge rows. Conservative operating point; the earlier 93.8%/66.3% figure over-flagged summarisation/dialogue and does not reproduce. |
| `hallubench_geospatial_internal` | multimodal geospatial | No public score yet | Gated internal validation harness only; publish no HalluBench score until model provenance, dataset access, and metric review are recorded. |

## Reproduction Commands

BEIR retrieval (grounded() hybrid pipeline; download the BEIR
`nfcorpus` and `scifact` zips into `benchmarks/data/beir/` first):

```bash
python -m benchmarks.beir_competitive_bench
```

BEIR fusion strategies (same datasets, unreranked pipeline, all
shipped fusion methods as shared-index views):

```bash
python -m benchmarks.beir_fusion_bench
```

AggreFact default:

```bash
python -m benchmarks.aggrefact_eval \
  --model yaxili96/FactCG-DeBERTa-v3-Large \
  --threshold 0.46 \
  --save-scores benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json
```

AggreFact threshold replay from cached scores:

```bash
python -m benchmarks.aggrefact_eval \
  --load-scores benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json \
  --sweep
```

Routed local judge:

```bash
python benchmarks/gemma_aggrefact_routed.py \
  --model /path/to/local-judge.gguf \
  --max-samples 29320 \
  --output benchmarks/results/gemma_e4b_q6_routed.json
```

End-to-end HaluEval guardrail:

```bash
python -m benchmarks.e2e_eval \
  --nli \
  --max-samples 100 \
  --threshold 0.35 \
  --soft-limit 0.45 \
  --output-json benchmarks/results/e2e_guardrail.json
```

Direct HaluEval runner:

```bash
python -m benchmarks.halueval_eval 100
```

Local judge comparison:

```bash
python -m benchmarks.run_judge_benchmark \
  --samples 200 \
  --latency-iters 200
```

HalluBench gated geospatial VLM validation:

```bash
HF_TOKEN=<accepted-access-token> python -m benchmarks.hallubench_eval \
  --predictions-jsonl validation/hallubench_predictions.jsonl \
  --output-json benchmarks/results/hallubench_internal_validation.json
```

The prediction JSONL must contain one object per evaluated sample with
`question_id` and `prediction` fields. Keep the JSONL outside tracked public
paths. The runner writes aggregate metrics plus hashes of answers and
predictions; it does not copy raw images, questions, ground-truth answers, or
model predictions into the result file. Until the output is independently
reviewed, this table is a prepared validation path, not a public benchmark
claim.

Streaming false-halt:

```bash
python -m benchmarks.streaming_false_halt_bench
python -m benchmarks.streaming_false_halt_bench --nli
```

Latency runner:

```bash
python -m benchmarks.latency_bench --nli --onnx --iterations 30 --warmup 5
```

## Rules For New Public Tables

When adding a public benchmark table:

1. Add a `[[public_accuracy_tables]]` entry to
   `benchmarks/public_accuracy_manifest.toml`.
2. List every runner file and every result/cache artefact required to
   reproduce the table.
3. Point to a documented cache schema in `benchmarks/CACHE_SCHEMA.md`.
4. Keep approximate preview rows labelled as approximate.
5. Do not copy numbers from notes into public docs without a matching
   result artefact.
6. Do not publish HalluBench scores from gated data until the output is reviewed
   against the dataset license, model provenance, and claim-boundary rules.
