<!--
SPDX-License-Identifier: AGPL-3.0-or-later
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
| `streaming_good_passages` | In-repo fixture list | Tracked source fixture | none |
| `streaming_bad_passages` | In-repo labelled contradiction smoke set | Tracked source fixture | none |
| `latency_fixture_pairs` | In-repo fixture list | Tracked source fixture | none |

## Public Tables

| Table id | Public file | Runner(s) | Primary result artefact |
|---|---|---|---|
| `readme_scoring_pyramid` | `README.md` | `aggrefact_eval.py`, `latency_bench.py`, `streaming_false_halt_bench.py` | `benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json` |
| `readme_aggrefact_leaderboard` | `README.md` | `aggrefact_eval.py` | `benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json` |
| `readme_routed_local_judge` | `README.md` | `gemma_aggrefact_routed.py` | `benchmarks/results/gemma_e4b_q6_routed.json` |
| `benchmark_report_e2e_halueval` | `benchmarks/BENCHMARK_REPORT.md` | `e2e_eval.py`, `halueval_eval.py` | `benchmarks/results/e2e_guardrail.json` |
| `benchmark_report_local_judge` | `benchmarks/BENCHMARK_REPORT.md` | `run_judge_benchmark.py` | `benchmarks/results/judge_bench_summary_1000.json` |
| `benchmark_report_streaming_false_halt` | `benchmarks/BENCHMARK_REPORT.md` | `streaming_false_halt_bench.py` | `benchmarks/results/streaming_false_halt_heuristic.json` |

## Benchmark Mode Cards

These cards define which public numbers may be compared. A catch-rate card is
not interchangeable with an accuracy card, and a judge-assisted card is not a
pure NLI result.

| Card id | Mode family | Public metric | Boundary |
|---|---|---|---|
| `heuristic_streaming_false_halt` | heuristic | 4.4% false-halt rate; 33.3% halt recall on labelled smoke set | False-halt rate and small smoke-set diagnostics only; not a customer-domain hallucination catch-rate claim. |
| `pure_nli_aggrefact_global` | pure NLI | 75.8% per-dataset mean BA | Global-threshold FactCG result; separate from tuned thresholds. |
| `tuned_threshold_aggrefact` | tuned-threshold NLI | 77.76% per-dataset mean BA | Threshold replay only; not default runtime behaviour. |
| `pure_nli_halueval_e2e` | pure NLI | 46.7% catch, 56.9% precision | End-to-end HaluEval guardrail mode; not an AggreFact row. |
| `hybrid_remote_judge_halueval` | hybrid judge | 90.7% catch, 64.0% FPR | Judge-assisted HaluEval mode; never merge with pure NLI. |
| `local_judge_halueval` | local judge | 93.80% catch, 66.33% FPR | Local classifier mode; report apart from remote-judge rows. |

## Reproduction Commands

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
  --samples 1000 \
  --latency-iters 200
```

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
