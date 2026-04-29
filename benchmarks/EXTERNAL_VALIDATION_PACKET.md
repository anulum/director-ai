<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# External Accuracy Validation Packet

This packet gives an independent evaluator the files, commands, expected
outputs, and claim boundaries needed to validate Director-AI public accuracy
claims without relying on marketing copy.

Machine-readable packet:
`benchmarks/external_validation_packet.toml`

Public benchmark manifest:
`benchmarks/public_accuracy_manifest.toml`

Cache and result schemas:

- `benchmarks/CACHE_SCHEMA.md`
- `benchmarks/results/SCHEMA.md`

## Validation Scope

| Task id | Mode | Claim checked | Primary result |
|---|---|---|---|
| `aggrefact_global_factcg` | pure NLI | Default FactCG global-threshold factual-consistency claim | `benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json` |
| `aggrefact_tuned_threshold_replay` | tuned-threshold NLI | Tuned-threshold replay claim | `benchmarks/results/aggrefact_sweep_lr_low.json` |
| `halueval_nli_e2e` | pure NLI end-to-end | HaluEval guardrail catch/FPR/F1 claim | `benchmarks/results/e2e_guardrail.json` |
| `local_judge_halueval` | local judge | Local borderline-case judge claim | `benchmarks/results/judge_bench_local_judge_1000.json` |
| `streaming_false_halt_heuristic` | heuristic | Streaming false-halt claim | `benchmarks/results/streaming_false_halt_heuristic.json` |

The reviewer may run a smaller smoke subset first, but the final report should
state whether each public claim was checked on the full declared dataset or on a
bounded sample.

## Required Environment Record

Record these fields before running any task:

| Field | Required detail |
|---|---|
| Host | CPU model, RAM, OS, kernel |
| Python | Python version, virtual environment path, package lock hash if used |
| GPU | GPU model, driver, CUDA/ROCm provider, VRAM |
| Models | Model identifier, revision, local path or cache fingerprint |
| Datasets | Dataset source, split, row count, cache path, hash where available |
| Commands | Exact command line for each run |

Store the record as `validation/environment.json`.

## Commands

Default AggreFact pure NLI:

```bash
python -m benchmarks.aggrefact_eval \
  --model yaxili96/FactCG-DeBERTa-v3-Large \
  --threshold 0.46 \
  --save-scores benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json
```

AggreFact tuned-threshold replay:

```bash
python -m benchmarks.aggrefact_eval \
  --load-scores benchmarks/results/aggrefact_yaxili96_FactCG-DeBERTa-v3-Large.json \
  --sweep
```

HaluEval end-to-end NLI:

```bash
python -m benchmarks.e2e_eval \
  --nli \
  --max-samples 100 \
  --threshold 0.35 \
  --soft-limit 0.45 \
  --output-json benchmarks/results/e2e_guardrail.json
```

Local judge:

```bash
python benchmarks/run_judge_benchmark.py --samples 1000
```

Streaming false-halt:

```bash
python -m benchmarks.streaming_false_halt_bench
```

## Required Report Outputs

| Output path | Contents |
|---|---|
| `validation/environment.json` | Runtime and data fingerprint |
| `validation/raw_results/` | Unedited runner outputs |
| `validation/metric_recalculation.md` | Independent metric recomputation notes |
| `validation/failure_cases.jsonl` | Representative disagreements and wrong halts |
| `validation/summary.md` | Pass/fail table per public claim |

## Claim Boundary Rules

The report must keep these claims separate:

1. Pure NLI AggreFact balanced accuracy.
2. Tuned-threshold AggreFact replay.
3. HaluEval end-to-end catch, FPR, precision, and F1.
4. Local judge HaluEval metrics.
5. Heuristic streaming false-halt rate.

Do not convert false-halt rate into hallucination catch rate. Do not merge
judge-assisted rows into pure NLI rows. Do not describe tuned thresholds as
model training.

## Auditor Questions

The final report should answer:

1. Which public claims reproduced within normal hardware and sampling variance?
2. Which claims depend on gated datasets or local model artefacts?
3. Which metrics changed when recomputed from raw labels or TP/FP/TN/FN counts?
4. Which task families produced the highest false-positive and false-negative
   counts?
5. Which claim boundaries need clearer public wording before release?
