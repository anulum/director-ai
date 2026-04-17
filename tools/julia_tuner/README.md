# Director-AI Threshold Tuner (Julia)

Offline companion to the Director-AI scoring pipeline. Given a
labelled set of scorer outputs, it reports:

- the balanced-accuracy-maximising **point threshold**
- a non-parametric **bootstrap 95% CI**
- a **Bayesian posterior** over the threshold (`Turing.jl`, NUTS)

All three are additive diagnostics: the Python production pipeline
is unchanged. Operators run this tool against exported score logs,
inspect the CI band, and commit a threshold back into
`DirectorConfig`.

## Why a separate Julia module?

The Python tuner in `benchmarks/aggrefact_eval.py` returns a point
estimate only. Shipping a threshold without an uncertainty band has
burned us in the past (see the FPR-campaign metric reset of
2026-04-12, where a per-dataset threshold hid a fragile operating
point). Julia's `Turing.jl` gives an off-the-shelf, trustworthy
Bayesian posterior without hand-rolling MCMC in Python.

## Layout

```
tools/julia_tuner/
├── Project.toml                   # Julia project manifest
├── src/DirectorThresholdTuner.jl  # library module
├── bin/tune_threshold.jl          # CLI entry
├── test/runtests.jl               # ~30 test cases
└── README.md                      # this file
```

Related Python helper:

```
tools/prepare_threshold_data.py   # CSV/JSON → tuner JSONL feed
tests/test_prepare_threshold_data.py
```

## Quick start

```bash
# 1. Instantiate once per machine
cd tools/julia_tuner
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 2. Export a feed from a Python eval run
cd ../..
python tools/prepare_threshold_data.py \
    --input benchmarks/results/factcg_summ_per_sample.csv \
    --output /tmp/scores.jsonl

# 3. Tune
julia --project=tools/julia_tuner \
    tools/julia_tuner/bin/tune_threshold.jl \
    --input /tmp/scores.jsonl \
    --output /tmp/threshold.json \
    --bootstrap 2000 --samples 800 --warmup 400 --seed 42

cat /tmp/threshold.json | jq
```

## Input format

One JSON object per line, UTF-8:

```
{"score": 0.74, "label": true,  "source": "aggrefact:summ"}
{"score": 0.31, "label": false, "source": "aggrefact:summ"}
```

- `score` — scorer output. Any real number; typically in `[0, 1]`.
- `label` — ground-truth boolean, `0`/`1`, or one of
  `"true"/"false"/"supported"/"unsupported"/"grounded"/"hallucinated"`.
- `source` — optional, copied through to the output so multi-dataset
  pivots are possible downstream.

## Output format

`director-ai.threshold-tune.v1`:

```json
{
  "schema": "director-ai.threshold-tune.v1",
  "n_examples": 5270,
  "positive_rate": 0.48,
  "point":     {"threshold": 0.515, "balanced_accuracy": 0.7776},
  "bootstrap": {"lo": 0.48, "hi": 0.55, "mean": 0.516, "std": 0.017},
  "bayesian":  {"mean": 0.518, "std": 0.019,
                "lo": 0.481, "hi": 0.557}
}
```

`point.threshold` is the sample optimum; `bootstrap.*` and
`bayesian.*` bracket the uncertainty. When the two intervals agree
within their reported spread, the operating point is robust; when
they disagree, the sample is likely too small or too skewed.

## Tests

```bash
julia --project=tools/julia_tuner -e 'using Pkg; Pkg.test()'
```

The Julia side currently has **9 testsets covering ~30 cases**:
balanced-accuracy edge conditions (perfect separation, inversion,
degenerate labels, boundary), grid search (known optimum, empty,
degenerate, custom grid), bootstrap (CI brackets, narrowing with
data, bad arguments, CI scaling), Bayesian concentration, `tune`
round-trip, JSONL load (valid, missing fields, bad label strings),
and `save_result_json` schema round-trip.

The Python feeder is covered by
`tests/test_prepare_threshold_data.py` — ~27 parametrised cases
(label/score coercion, file-format dispatch, dropped-row
accounting, CLI wiring, custom keys).

## Performance notes

| Stage | 5 000 examples | Notes |
| --- | --- | --- |
| grid search | `≈ 2 ms` | pure Julia loop |
| bootstrap × 2000 | `≈ 5 s` | trivially parallelisable |
| Bayesian NUTS (800+400) | `≈ 20 s` | cold start dominates |

Numbers are indicative — re-run against your own hardware before
quoting them anywhere.

## Not covered

- **Online threshold update** — the tuner is offline; pipe its
  output into `DirectorConfig` manually or via a deployment hook.
- **Multi-threshold (per-class, per-intent) optimisation** —
  extend `tune` with a second dimension if your scorer has mode
  routing.
- **Calibration metrics beyond BA** — ROC-AUC and expected
  calibration error are out of scope; add them in a follow-up
  if the business case emerges.
