<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- Director-AI — Julia threshold tuner usage -->

# Director-AI Threshold Tuner (Julia)

Offline companion to the Director-AI scoring pipeline. Given a
labelled set of scorer outputs, it reports:

- the balanced-accuracy-maximising **point threshold**
- a non-parametric **bootstrap 95% CI**
- a **Bayesian posterior** over the threshold, integrated across the score intervals

All three are additive diagnostics: the Python production pipeline
is unchanged. Operators run this tool against exported score logs,
inspect the CI band, and commit a threshold back into
`DirectorConfig`.

## Why a separate Julia module?

The Python tuner in `benchmarks/aggrefact_eval.py` returns a point
estimate only. Shipping a threshold without an uncertainty band has
burned us in the past (see the FPR-campaign metric reset of
2026-04-12, where a per-dataset threshold hid a fragile operating
point). The threshold has a Beta(2, 2) prior. A matching label has likelihood
0.99 and a mismatching label has likelihood 0.01. The likelihood is
constant between scores, so the tuner integrates across those intervals
and can draw independently from the entire posterior.

## Layout

```
tools/julia_tuner/
├── Project.toml                   # Julia project manifest
├── src/                          # library module and posterior integration
├── bin/tune_threshold.jl          # CLI entry
├── test/                          # native library and CLI tests
└── README.md                      # this file
```

Related Python helper:

```
tools/prepare_threshold_data.py   # CSV/JSON → tuner JSONL feed
tests/test_prepare_threshold_data.py
```

## Quick start

Use Julia 1.13.4 or a later Julia 1.x release with the checked-in project
and manifest. CI tests the CLI and builds the API reference on Julia 1.13.4.
Run `Pkg.instantiate()` after updating the checkout to install the locked
JSON.jl and MCMCChains dependencies.

After instantiating the project, run `julia tools/julia_tuner/bin/tune_threshold.jl --help`
from the repository root to see the available options. Argument parsing runs
before the tuner library loads.

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
    --bootstrap 2000 --samples 800 --seed 42

cat /tmp/threshold.json | jq
```

## Input format

One JSON object per line, UTF-8:

```
{"score": 0.74, "label": true,  "source": "aggrefact:summ"}
{"score": 0.31, "label": false, "source": "aggrefact:summ"}
```

- `score` — a JSON number representable as a finite `Float64`, typically
  in `[0, 1]`. Numeric `0` and `1` are valid scores. The JSONL loader
  rejects boolean scores and reports the line for invalid scores.
  The Julia `ScoredExample` constructor also rejects NaN and infinities.
- `label` — ground-truth boolean, `0`/`1`, or one of
  `"true"/"false"/"supported"/"unsupported"/"grounded"/"hallucinated"`.
- `source` — optional input metadata. The tuner ignores this field.

Finite scores below 0 or above 1 retain their value. With the default grid
they always predict the same class at every threshold and contribute a
constant likelihood factor to the Bayesian posterior.

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

`point.threshold` maximises balanced accuracy on the supplied sample.
The bootstrap reports variation under resampling. The Bayesian interval
is the equal-tail 95% interval of the specified threshold model and can
span multiple modes. The Bayesian mean, standard deviation and interval
are independent of the seed and sample budget. The seed controls the
bootstrap resamples and the independent posterior draws returned as a
`Chains` object by the Julia API. The legacy positive `--warmup` option
and `n_warmup` keyword preserve the iteration numbering of the Julia
chain. The sampler has no warmup. For repeatable draws, use the same Julia
version, manifest, data and seed.

## Tests

```bash
julia --project=tools/julia_tuner -e 'using Pkg; Pkg.test()'
```

The native Julia suite covers
balanced-accuracy edge conditions (perfect separation, inversion,
degenerate labels, boundary), grid search (known optimum, empty,
degenerate, custom grid), bootstrap (CI brackets, narrowing with
data, bad arguments, CI scaling), Bayesian concentration and multimodal integration, `tune`
round-trip, JSONL load (valid, missing fields, bad label strings),
`save_result_json` schema round-trip, and the CLI subprocess contract.

The Python feeder is covered by
`tests/test_prepare_threshold_data.py` plus
`tests/test_prepare_threshold_data_real_surface.py`: unit coverage for
label/score coercion, file-format dispatch, dropped-row accounting,
and custom keys, with subprocess coverage for the production
`tools/prepare_threshold_data.py` CLI converting CSV/JSON inputs into
Julia tuner JSONL and preserving output atomicity on input errors.

## Measure runtime

Time the quick-start command with a fixed JSONL input, seed and sampling
budget. Measure fresh-process runs for CLI latency and warm calls to the
individual Julia functions for grid search, bootstrap and sampling timings.
Record the build, lock and thread counts alongside the measurements.

## Not covered

- **Online threshold update** — the tuner is offline; pipe its
  output into `DirectorConfig` manually or via a deployment hook.
- **Multi-threshold (per-class, per-intent) optimisation** —
  extend `tune` with a second dimension if your scorer has mode
  routing.
- **Calibration metrics beyond BA** — ROC-AUC and expected
  calibration error are out of scope; add them in a follow-up
  if the business case emerges.
