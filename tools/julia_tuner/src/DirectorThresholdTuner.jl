#= SPDX-License-Identifier: Apache-2.0
   Commercial licence available
   © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
   © Code 2020–2026 Miroslav Šotek. All rights reserved.
   ORCID: 0009-0009-3560-0851
   Contact: www.anulum.li | protoscience@anulum.li
   Director-Class AI — Julia threshold tuner =#

"""
    DirectorThresholdTuner

Offline analytics companion for Director-AI. Given historical
`(score, label)` pairs — produced by any scorer backend against a
labelled eval set (AggreFact, HaluEval, custom KB) — compute:

1. **Grid-search optimum** — threshold that maximises balanced
   accuracy on the sample.
2. **Bootstrap 95% CI** — non-parametric resampling over the
   per-example pairs to bound threshold uncertainty.
3. **Bayesian posterior** — piecewise integration of a `Beta` threshold
   prior and Bernoulli likelihood, yielding posterior moments and a
   95% credible interval.

The module is deliberately standalone — no FFI into Python, no
runtime dependency from Director-AI on Julia. Operators run it as a
CLI against exported score logs, inspect the CI band, and commit a
threshold back into `DirectorConfig`.
"""
module DirectorThresholdTuner

import JSON
using MCMCChains: Chains
using Random
using Statistics

export ScoredExample, balanced_accuracy, optimise_threshold,
       bootstrap_threshold_ci, bayesian_threshold_posterior,
       TuneResult, tune, load_scores_jsonl, save_result_json

"""
    ScoredExample(score, label)

One labelled observation from a scorer run.

The constructor accepts numeric scores convertible to a finite `Float64`.
Boolean scores raise `ArgumentError`.

Fields:
- `score::Float64` — finite scorer output. Higher values indicate stronger
  grounding. Finite values outside ``[0, 1]`` retain their original value.
- `label::Bool` — ground-truth label (`true` = grounded,
  `false` = hallucinated). Mapped from 0/1 integers on load.
"""
struct ScoredExample
    "Scorer output. Higher scores indicate stronger grounding."
    score::Float64
    "Ground-truth label. A true value marks a supported example."
    label::Bool

    """Construct an observation with a finite score."""
    function ScoredExample(score, label)
        score isa Bool && throw(ArgumentError("score must be numeric"))
        converted = convert(Float64, score)
        isfinite(converted) || throw(ArgumentError("score must be finite"))
        return new(converted, label)
    end
end

"""
    balanced_accuracy(examples, threshold)

Standard balanced-accuracy metric: the arithmetic mean of the true
positive rate and the true negative rate at the given threshold.

Returns `NaN` when either class is empty, matching `scikit-learn`'s
behaviour for degenerate label distributions.
"""
function balanced_accuracy(examples::AbstractVector{ScoredExample},
                           threshold::Real)::Float64
    tp = tn = pos = neg = 0
    for ex in examples
        pred = ex.score >= threshold
        if ex.label
            pos += 1
            tp += pred
        else
            neg += 1
            tn += !pred
        end
    end
    if pos == 0 || neg == 0
        return NaN
    end
    tpr = tp / pos
    tnr = tn / neg
    return (tpr + tnr) / 2
end

"""
    optimise_threshold(examples; grid)

Grid-search the threshold that maximises balanced accuracy. `grid`
defaults to a 201-point linear grid over ``[0, 1]`` (0.5% steps).

Returns `(threshold, balanced_accuracy)`.
"""
function optimise_threshold(examples::AbstractVector{ScoredExample};
                            grid::AbstractRange = range(0.0, 1.0; length = 201))
    isempty(examples) && throw(ArgumentError("no examples supplied"))
    best_t = first(grid)
    best_ba = -Inf
    for t in grid
        ba = balanced_accuracy(examples, t)
        if !isnan(ba) && ba > best_ba
            best_ba = ba
            best_t = t
        end
    end
    best_ba == -Inf && throw(ArgumentError("degenerate label set — all one class"))
    return (threshold = float(best_t), balanced_accuracy = best_ba)
end

"""
    bootstrap_threshold_ci(examples; n_resamples, ci, rng)

Non-parametric bootstrap over `(score, label)` pairs. For each
resample, `optimise_threshold` is called; the returned thresholds
form the empirical distribution whose `ci`-level quantiles give the
confidence interval.

- `n_resamples` — default 2000. The law of large numbers suggests
  > 1000 for stable 95% quantiles.
- `ci` — default 0.95.

Returns a named tuple `(lo, hi, mean, std, samples)` with `samples`
being the raw bootstrap vector (useful for plotting).
"""
function bootstrap_threshold_ci(examples::AbstractVector{ScoredExample};
                                n_resamples::Integer = 2000,
                                ci::Real = 0.95,
                                rng::AbstractRNG = Random.default_rng())
    n_resamples > 0 || throw(ArgumentError("n_resamples must be positive"))
    0 < ci < 1 || throw(ArgumentError("ci must lie in (0, 1)"))
    n = length(examples)
    n > 0 || throw(ArgumentError("no examples supplied"))
    samples = Vector{Float64}(undef, n_resamples)
    for i in 1:n_resamples
        idx = rand(rng, 1:n, n)
        resample = examples[idx]
        try
            r = optimise_threshold(resample)
            samples[i] = r.threshold
        catch err
            err isa ArgumentError || rethrow(err)
            samples[i] = NaN
        end
    end
    valid = filter(!isnan, samples)
    length(valid) >= max(50, div(n_resamples, 4)) ||
        error("too few non-degenerate resamples ($(length(valid))/$n_resamples)")
    lo_q = (1 - ci) / 2
    hi_q = 1 - lo_q
    lo = quantile(valid, lo_q)
    hi = quantile(valid, hi_q)
    return (lo = lo, hi = hi,
            mean = mean(valid), std = std(valid),
            samples = valid)
end

include("threshold_posterior.jl")

"""
    TuneResult

Combined output of `tune`. All thresholds are on the ``[0, 1]`` scale.
"""
struct TuneResult
    "Number of labelled observations used in the tune."
    n_examples::Int
    "Fraction of observations with a positive label."
    positive_rate::Float64
    "Grid threshold with the highest sampled balanced accuracy."
    point_threshold::Float64
    "Balanced accuracy at the selected grid threshold."
    point_balanced_accuracy::Float64
    "Lower 95% bootstrap confidence bound for the threshold."
    bootstrap_lo::Float64
    "Upper 95% bootstrap confidence bound for the threshold."
    bootstrap_hi::Float64
    "Mean of the valid resampled thresholds."
    bootstrap_mean::Float64
    "Sample standard deviation of the valid resampled thresholds."
    bootstrap_std::Float64
    "Mean of the integrated threshold posterior."
    bayesian_mean::Float64
    "Standard deviation of the integrated threshold posterior."
    bayesian_std::Float64
    "Lower equal-tail 95% posterior credible bound."
    bayesian_lo::Float64
    "Upper equal-tail 95% posterior credible bound."
    bayesian_hi::Float64
end

"""
    tune(examples; rng, n_bootstrap, n_bayes_samples, n_bayes_warmup)

Orchestrates all three analyses and returns a `TuneResult`. This is
the single call sites should use from Python or the CLI.
"""
function tune(examples::AbstractVector{ScoredExample};
              rng::AbstractRNG = Random.default_rng(),
              n_bootstrap::Integer = 2000,
              n_bayes_samples::Integer = 800,
              n_bayes_warmup::Integer = 400)
    point = optimise_threshold(examples)
    boot = bootstrap_threshold_ci(examples; n_resamples = n_bootstrap, rng = rng)
    bayes = bayesian_threshold_posterior(examples;
                                         n_samples = n_bayes_samples,
                                         n_warmup = n_bayes_warmup, rng = rng)
    pos_rate = count(ex -> ex.label, examples) / length(examples)
    return TuneResult(length(examples), pos_rate,
                      point.threshold, point.balanced_accuracy,
                      boot.lo, boot.hi, boot.mean, boot.std,
                      bayes.mean, bayes.std, bayes.lo, bayes.hi)
end

"""
    load_scores_jsonl(path) -> Vector{ScoredExample}

Read a JSON-lines file. Each line must be an object with a `score` field
containing a JSON number representable as a finite `Float64` and a `label` field
(accepted as boolean, 0/1, or the strings "true"/"false"). Score and JSON
syntax errors report the physical input line. Other keys are ignored so the
Python feeder can emit a superset of fields.
"""
function load_scores_jsonl(path::AbstractString)
    out = ScoredExample[]
    open(path, "r") do io
        for (lineno, line) in enumerate(eachline(io))
            stripped = strip(line)
            isempty(stripped) && continue
            obj = try
                JSON.parse(stripped)
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError("line $lineno: invalid JSON"))
            end
            obj isa AbstractDict || throw(ArgumentError("expected a JSON object"))
            haskey(obj, "score") || throw(ArgumentError("missing 'score'"))
            haskey(obj, "label") || throw(ArgumentError("missing 'label'"))
            raw_score = obj["score"]
            raw_score isa Real || throw(ArgumentError("line $lineno: score must be numeric"))
            label = _coerce_label(obj["label"])
            example = try
                ScoredExample(raw_score, label)
            catch err
                err isa ArgumentError || rethrow()
                throw(ArgumentError("line $lineno: $(err.msg)"))
            end
            push!(out, example)
        end
    end
    return out
end

"""Interpret boolean, numeric and documented string label encodings."""
function _coerce_label(value)
    if value isa Bool
        return value
    elseif value isa Integer
        return value != 0
    elseif value isa Real
        return value != 0.0
    elseif value isa AbstractString
        lowered = lowercase(String(value))
        lowered in ("true", "1", "yes", "grounded", "supported") && return true
        lowered in ("false", "0", "no", "hallucinated", "unsupported") && return false
    end
    throw(ArgumentError("cannot coerce label value $(repr(value)) to Bool"))
end

"""
    save_result_json(path, result)

Write a `director-ai.threshold-tune.v1` JSON object to `path`, replacing
any existing file. The document records the point, bootstrap and Bayesian
estimates for the supplied result.
"""
function save_result_json(path::AbstractString, result::TuneResult)
    doc = Dict(
        "schema" => "director-ai.threshold-tune.v1",
        "n_examples" => result.n_examples,
        "positive_rate" => result.positive_rate,
        "point" => Dict(
            "threshold" => result.point_threshold,
            "balanced_accuracy" => result.point_balanced_accuracy,
        ),
        "bootstrap" => Dict(
            "lo" => result.bootstrap_lo,
            "hi" => result.bootstrap_hi,
            "mean" => result.bootstrap_mean,
            "std" => result.bootstrap_std,
        ),
        "bayesian" => Dict(
            "mean" => result.bayesian_mean,
            "std" => result.bayesian_std,
            "lo" => result.bayesian_lo,
            "hi" => result.bayesian_hi,
        ),
    )
    open(path, "w") do io
        JSON.json(io, doc)
    end
    return path
end

end # module
