#= SPDX-License-Identifier: AGPL-3.0-or-later
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
3. **Bayesian posterior** — `Turing.jl` model with `Beta` prior on
   the threshold and Bernoulli likelihood, yielding a posterior mean
   and 95% credible interval.

The module is deliberately standalone — no FFI into Python, no
runtime dependency from Director-AI on Julia. Operators run it as a
CLI against exported score logs, inspect the CI band, and commit a
threshold back into `DirectorConfig`.
"""
module DirectorThresholdTuner

using Distributions
using JSON3
using Random
using Statistics
using StatsBase
using Turing

export ScoredExample, balanced_accuracy, optimise_threshold,
       bootstrap_threshold_ci, bayesian_threshold_posterior,
       TuneResult, tune, load_scores_jsonl, save_result_json

"""
    ScoredExample

One labelled observation from a scorer run.

Fields:
- `score::Float64` — scorer output in ``[0, 1]`` (higher = more
  confident the response is grounded / supported).
- `label::Bool` — ground-truth label (`true` = grounded,
  `false` = hallucinated). Mapped from 0/1 integers on load.
"""
struct ScoredExample
    score::Float64
    label::Bool
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
    tp = tn = fp = fn = 0
    for ex in examples
        pred = ex.score >= threshold
        if ex.label && pred
            tp += 1
        elseif ex.label && !pred
            fn += 1
        elseif !ex.label && pred
            fp += 1
        else
            tn += 1
        end
    end
    pos = tp + fn
    neg = tn + fp
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

@model function _threshold_model(scores, labels)
    threshold ~ Beta(2.0, 2.0)
    for i in eachindex(scores)
        p = scores[i] >= threshold ? 0.99 : 0.01
        labels[i] ~ Bernoulli(p)
    end
end

"""
    bayesian_threshold_posterior(examples; n_samples, n_warmup, rng)

Fit a simple generative model: a threshold ``t`` drawn from
``Beta(2, 2)`` where each example's label is Bernoulli(0.99 or 0.01)
depending on whether its score exceeds ``t``. MCMC with NUTS yields
the posterior over ``t``.

Returns `(mean, std, lo, hi, chain)` with `lo`/`hi` the 95%
credible-interval bounds and `chain` the underlying `Chains` object
for downstream diagnostics.
"""
function bayesian_threshold_posterior(examples::AbstractVector{ScoredExample};
                                      n_samples::Integer = 800,
                                      n_warmup::Integer = 400,
                                      rng::AbstractRNG = Random.default_rng())
    n_samples > 0 && n_warmup > 0 ||
        throw(ArgumentError("n_samples and n_warmup must be positive"))
    isempty(examples) && throw(ArgumentError("no examples supplied"))
    scores = Float64[ex.score for ex in examples]
    labels = Bool[ex.label for ex in examples]
    model = _threshold_model(scores, labels)
    chain = sample(rng, model, NUTS(), n_samples;
                   discard_initial = n_warmup, progress = false)
    ts = vec(chain[:threshold].data)
    lo = quantile(ts, 0.025)
    hi = quantile(ts, 0.975)
    return (mean = mean(ts), std = std(ts), lo = lo, hi = hi, chain = chain)
end

"""
    TuneResult

Combined output of `tune`. All thresholds are on the ``[0, 1]`` scale.
"""
struct TuneResult
    n_examples::Int
    positive_rate::Float64
    point_threshold::Float64
    point_balanced_accuracy::Float64
    bootstrap_lo::Float64
    bootstrap_hi::Float64
    bootstrap_mean::Float64
    bootstrap_std::Float64
    bayesian_mean::Float64
    bayesian_std::Float64
    bayesian_lo::Float64
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

Read a JSON-lines file. Each line must be an object with at least a
numeric `score` field and a `label` field (accepted as boolean, 0/1,
or the strings "true"/"false"). Other keys are ignored, so the
Python feeder can emit a superset of fields.
"""
function load_scores_jsonl(path::AbstractString)
    out = ScoredExample[]
    open(path, "r") do io
        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            obj = JSON3.read(stripped)
            haskey(obj, :score) || throw(ArgumentError("missing 'score'"))
            haskey(obj, :label) || throw(ArgumentError("missing 'label'"))
            score = Float64(obj[:score])
            raw_label = obj[:label]
            label = _coerce_label(raw_label)
            push!(out, ScoredExample(score, label))
        end
    end
    return out
end

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

Write the `TuneResult` as a JSON object to `path`, replacing any
existing file. The schema is a flat, language-neutral document —
deliberately matched to what the Python side expects from
`tools/prepare_threshold_data.py`.
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
        JSON3.write(io, doc)
    end
    return path
end

end # module
