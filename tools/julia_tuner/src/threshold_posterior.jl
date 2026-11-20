# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Piecewise threshold posterior

struct ThresholdInterval
    lower::Float64
    upper::Float64
    prior_weights::NTuple{3,Float64}
    log_mass::Float64
end

function ThresholdInterval(lower::Float64, upper::Float64, agreements::Int)
    # The restricted Beta(2, 2) prior is a mixture of Beta(1, 3), Beta(2, 2)
    # and Beta(3, 1) on the rescaled interval. These positive terms avoid
    # subtracting nearly equal prior CDFs at closely spaced scores.
    left = 2 * lower * (1 - lower)
    middle = lower * (1 - upper) + upper * (1 - lower)
    right = 2 * upper * (1 - upper)
    total = left + middle + right
    log_mass = log(upper - lower) + log(total) + agreements * log(99.0)
    return ThresholdInterval(lower, upper, (left / total, middle / total, right / total),
                             log_mass)
end

function _threshold_intervals(examples::AbstractVector{ScoredExample})
    changes = Dict{Float64,Int}()
    agreements = 0
    for example in examples
        score = example.score
        # Outside this range the prediction stays constant for all interior
        # thresholds, so its likelihood factor cancels in normalization.
        if 0.0 < score < 1.0
            agreements += example.label
            changes[score] = get(changes, score, 0) + (example.label ? -1 : 1)
        end
    end

    intervals = ThresholdInterval[]
    lower = 0.0
    for upper in sort!(collect(keys(changes)))
        push!(intervals, ThresholdInterval(lower, upper, agreements))
        agreements += changes[upper]
        lower = upper
    end
    push!(intervals, ThresholdInterval(lower, 1.0, agreements))
    return intervals
end

function _interval_mean(interval::ThresholdInterval)
    left, middle, right = interval.prior_weights
    return muladd(interval.upper - interval.lower,
                  0.25 * left + 0.5 * middle + 0.75 * right, interval.lower)
end

function _posterior_std(intervals, peak::Float64, total::Float64, center::Float64)
    deviation = 0.0
    normalizer = sqrt(total)
    component_means = (0.25, 0.5, 0.75)
    component_stds = (sqrt(0.0375), sqrt(0.05), sqrt(0.0375))
    for interval in intervals
        scale = exp((interval.log_mass - peak) / 2) / normalizer
        width = interval.upper - interval.lower
        # Pool centered component moments without squaring very small widths.
        for (weight, mean, std) in zip(interval.prior_weights, component_means, component_stds)
            distance = muladd(width, mean, interval.lower - center)
            contribution = scale * sqrt(weight) * hypot(distance, width * std)
            deviation = hypot(deviation, contribution)
        end
    end
    return deviation
end

function _posterior_quantile(intervals, cumulative, probability::Float64)
    index = searchsortedlast(cumulative, probability) + 1
    interval = intervals[index]
    before = index == 1 ? 0.0 : cumulative[index - 1]
    target = (probability - before) / (cumulative[index] - before)
    left, middle, right = interval.prior_weights
    lower, upper = 0.0, 1.0
    for _ in 1:53
        u = (lower + upper) / 2
        # CDFs of Beta(1, 3), Beta(2, 2) and Beta(3, 1), respectively.
        cdf = u * (left * (3 + u * (u - 3)) + u * (middle * (3 - 2u) + right * u))
        if cdf < target
            lower = u
        else
            upper = u
        end
    end
    return muladd(interval.upper - interval.lower, (lower + upper) / 2, interval.lower)
end

"""
    bayesian_threshold_posterior(examples; n_samples, n_warmup, rng)

Compute the posterior for a threshold drawn from ``Beta(2, 2)`` with
conditionally independent Bernoulli labels. A score at least as large as
the threshold predicts a positive label with probability 0.99. Lower
scores predict a positive label with probability 0.01.

Returns `(mean, std, lo, hi, chain)`. The moments and equal-tail 95%
credible interval integrate the full posterior over the score intervals
and are independent of the RNG and sample count. `chain` is a one-chain
`Chains` object with `n_samples` independent draws for `:threshold`.
`n_warmup` remains positive and preserves the legacy chain iteration
labels. Direct sampling requires zero warmup draws.
"""
function bayesian_threshold_posterior(examples::AbstractVector{ScoredExample};
                                      n_samples::Integer = 800,
                                      n_warmup::Integer = 400,
                                      rng::AbstractRNG = Random.default_rng())
    n_samples > 0 && n_warmup > 0 ||
        throw(ArgumentError("n_samples and n_warmup must be positive"))
    isempty(examples) && throw(ArgumentError("no examples supplied"))
    intervals = _threshold_intervals(examples)
    peak = maximum(interval -> interval.log_mass, intervals)
    masses = [exp(interval.log_mass - peak) for interval in intervals]
    cumulative = cumsum(masses)
    total = last(cumulative)
    cumulative ./= total
    center = sum(mass * _interval_mean(interval) for (mass, interval) in zip(masses, intervals)) / total
    std = _posterior_std(intervals, peak, total, center)
    lo = _posterior_quantile(intervals, cumulative, 0.025)
    hi = _posterior_quantile(intervals, cumulative, 0.975)
    samples = [_posterior_quantile(intervals, cumulative, rand(rng)) for _ in 1:n_samples]
    chain = Chains(reshape(samples, :, 1, 1), [:threshold]; start = Int(n_warmup) + 1)
    return (mean = center, std = std, lo = lo, hi = hi, chain = chain)
end
