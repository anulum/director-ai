#= SPDX-License-Identifier: Apache-2.0
   Commercial licence available
   © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
   © Code 2020–2026 Miroslav Šotek. All rights reserved.
   ORCID: 0009-0009-3560-0851
   Contact: www.anulum.li | protoscience@anulum.li
   Director-Class AI — Julia threshold tuner tests =#

using Test
using Random
import JSON
using MCMCChains: Chains

include(joinpath(@__DIR__, "..", "src", "DirectorThresholdTuner.jl"))
using .DirectorThresholdTuner

# Synthetic generator: Gaussian scores whose class means straddle
# ``true_threshold`` by a small margin. Classes overlap, so there is a
# well-defined Bayes-optimal threshold at the boundary crossing point
# — which is exactly ``true_threshold`` when the noise is symmetric
# and class-balanced.
function _synthetic(n::Int, true_threshold::Float64, noise::Float64;
                    rng = MersenneTwister(42))
    out = ScoredExample[]
    shift = 1.5 * noise
    for _ in 1:n
        label = rand(rng) < 0.5
        base = label ? true_threshold + shift : true_threshold - shift
        score = clamp(base + noise * randn(rng), 0.0, 1.0)
        push!(out, ScoredExample(score, label))
    end
    return out
end

@testset "balanced_accuracy" begin
    @testset "perfect separation gives BA=1" begin
        ex = [ScoredExample(0.9, true), ScoredExample(0.1, false)]
        @test balanced_accuracy(ex, 0.5) ≈ 1.0
    end
    @testset "inverted classifier gives BA=0" begin
        ex = [ScoredExample(0.9, false), ScoredExample(0.1, true)]
        @test balanced_accuracy(ex, 0.5) ≈ 0.0
    end
    @testset "degenerate label set returns NaN" begin
        ex = [ScoredExample(0.9, true), ScoredExample(0.1, true)]
        @test isnan(balanced_accuracy(ex, 0.5))
    end
    @testset "boundary condition at threshold" begin
        # pred = score >= t, so a score exactly at t predicts positive.
        ex = [ScoredExample(0.5, true), ScoredExample(0.4, false)]
        @test balanced_accuracy(ex, 0.5) ≈ 1.0
    end
end

@testset "optimise_threshold" begin
    @testset "recovers known optimum on clean data" begin
        ex = _synthetic(1000, 0.6, 0.05; rng = MersenneTwister(1))
        r = optimise_threshold(ex)
        @test abs(r.threshold - 0.6) <= 0.08
        @test r.balanced_accuracy > 0.9
    end
    @testset "empty input rejected" begin
        @test_throws ArgumentError optimise_threshold(ScoredExample[])
    end
    @testset "degenerate labels rejected" begin
        ex = [ScoredExample(0.9, true), ScoredExample(0.5, true)]
        @test_throws ArgumentError optimise_threshold(ex)
    end
    @testset "custom grid honoured" begin
        ex = _synthetic(200, 0.5, 0.02; rng = MersenneTwister(2))
        r = optimise_threshold(ex; grid = 0.0:0.25:1.0)
        @test r.threshold in 0.0:0.25:1.0
    end
end

@testset "bootstrap_threshold_ci" begin
    @testset "CI brackets the point estimate" begin
        ex = _synthetic(300, 0.5, 0.05; rng = MersenneTwister(3))
        pt = optimise_threshold(ex)
        boot = bootstrap_threshold_ci(ex; n_resamples = 300,
                                      rng = MersenneTwister(10))
        @test boot.lo <= pt.threshold <= boot.hi
        @test boot.hi - boot.lo < 0.3
    end
    @testset "narrow CI with more data" begin
        wide = bootstrap_threshold_ci(
            _synthetic(100, 0.5, 0.05; rng = MersenneTwister(4));
            n_resamples = 300, rng = MersenneTwister(11))
        narrow = bootstrap_threshold_ci(
            _synthetic(1000, 0.5, 0.05; rng = MersenneTwister(4));
            n_resamples = 300, rng = MersenneTwister(12))
        @test (narrow.hi - narrow.lo) <= (wide.hi - wide.lo)
    end
    @testset "non-positive n_resamples rejected" begin
        ex = _synthetic(50, 0.5, 0.05; rng = MersenneTwister(5))
        @test_throws ArgumentError bootstrap_threshold_ci(ex; n_resamples = 0)
    end
    @testset "ci out of range rejected" begin
        ex = _synthetic(50, 0.5, 0.05; rng = MersenneTwister(5))
        @test_throws ArgumentError bootstrap_threshold_ci(ex; ci = 1.5)
        @test_throws ArgumentError bootstrap_threshold_ci(ex; ci = 0.0)
    end
    @testset "ci=0.5 narrower than ci=0.95" begin
        ex = _synthetic(300, 0.5, 0.05; rng = MersenneTwister(6))
        b50 = bootstrap_threshold_ci(ex; n_resamples = 300, ci = 0.5,
                                     rng = MersenneTwister(13))
        b95 = bootstrap_threshold_ci(ex; n_resamples = 300, ci = 0.95,
                                     rng = MersenneTwister(13))
        @test (b50.hi - b50.lo) <= (b95.hi - b95.lo) + 1e-9
    end
end

@testset "bayesian_threshold_posterior" begin
    @testset "posterior concentrates on the true threshold" begin
        ex = _synthetic(400, 0.5, 0.03; rng = MersenneTwister(7))
        bayes = bayesian_threshold_posterior(ex; n_samples = 200,
                                             n_warmup = 200,
                                             rng = MersenneTwister(14))
        @test 0.35 <= bayes.mean <= 0.65
        @test bayes.lo < bayes.mean < bayes.hi
        @test 0.0 <= bayes.lo && bayes.hi <= 1.0
        @test bayes.chain isa Chains
        @test length(bayes.chain[:threshold]) == 200
    end
    @testset "empty input rejected" begin
        @test_throws ArgumentError bayesian_threshold_posterior(ScoredExample[])
    end
end

@testset "tune" begin
    @testset "round-trip produces a TuneResult" begin
        ex = _synthetic(200, 0.55, 0.05; rng = MersenneTwister(8))
        r = tune(ex; n_bootstrap = 200, n_bayes_samples = 150,
                 n_bayes_warmup = 150, rng = MersenneTwister(15))
        @test r isa TuneResult
        @test r.n_examples == 200
        @test 0.0 <= r.positive_rate <= 1.0
        @test r.bootstrap_lo <= r.bootstrap_hi
        @test r.bayesian_lo <= r.bayesian_hi
    end
end

@testset "ScoredExample score validity" begin
    for score in (NaN, Inf, -Inf, big"1e999")
        @test_throws ArgumentError("score must be finite") ScoredExample(score, true)
    end
    extremes = [ScoredExample(-0.25, false), ScoredExample(1.25, true)]
    @test extremes[1].score == -0.25
    @test extremes[2].score == 1.25
    @test balanced_accuracy(extremes, 0.0) == balanced_accuracy(extremes, 1.0) == 1.0
end

@testset "load_scores_jsonl" begin
    @testset "parses valid records" begin
        mktempdir() do dir
            p = joinpath(dir, "scores.jsonl")
            open(p, "w") do io
                println(io, JSON.json(Dict("score" => 0.9, "label" => true)))
                println(io, JSON.json(Dict("score" => 0.2, "label" => 0)))
                println(io, "")  # blank line skipped
                println(io,
                        JSON.json(Dict("score" => 0.7, "label" => "grounded")))
                println(io, "{\"score\":0,\"label\":false}")
                println(io, "{\"score\":1,\"label\":true}")
            end
            ex = load_scores_jsonl(p)
            @test length(ex) == 5
            @test ex[1] == ScoredExample(0.9, true)
            @test ex[2] == ScoredExample(0.2, false)
            @test ex[3] == ScoredExample(0.7, true)
            @test ex[4] == ScoredExample(0.0, false)
            @test ex[5] == ScoredExample(1.0, true)
        end
    end
    @testset "rejects records missing fields" begin
        mktempdir() do dir
            p = joinpath(dir, "bad.jsonl")
            for (record, missing) in ((Dict("score" => 0.9), "label"),
                                      (Dict("label" => true), "score"))
                write(p, JSON.json(record) * "\n")
                @test_throws ArgumentError("missing '$missing'") load_scores_jsonl(p)
            end
        end
    end
    @testset "rejects non-object records" begin
        mktempdir() do dir
            p = joinpath(dir, "array.jsonl")
            write(p, "[0.9, true]\n")
            @test_throws ArgumentError load_scores_jsonl(p)
        end
    end
    @testset "reports the physical line of invalid scores" begin
        mktempdir() do dir
            p = joinpath(dir, "bad-scores.jsonl")
            for (score, message) in (
                ("NaN", "line 3: invalid JSON"),
                ("Infinity", "line 3: invalid JSON"),
                ("-Infinity", "line 3: invalid JSON"),
                ("1e999", "line 3: score must be finite"),
                ("-1e999", "line 3: score must be finite"),
                ("\"NaN\"", "line 3: score must be numeric"),
                ("true", "line 3: score must be numeric"),
                ("false", "line 3: score must be numeric"),
            )
                write(p, "{\"score\":0.8,\"label\":true}\n\n{\"score\":$score,\"label\":false}\n")
                @test_throws ArgumentError(message) load_scores_jsonl(p)
            end
            write(p, "{\"score\":-0.25,\"label\":false}\n{\"score\":1.25,\"label\":true}\n")
            @test getfield.(load_scores_jsonl(p), :score) == [-0.25, 1.25]
        end
    end
    @testset "rejects bad label strings" begin
        mktempdir() do dir
            p = joinpath(dir, "bad.jsonl")
            open(p, "w") do io
                println(io,
                        JSON.json(Dict("score" => 0.9, "label" => "maybe")))
            end
            @test_throws ArgumentError load_scores_jsonl(p)
        end
    end
end

@testset "save_result_json" begin
    @testset "round-trip schema" begin
        mktempdir() do dir
            ex = _synthetic(120, 0.5, 0.05; rng = MersenneTwister(9))
            r = tune(ex; n_bootstrap = 100, n_bayes_samples = 100,
                     n_bayes_warmup = 100, rng = MersenneTwister(16))
            out = joinpath(dir, "result.json")
            save_result_json(out, r)
            doc = JSON.parsefile(out)
            @test doc["schema"] == "director-ai.threshold-tune.v1"
            @test doc["n_examples"] == 120
            @test doc["point"]["balanced_accuracy"] >= 0.0
            @test doc["bootstrap"]["lo"] <= doc["bootstrap"]["hi"]
            @test doc["bayesian"]["lo"] <= doc["bayesian"]["hi"]
        end
    end
end

include("posterior.jl")
include("cli.jl")
