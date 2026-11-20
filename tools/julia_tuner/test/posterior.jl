# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Threshold posterior numerical regression tests

@testset "full threshold posterior" begin
    scores_path = joinpath(@__DIR__, "..", "..", "..", "benchmarks", "scores", "factcg-anli.json")
    rows = JSON.parsefile(scores_path)["AggreFact-CNN"]
    multimodal = ScoredExample[
        ScoredExample(row[2], !iszero(row[1]))
        for label in (0, 1)
        for row in Iterators.take((row for row in rows if row[1] == label), 8)
    ]
    separable = ScoredExample[
        ScoredExample(percent / 100, percent > 50)
        for percent in Iterators.flatten((15:2:45, 55:2:85))
    ]
    # Mean, standard deviation and equal-tail quantiles from independent
    # high-precision integration of the beta density times the likelihood.
    for (examples, reference, mass_below_half) in (
        (multimodal, (0.8092590290144229, 0.31814980800707654,
                      0.12333050586352708, 0.9994250598923783), 0.26768067712288734),
        (separable, (0.5, 0.029025367879868758,
                     0.4523227946056515, 0.5476772053943485), 0.5),
    )
        for seed in (7, 42, 123)
            result = bayesian_threshold_posterior(examples; n_samples = 8000,
                                                  rng = MersenneTwister(seed))
            @test all(isapprox.((result.mean, result.std, result.lo, result.hi), reference;
                                atol = 5e-12, rtol = 0))
            draws = vec(result.chain[:threshold].data)
            @test length(draws) == 8000
            @test abs(count(<(0.5), draws) / length(draws) - mass_below_half) < 0.025
            @test abs(sum(draws) / length(draws) - reference[1]) < 0.018
        end
    end

    first_run = bayesian_threshold_posterior(multimodal; n_samples = 10, n_warmup = 1,
                                            rng = MersenneTwister(7))
    repeat_run = bayesian_threshold_posterior(reverse(multimodal); n_samples = 20,
                                             n_warmup = 300, rng = MersenneTwister(7))
    @test (first_run.mean, first_run.std, first_run.lo, first_run.hi) ==
          (repeat_run.mean, repeat_run.std, repeat_run.lo, repeat_run.hi)
    @test vec(first_run.chain[:threshold].data) == vec(repeat_run.chain[:threshold].data)[1:10]
    @test range(first_run.chain) == 2:11
    @test range(repeat_run.chain) == 301:320
    @test_throws ArgumentError bayesian_threshold_posterior(multimodal; n_samples = 0)
    @test_throws ArgumentError bayesian_threshold_posterior(multimodal; n_warmup = 0)
end

@testset "posterior score boundaries" begin
    # Two opposite labels at the same score make the likelihood constant.
    # Scores outside the threshold domain only add constants.
    examples = [
        ScoredExample(0.4, false), ScoredExample(0.4, true),
        ScoredExample(0.0, true), ScoredExample(1.0, false),
        ScoredExample(-0.2, false), ScoredExample(1.2, true),
    ]
    result = bayesian_threshold_posterior(examples; n_samples = 1, rng = MersenneTwister(1))
    @test result.mean ≈ 0.5
    @test result.std ≈ sqrt(0.05)
    @test result.lo ≈ 0.09429932405024608
    @test result.hi ≈ 1 - result.lo
    @test length(result.chain[:threshold]) == 1

    # Both neighbouring intervals have negligible posterior mass and the
    # useful interval is narrower than a reliable difference of prior CDFs.
    lower, upper = 0.5 - eps(0.5), 0.5 + eps(0.5)
    narrow = repeat([ScoredExample(lower, false), ScoredExample(upper, true)], 500)
    concentrated = bayesian_threshold_posterior(narrow; n_samples = 16, rng = MersenneTwister(2))
    @test concentrated.mean ≈ 0.5
    @test lower <= concentrated.lo <= concentrated.hi <= upper
    @test concentrated.std ≈ (upper - lower) / sqrt(12)
    @test all(t -> lower <= t <= upper, concentrated.chain[:threshold])
end
