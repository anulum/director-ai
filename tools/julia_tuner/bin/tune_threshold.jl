#!/usr/bin/env julia
#= SPDX-License-Identifier: AGPL-3.0-or-later
   Commercial licence available
   © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
   © Code 2020–2026 Miroslav Šotek. All rights reserved.
   ORCID: 0009-0009-3560-0851
   Contact: www.anulum.li | protoscience@anulum.li
   Director-Class AI — Threshold tuner CLI =#

using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))

using ArgParse
using Random

include(joinpath(@__DIR__, "..", "src", "DirectorThresholdTuner.jl"))
using .DirectorThresholdTuner

function _parse()
    s = ArgParseSettings(
        prog = "tune_threshold.jl",
        description = "Optimise a scoring threshold with bootstrap CI and " *
                      "Bayesian posterior.",
    )
    @add_arg_table! s begin
        "--input", "-i"
            help = "JSONL file of {score, label} records"
            required = true
        "--output", "-o"
            help = "Output JSON path"
            required = true
        "--bootstrap", "-b"
            help = "Bootstrap resamples"
            arg_type = Int
            default = 2000
        "--samples"
            help = "Bayesian MCMC samples (post-warmup)"
            arg_type = Int
            default = 800
        "--warmup"
            help = "Bayesian MCMC warmup"
            arg_type = Int
            default = 400
        "--seed"
            help = "RNG seed (omit for non-deterministic)"
            arg_type = Int
            default = -1
    end
    return parse_args(s)
end

function main()
    args = _parse()
    rng = args["seed"] >= 0 ? MersenneTwister(args["seed"]) : Random.default_rng()
    examples = load_scores_jsonl(args["input"])
    result = tune(examples;
                  rng = rng,
                  n_bootstrap = args["bootstrap"],
                  n_bayes_samples = args["samples"],
                  n_bayes_warmup = args["warmup"])
    save_result_json(args["output"], result)
    println("wrote $(args["output"]) — point=$(result.point_threshold) " *
            "BA=$(round(result.point_balanced_accuracy; digits = 4)) " *
            "bootstrap=[$(round(result.bootstrap_lo; digits = 3)), " *
            "$(round(result.bootstrap_hi; digits = 3))] " *
            "bayes_mean=$(round(result.bayesian_mean; digits = 3))")
end

abspath(PROGRAM_FILE) == @__FILE__ && main()
