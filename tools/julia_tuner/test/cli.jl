# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Threshold tuner process contracts

using Test
using JSON3

"""
    run_tuner(script, arguments...)

Run the maintained Julia entry point in a separate native process and retain
its status, standard output and diagnostic output.
"""
function run_tuner(script::AbstractString, arguments::AbstractString...)
    output = IOBuffer()
    errors = IOBuffer()
    command = `$(Base.julia_cmd()) --startup-file=no $script $arguments`
    process = run(pipeline(ignorestatus(command), stdout = output, stderr = errors))
    return (status = process.exitcode, output = String(take!(output)),
            errors = String(take!(errors)))
end

@testset "threshold tuner CLI" begin
    script = normpath(joinpath(@__DIR__, "..", "bin", "tune_threshold.jl"))
    help = run_tuner(script, "--help")
    @test help.status == 0
    @test occursin("--input", help.output)
    @test occursin("--output", help.output)

    missing = run_tuner(script)
    @test missing.status != 0
    @test occursin("required", lowercase(missing.errors))

    probe = "include(ARGS[1]); @assert !isdefined(Main, :DirectorThresholdTuner)"
    @test success(`$(Base.julia_cmd()) --startup-file=no -e $probe $script`)

    scores_path = joinpath(@__DIR__, "..", "..", "..", "benchmarks", "scores", "factcg-anli.json")
    scores = JSON3.read(read(scores_path, String))["AggreFact-CNN"]
    mktempdir() do directory
        input = joinpath(directory, "scores.jsonl")
        open(input, "w") do stream
            for label in (0, 1)
                selected = Iterators.take((row for row in scores if row[1] == label), 8)
                for row in selected
                    println(stream, JSON3.write((score = row[2], label = label)))
                end
            end
        end

        for (filename, seed) in (("seeded.json", ("--seed", "42")), ("unseeded.json", ()))
            output = joinpath(directory, filename)
            run = run_tuner(script, "--input", input, "--output", output, "--bootstrap", "100",
                            "--samples", "40", "--warmup", "20", seed...)
            @test run.status == 0
            @test occursin("wrote $output", run.output)
            result = JSON3.read(read(output, String))
            @test result.schema == "director-ai.threshold-tune.v1"
            @test result.n_examples == 16
            @test result.positive_rate == 0.5
            @test result.point.threshold ≈ 0.005
            @test result.point.balanced_accuracy == 0.625
            @test 0 <= result.bootstrap.lo <= result.bootstrap.hi <= 1
            @test 0 <= result.bayesian.lo <= result.bayesian.hi <= 1
        end
    end
end
