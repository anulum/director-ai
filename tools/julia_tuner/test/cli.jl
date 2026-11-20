# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Threshold tuner process contracts

using Test
import JSON

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
    @test occursin("--warmup", help.output)

    missing = run_tuner(script)
    @test missing.status != 0
    @test occursin("required", lowercase(missing.errors))

    probe = "include(ARGS[1]); @assert !isdefined(Main, :DirectorThresholdTuner)"
    @test success(`$(Base.julia_cmd()) --startup-file=no -e $probe $script`)

    scores_path = joinpath(@__DIR__, "..", "..", "..", "benchmarks", "scores", "factcg-anli.json")
    scores = JSON.parsefile(scores_path)["AggreFact-CNN"]
    mktempdir() do directory
        input = joinpath(directory, "scores.jsonl")
        open(input, "w") do stream
            for label in (0, 1)
                selected = Iterators.take((row for row in scores if row[1] == label), 8)
                for row in selected
                    println(stream, JSON.json((score = row[2], label = label)))
                end
            end
        end

        for (filename, seed) in (("seeded.json", ("--seed", "42")), ("unseeded.json", ()))
            output = joinpath(directory, filename)
            run = run_tuner(script, "--input", input, "--output", output, "--bootstrap", "100",
                            "--samples", "40", "--warmup", "20", seed...)
            @test run.status == 0
            @test occursin("wrote $output", run.output)
            result = JSON.parsefile(output)
            @test result["schema"] == "director-ai.threshold-tune.v1"
            @test result["n_examples"] == 16
            @test result["positive_rate"] == 0.5
            @test result["point"]["threshold"] ≈ 0.005
            @test result["point"]["balanced_accuracy"] == 0.625
            @test 0 <= result["bootstrap"]["lo"] <= result["bootstrap"]["hi"] <= 1
            @test sort!(collect(keys(result["bayesian"]))) == ["hi", "lo", "mean", "std"]
            @test result["bayesian"]["mean"] ≈ 0.8092590290144229
            @test result["bayesian"]["std"] ≈ 0.31814980800707654
            @test result["bayesian"]["lo"] ≈ 0.12333050586352708
            @test result["bayesian"]["hi"] ≈ 0.9994250598923783
        end

        write(input, "{\"score\":0.8,\"label\":true}\n\n{\"score\":1e999,\"label\":false}\n")
        output = joinpath(directory, "seeded.json")
        original_result = read(output)
        invalid = run_tuner(script, "--input", input, "--output", output)
        @test invalid.status != 0
        @test occursin("line 3: score must be finite", invalid.errors)
        @test read(output) == original_result
    end
end
