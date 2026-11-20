# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Julia declaration reader tests

using Test
import TOML

include("../Declarations.jl")

const TUNER = "tools/julia_tuner"
const LIBRARY = joinpath(TUNER, "src", "DirectorThresholdTuner.jl")
const LAUNCHER = joinpath(TUNER, "bin", "tune_threshold.jl")
const HISTORICAL = "391228da74540de1f5edc948df83765c1619e42a"

@testset "Julia branch producer pins" begin
    record = TOML.parsefile(joinpath(@__DIR__, "..", "branch_dispositions.toml"))
    manifests = Dict(
        "analyzer" => TOML.parsefile(joinpath(@__DIR__, "..", "Manifest.toml")),
        "tuner" => TOML.parsefile(joinpath(TUNER, "Manifest.toml")),
    )
    @test record["schema"] == "director-ai.julia-branch-dispositions.v1"
    @test record["julia_version"] == string(VERSION)
    for (name, manifest) in manifests
        @testset "$name Julia version" begin
            @test manifest["julia_version"] == record["julia_version"]
        end
    end

    pins = record["package_pins"]
    @test Set((pin["manifest"], pin["package"]) for pin in pins) == Set((
        ("analyzer", "ArgParse"),
        ("analyzer", "DynamicPPL"),
        ("analyzer", "Turing"),
        ("tuner", "ArgParse"),
        ("tuner", "JSON"),
    ))
    for pin in pins
        scope, name = pin["manifest"], pin["package"]
        @testset "$scope $name producer pin" begin
            entry = only(manifests[scope]["deps"][name])
            @test entry["version"] == pin["version"]
            @test entry["git-tree-sha1"] == pin["git-tree-sha1"]
        end
    end

    residuals = record["unobserved"]
    @test length(residuals) == record["outcomes"] - record["observed"]
    @test length(Set((row["owner"], row["site"], row["outcome"]) for row in residuals)) ==
        length(residuals)
    @test Set(row["disposition"] for row in residuals) == Set(("A", "B", "OPEN"))
    @test Set(row["producer"] for row in residuals if row["disposition"] == "B") ==
        Set(("ArgParse", "DynamicPPL", "Git", "JSON", "JuliaSyntax"))

    bindings = Set((row["owner"], row["owner_blob"]) for row in residuals)
    for row in residuals
        for guard in get(row, "dominating_guards", ())
            push!(bindings, (guard["owner"], guard["owner_blob"]))
        end
    end
    for (owner, expected_blob) in sort!(collect(bindings))
        @testset "$owner Git blob" begin
            @test readchomp(`git hash-object --no-filters -- $owner`) == expected_blob
        end
    end
end

@testset "Julia tuner source declarations" begin
    source = read(LIBRARY, String)
    inventory = Declarations.inspect(source, LIBRARY; validate_public=false)
    fields = filter(declaration -> declaration.kind == :field, inventory.declarations)
    @test length(fields) == 14
    @test any(fields) do field
        field.qualified_name == "DirectorThresholdTuner.ScoredExample.score"
    end
    @test length(inventory.public_names) == 9
    @test any(inventory.declarations) do declaration
        declaration.qualified_name == "DirectorThresholdTuner.balanced_accuracy" &&
            declaration.documented
    end
    @test isempty(inventory.invocations)
    posterior = only(inventory.includes)
    @test posterior.owner == "DirectorThresholdTuner"
    @test posterior.parts == ["threshold_posterior.jl"]
end

@testset "Julia generator bindings" begin
    source = """
    module ObservationModels
    using Turing
    @model function observed_reading(value)
        value ~ Normal()
    end
    end
    """
    file = "observation-model.jl"
    inventory = Declarations.inspect(source, file)
    @test only(inventory.invocations).name == Symbol("@model")
    @test only(inventory.invocations).owner == "ObservationModels"
    @test only(inventory.invocations).line == 3
    @test_throws ErrorException Declarations.inspect(
        replace(source, "using Turing" => ""), file,
    )
    @test_throws ErrorException Declarations.inspect(
        replace(source, "using Turing" => "if false\nusing Turing\nend"), file,
    )
    @test_throws ErrorException Declarations.inspect(
        replace(source, "@model function" => "@eval function"), file,
    )
    @test_throws ErrorException Declarations.inspect(
        replace(source, "using Turing" => "using Turing\nexport unbound_api"), file,
    )
    @test_throws ErrorException Declarations.inspect(
        replace(source, "using Turing" => "using Unrelated: @model\nusing Turing"), file,
    )
    @test_throws ErrorException Declarations.inspect(
        replace(source, "using Turing" => "using Turing\nmacro model(node)\nnode\nend"), file,
    )
end

@testset "Julia tuner launcher provenance" begin
    source = read(LAUNCHER, String)
    inventory = Declarations.inspect(source, LAUNCHER)
    @test only(inventory.includes).parts == ["..", "src", "DirectorThresholdTuner.jl"]
    @test only(inventory.invocations).owner == "_parse"
    @test only(inventory.invocations).name == Symbol("@add_arg_table!")
    @test all(declaration -> declaration.documented, inventory.declarations)
    @test_throws ErrorException Declarations.inspect(
        replace(source, "include(joinpath" => "Base.include(joinpath"), LAUNCHER,
    )
    @test_throws ErrorException Declarations.inspect(
        replace(source, "include(joinpath" => "include(select_dynamic_path"), LAUNCHER,
    )
    old = read(`git show $(HISTORICAL):$LAUNCHER`, String)
    @test_throws ErrorException Declarations.inspect(old, LAUNCHER)
end

@testset "Qualified Julia types, fields and short methods" begin
    source = """
    module Example
    export Reading, Code, Sample, LIMIT, typed, generic
    abstract type Reading end
    primitive type Code <: Reading 8 end
    struct Sample <: Reading
        "Measured quantity."
        quantity::Int
        raw
        "Construct a sample."
        Sample(quantity) = new(quantity, nothing)
    end
    "Maximum quantity."
    const LIMIT::Int = 42
    typed(x)::Int = x
    generic(x::T) where {T} = x
    include("extra.jl")
    end
    """
    inventory = Declarations.inspect(source, "example.jl")
    @test inventory.public_names == Set([
        "Example.Reading", "Example.Code", "Example.Sample", "Example.LIMIT",
        "Example.typed", "Example.generic",
    ])
    @test only(inventory.includes).parts == ["extra.jl"]
    @test [(d.qualified_name, d.kind, d.documented) for d in inventory.declarations] == [
        ("Example", :module, false),
        ("Example.Reading", :abstract, false),
        ("Example.Code", :primitive, false),
        ("Example.Sample", :struct, false),
        ("Example.Sample.quantity", :field, true),
        ("Example.Sample.raw", :field, false),
        ("Example.Sample.Sample", :method, true),
        ("Example.LIMIT", :constant, true),
        ("Example.typed", :method, false),
        ("Example.generic", :method, false),
    ]
end

@testset "Unsupported Julia declaration identity" begin
    for source in ("const (left, right) = (1, 2)", "global const (left, right) = (1, 2)")
        @test_throws r"Unsupported Julia declaration name at line 1" Declarations.inspect(
            source, "destructured.jl",
        )
    end
    for source in ("eval(:(value = 1))", "Core.eval(@__MODULE__, :(value = 1))",
                   "include_string(Main, \"value = 1\")")
        @test_throws r"Unsupported dynamic Julia declaration at line 1" Declarations.inspect(
            source, "dynamic.jl",
        )
    end
end

@testset "Julia mutable structs retain constant field declarations" begin
    source = """
    module StableMeasurements
    export Measurement, SCALE
    "Measurement with an immutable identifier and a mutable value."
    mutable struct Measurement
        "Permanent measurement identifier."
        const identifier::Int
        "Source label."
        const label
        const schema_version::Int
        unchecked::Int
        "Current measurement value."
        value::Int
        global const SCALE = 2
    end
    end
    """
    inventory = Declarations.inspect(source, "StableMeasurements.jl")
    @test [(item.qualified_name, item.line, item.documented)
           for item in inventory.declarations if item.kind == :field] == [
        ("StableMeasurements.Measurement.identifier", 5, false),
        ("StableMeasurements.Measurement.label", 7, false),
        ("StableMeasurements.Measurement.schema_version", 9, false),
        ("StableMeasurements.Measurement.unchecked", 10, false),
        ("StableMeasurements.Measurement.value", 11, true),
    ]
    @test only(item.qualified_name for item in inventory.declarations if item.kind == :constant) ==
        "StableMeasurements.SCALE"
    for source in (
        "struct Sample; const value::Int; end",
        "mutable struct Sample; const value = 1; end",
        "mutable struct Sample; const ::Int; end",
    )
        @test_throws r"Unsupported Julia constant field" Declarations.inspect(
            source, "invalid-constant-field.jl",
        )
    end
end

@testset "Documented Julia constants with explicit global scope" begin
    source = """
    module GlobalConstants
    export LIMIT, MASK
    "Document an explicit global typed constant."
    global const LIMIT::Int = 42
    "Document a constant with the alternate modifier order."
    const global MASK = 0xFF
    end
    """
    inventory = Declarations.inspect(source, "GlobalConstants.jl")
    @test inventory.public_names == Set(["GlobalConstants.LIMIT", "GlobalConstants.MASK"])
    @test [(d.qualified_name, d.line, d.documented)
           for d in inventory.declarations if d.kind == :constant] == [
        ("GlobalConstants.LIMIT", 3, true),
        ("GlobalConstants.MASK", 5, true),
    ]
end

@testset "Julia lexical scope and quoted syntax" begin
    source = """
    public reading
    reading(value) = value
    macro marker(value)
        return value
    end
    quote
        eval(:(injected = 1))
        struct Hidden end
    end
    resolve().handle()
    """
    inventory = Declarations.inspect(source, "syntax.jl")
    @test inventory.public_names == Set(["reading"])
    @test [(d.qualified_name, d.kind) for d in inventory.declarations] == [
        ("reading", :method), ("marker", :macro),
    ]
end

@testset "Unsupported Julia include forms" begin
    @test_throws r"Unsupported Julia include target at line 1" Declarations.inspect(
        "include(identity, \"example.jl\")", "mapped-include.jl",
    )
    for source in (
        "include(filename)",
        "include(joinpath())",
        "include(joinpath(root, \"example.jl\"))",
        "include(joinpath(@__FILE__, \"example.jl\"))",
        "include(joinpath(@__DIR__, filename))",
    )
        @test_throws r"Unsupported dynamic Julia include at line 1" Declarations.inspect(
            source, "joined-include.jl",
        )
    end
end

@testset "Unsupported native model form" begin
    for source in (
        "using Turing\n@model",
        "using Turing\n@model label",
        "using Turing\n@model label = 1",
    )
        @test_throws r"Unsupported Turing model declaration" Declarations.inspect(
            source, "invalid-model.jl",
        )
    end
    @test_throws r"Turing model warning flag must be a literal Boolean" Declarations.inspect(
        "using Turing\n@model (function reading(); return 1; end) identity(true)",
        "invalid-model-warning.jl",
    )
end

@testset "Julia documentation must be literal" begin
    source = raw"""
    "Reading $label"
    reading() = 3
    """
    @test_throws r"Julia docstring is not a literal" Declarations.inspect(source, "doc.jl")
end

@testset "Explicit Julia macros do not inherit compiler literal privileges" begin
    source = """
    # An explicitly authored qualified macro is outside the literal syntax.
    reading() = Core.@big_str("12345678901234567890123456789012345678901234567890")
    """
    @test_throws r"Unsupported Julia macro .*@big_str.* at line 2" Declarations.inspect(
        source, "qualified-macro.jl",
    )
end

@testset "Native wide integer literals retain declarations" begin
    source = """
    "A signed wide integer."
    const SIGNED = 1000000000000000000000000000000
    "An unsigned wide integer."
    const MASK = 0x10000000000000001
    "An arbitrary-precision integer."
    const LIMIT = 100000000000000000000000000000000000000000000000000
    """
    inventory = Declarations.inspect(source, "wide-literals.jl")
    @test [(item.qualified_name, item.kind, item.line, item.documented)
           for item in inventory.declarations] == [
        ("SIGNED", :constant, 1, true),
        ("MASK", :constant, 3, true),
        ("LIMIT", :constant, 5, true),
    ]
    @test isempty(inventory.invocations)
end

@testset "Public declaration names must be static" begin
    for source in (raw"export $binding", raw"public $(compute_name())")
        @test_throws r"Unsupported Julia public binding at line 1" Declarations.inspect(
            source, "public-binding.jl",
        )
    end
end
