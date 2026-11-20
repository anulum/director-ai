# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Julia generator tests

using Test
using Turing

include("../Declarations.jl")
include("../GeneratedDeclarations.jl")

@testset "ArgParse generates the tuner arguments and their help metadata" begin
    file = "tools/julia_tuner/bin/tune_threshold.jl"
    source = read(file, String)
    invocation = only(Declarations.inspect(source, file).invocations)
    fields = GeneratedDeclarations.inspect(invocation)
    @test length(fields) == 6
    documented_inputs = count(fields) do field
        field.qualified_name == "_parse.--input,-i" && field.documented
    end
    @test documented_inputs == 1
    @test all(field -> field.kind == :argument, fields)
    @test all(field -> field.documented, fields)

    changed = replace(source, "help = \"RNG seed (omit for non-deterministic)\"" => "")
    offending = only(Declarations.inspect(changed, file).invocations)
    result = GeneratedDeclarations.inspect(offending)
    @test [field.qualified_name for field in result if !field.documented] == ["_parse.--seed"]
end

@testset "Native ArgParse rejects incomplete argument declarations" begin
    for (table, diagnostic) in (
        ("", r"ArgParse generated no argument declarations"),
        ("\"--label\"\nhelp = description", r"ArgParse field help must be literal"),
        ("\"--label\"\n:help => \"description\"", r"Unsupported generated ArgParse option"),
    )
        source = """
        using ArgParse
        function parse_args(settings)
            @add_arg_table! settings begin
                $table
            end
        end
        """
        invocation = only(Declarations.inspect(source, "arguments.jl").invocations)
        @test_throws diagnostic GeneratedDeclarations.inspect(invocation)
    end
end

@testset "Top-level Turing declarations retain their unqualified name" begin
    source = """
    using Turing
    @model function independent(x)
        x ~ Normal()
    end
    """
    invocation = only(Declarations.inspect(source, "model.jl").invocations)
    declarations = GeneratedDeclarations.inspect(invocation)
    @test [(item.qualified_name, item.kind) for item in declarations] == [
        ("independent", :model_factory), ("independent", :model_evaluator),
    ]
end

@testset "Turing generates methods for native model declaration forms" begin
    for syntax in (
        "@model reading(x) = (x ~ Normal())",
        "@model (function reading(x); x ~ Normal(); end) true",
        "@model (function reading(x); x ~ Normal(); end) false",
    )
        source = """
        module ModelFixture
        using Turing
        "Model an observed reading."
        $syntax
        end
        """
        fixture = Module(gensym(:TuringFixture))
        Base.include_string(fixture, source, "model-forms.jl")
        @test Base.invokelatest(fixture) do loaded
            model = loaded.ModelFixture.reading(0.0)
            model isa Turing.DynamicPPL.Model
        end
        inventory = Declarations.inspect(source, "model-forms.jl")
        declaration = only(filter(record -> record.kind == :method, inventory.declarations))
        @test (declaration.qualified_name, declaration.documented) == ("ModelFixture.reading", true)
        generated = GeneratedDeclarations.inspect(only(inventory.invocations))
        @test [(record.qualified_name, record.kind, record.documented) for record in generated] == [
            ("ModelFixture.reading", :model_factory, true),
            ("ModelFixture.reading", :model_evaluator, true),
        ]
    end
end

@testset "Turing callable models retain the authored receiver type" begin
    for (name, receiver, parameters) in (
        ("AnonymousModel", "::AnonymousModel", ""),
        ("NamedModel", "model::NamedModel", ""),
        ("ParametricModel", "::ParametricModel{T}", " where T"),
    )
        definition = isempty(parameters) ? name : "$name{T}"
        source = """
        using Turing
        struct $definition end
        @model function ($receiver)(x)$parameters
            x ~ Normal()
        end
        """
        inventory = Declarations.inspect(source, "callable-model.jl")
        @test only(filter(record -> record.kind == :method, inventory.declarations)).qualified_name == name
        generated = GeneratedDeclarations.inspect(only(inventory.invocations))
        @test [(record.qualified_name, record.kind) for record in generated] == [
            (name, :model_factory), (name, :model_evaluator),
        ]
    end
    source = """
    using Turing
    @model function (::choose_receiver_type())(x)
        x ~ Normal()
    end
    """
    @test_throws r"Unsupported Julia declaration name" Declarations.inspect(
        source, "dynamic-model-receiver.jl",
    )
end

@testset "Turing generates both observation model methods" begin
    file = "observation-model.jl"
    source = """
    module ObservationModels
    using Turing
    @model function observed_reading(value)
        value ~ Normal()
    end
    end
    """
    fixture = Module(gensym(:TuringFixture))
    Base.include_string(fixture, source, file)
    @test Base.invokelatest(fixture) do loaded
        loaded.ObservationModels.observed_reading(0.0) isa Turing.DynamicPPL.Model
    end
    invocation = only(Declarations.inspect(source, file).invocations)
    records = GeneratedDeclarations.inspect(invocation)
    @test Set(record.kind for record in records) == Set((:model_factory, :model_evaluator))
    @test all(record -> !record.documented, records)
    @test all(record -> record.qualified_name == "ObservationModels.observed_reading", records)
    @test all(record -> record.line == invocation.line, records)

    documented = replace(source, "@model function" => "\"The native threshold model.\"\n@model function")
    changed = only(Declarations.inspect(documented, file).invocations)
    @test all(record -> record.documented, GeneratedDeclarations.inspect(changed))
end
