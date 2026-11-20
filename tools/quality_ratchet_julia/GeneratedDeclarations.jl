# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — qualified native Julia generator expansions

module GeneratedDeclarations

using ArgParse
using Turing
using ..Declarations

"""Collect actual ArgParse field calls emitted by its pinned table macro."""
function collect_fields!(fields, node)
    node isa Expr || return
    if node.head == :call && first(node.args) == GlobalRef(ArgParse, :add_arg_field!)
        push!(fields, node)
    else
        for child in node.args
            collect_fields!(fields, child)
        end
    end
end

"""Obtain literal aliases and native help metadata from one generated field."""
function inspect_field(call, invocation)
    length(call.args) == 4 || error("Unsupported generated ArgParse field")
    params, names = call.args[2], call.args[4]
    aliases = names isa AbstractString ? [names] :
        Meta.isexpr(names, :vect) ? names.args : nothing
    aliases !== nothing && !isempty(aliases) && all(x -> x isa AbstractString, aliases) ||
        error("ArgParse generated field has no literal names")
    Meta.isexpr(params, :parameters) && length(params.args) == 1 ||
        error("Unsupported generated ArgParse metadata")
    splat = only(params.args)
    Meta.isexpr(splat, :...) && length(splat.args) == 1 ||
        error("Unsupported generated ArgParse option table")
    options = only(splat.args)
    Meta.isexpr(options, :call) && first(options.args) == GlobalRef(ArgParse, :Dict) ||
        error("Unsupported generated ArgParse option dictionary")
    documented = false
    for option in options.args[2:end]
        Meta.isexpr(option, :call) && length(option.args) == 3 &&
            first(option.args) == GlobalRef(ArgParse, :(=>)) &&
            option.args[2] isa QuoteNode ||
            error("Unsupported generated ArgParse option")
        if option.args[2].value == :help
            value = option.args[3]
            value isa AbstractString || error("ArgParse field help must be literal")
            documented = !isempty(strip(value))
        end
    end
    name = string(invocation.owner, ".", join(aliases, ","))
    return Declarations.Declaration(invocation.line, :argument, name, documented)
end

"""Verify and identify the two methods generated for a native Turing model."""
function inspect_model(expansion, invocation)
    Meta.isexpr(expansion, :block) || error("Unsupported native Turing expansion")
    methods = filter(item -> !(item isa LineNumberNode), expansion.args)
    length(methods) == 2 || error("Unexpected Turing generated declaration count")
    evaluator, attached = methods
    marker = Meta.isexpr(attached, :macrocall) ? Declarations.reference_name(attached.args[1]) : nothing
    Meta.isexpr(evaluator, :function) && Meta.isexpr(attached, :macrocall) &&
        marker == "Base.@__doc__" && length(attached.args) == 3 ||
        error("Turing expansion has unsupported evaluator or factory syntax (marker: $marker)")
    constructor = attached.args[3]
    Meta.isexpr(constructor, :function) || error("Turing generated factory is not a method")
    authored_signature = invocation.syntax.args[3].args[1]
    target = Declarations.signature_target(authored_signature)
    Declarations.signature_target(evaluator.args[1]) == target &&
        Declarations.signature_target(constructor.args[1]) == target ||
        error("Turing generated declaration targets differ from the source")
    original = Declarations.signature_name(authored_signature)
    qualified = isempty(invocation.owner) ? original : string(invocation.owner, ".", original)
    return [
        Declarations.Declaration(invocation.line, :model_factory, qualified, invocation.documented),
        Declarations.Declaration(invocation.line, :model_evaluator, qualified, invocation.documented),
    ]
end

"""Expand one qualified dependency macro and return its source-linked records."""
function inspect(invocation::Declarations.Invocation)
    expansion = macroexpand(@__MODULE__, deepcopy(invocation.syntax); recursive=false)
    if invocation.name == Declarations.MODEL
        return inspect_model(expansion, invocation)
    end
    invocation.name == Declarations.ARGUMENT_TABLE ||
        error("Unknown Julia declaration generator")
    fields = Expr[]
    collect_fields!(fields, expansion)
    isempty(fields) && error("ArgParse generated no argument declarations")
    return [inspect_field(field, invocation) for field in fields]
end

"""Read the versions of the actual packages owning the checked macros."""
function versions()
    return Base.pkgversion(ArgParse), Base.pkgversion(Turing),
        Base.pkgversion(parentmodule(getfield(Turing, Symbol("@model"))))
end

end
