# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native Julia declaration inventory

module Declarations

"""Represent one authored declaration and its native documentation attachment."""
struct Declaration
    line::Int
    kind::Symbol
    qualified_name::String
    documented::Bool
end

"""Retain an applicable native macro invocation and its lexical owner."""
struct Invocation
    line::Int
    owner::String
    name::Symbol
    syntax::Expr
    documented::Bool
end

"""Describe a literal include and its lexical evaluation owner."""
struct IncludeTarget
    owner::String
    parts::Vector{String}
    unconditional::Bool
end

"""Accumulate declarations and generated surfaces for one source file."""
struct SourceInventory
    declarations::Vector{Declaration}
    invocations::Vector{Invocation}
    imports::Set{Tuple{String,String}}
    public_names::Set{String}
    includes::Vector{IncludeTarget}
end

const DOC = GlobalRef(Core, Symbol("@doc"))
const MODEL = Symbol("@model")
const ARGUMENT_TABLE = Symbol("@add_arg_table!")
const GENERATOR_PACKAGES = Dict(MODEL => "Turing", ARGUMENT_TABLE => "ArgParse")
const SUPPORTED_MACROS = (
    Symbol("@__DIR__"), Symbol("@__FILE__"), Symbol("@test"), Symbol("@testset"),
    Symbol("@test_throws"),
    # JuliaSyntax emits these Core references for commands and wide integer literals.
    GlobalRef(Core, Symbol("@cmd")),
    GlobalRef(Core, Symbol("@int128_str")),
    GlobalRef(Core, Symbol("@uint128_str")),
    GlobalRef(Core, Symbol("@big_str")),
)

"""Return the syntactic name of an unqualified or dotted Julia reference."""
function reference_name(value)
    value isa Symbol && return string(value)
    value isa Module && return string(value)
    value isa GlobalRef && return string(value.mod, ".", value.name)
    value isa QuoteNode && return reference_name(value.value)
    if Meta.isexpr(value, :.) && length(value.args) == 2
        left = reference_name(value.args[1])
        right = reference_name(value.args[2])
        if left !== nothing && right !== nothing
            return string(left, ".", right)
        end
    end
    return nothing
end

"""Identify a method target, keeping receiver types but dropping local receiver names."""
function signature_target(value)
    if Meta.isexpr(value, (:where, :(::)))
        return signature_target(first(value.args))
    elseif Meta.isexpr(value, :call)
        callee = first(value.args)
        if Meta.isexpr(callee, :(::))
            return Expr(:(::), last(callee.args))
        end
        return callee
    end
    return value
end

"""Find the defined name of a Julia signature or parametric declaration."""
function signature_name(value)
    target = signature_target(value)
    while Meta.isexpr(target, (:where, :(::), :curly, :<:))
        target = first(target.args)
    end
    return reference_name(target)
end

"""Distinguish short method definitions from ordinary assignments."""
function is_method_signature(value)
    Meta.isexpr(value, :call) && return true
    Meta.isexpr(value, (:where, :(::))) || return false
    return is_method_signature(value.args[1])
end

"""Add a declaration after qualifying its lexical address and source position."""
function declare!(inventory, line, kind, owner, name, documented)
    name === nothing && error("Unsupported Julia declaration name at line $line")
    line > 0 || error("Julia declaration has no source line")
    qualified = isempty(owner) ? name : string(owner, ".", name)
    push!(inventory.declarations, Declaration(line, kind, qualified, documented))
    return qualified
end

"""Collect field declarations and attached strings in a native struct body."""
function scan_fields!(inventory, body, line, owner, enclosing_owner, mutable_struct::Bool)
    document = false
    for member in body.args
        if member isa LineNumberNode
            line = Int(member.line)
        elseif member isa AbstractString
            document = !isempty(strip(member))
        elseif member isa Symbol || Meta.isexpr(member, :(::))
            declare!(inventory, line, :field, owner, signature_name(member), document)
            document = false
        elseif Meta.isexpr(member, :const, 1)
            field = only(member.args)
            if Meta.isexpr(field, :global, 1)
                scan!(inventory, member, line, enclosing_owner)
            else
                name = Meta.isexpr(field, :(::), 2) ? first(field.args) : field
                mutable_struct && name isa Symbol ||
                    error("Unsupported Julia constant field at line $line")
                # Julia's native documentation metadata does not register these field docs.
                declare!(inventory, line, :field, owner, name, false)
            end
            document = false
        else
            scan!(inventory, member, line, owner, document)
            document = false
        end
    end
end

"""Prove that an applicable generator resolves through its literal package import."""
function require_generator_binding(inventory, owner, name, line)
    owner_parts = split(owner, '.')
    package = GENERATOR_PACKAGES[name]
    scopes = (join(owner_parts[1:count], '.') for count in length(owner_parts):-1:0)
    any(scope -> (scope, package) in inventory.imports, scopes) ||
        error("Julia generator $name has no qualified package import at line $line")
end

"""Record explicit module imports and refuse ambiguous aliases of known generators."""
function scan_imports!(inventory, args, line, owner, unconditional)
    for entry in args
        if Meta.isexpr(entry, :.) && length(entry.args) == 1 && entry.args[1] isa Symbol
            unconditional && push!(inventory.imports, (owner, string(entry.args[1])))
        elseif any(name -> occursin(string(name), repr(entry)), keys(GENERATOR_PACKAGES))
            error("Julia generator imported through an unsupported alias at line $line")
        end
    end
end

"""Record a literal include that can be reconciled with the complete Git inventory."""
function scan_include!(inventory, args, line, owner, unconditional)
    length(args) == 2 || error("Unsupported Julia include target at line $line")
    target = args[2]
    if target isa AbstractString
        push!(inventory.includes, IncludeTarget(owner, [String(target)], unconditional))
    elseif Meta.isexpr(target, :call) && length(target.args) >= 3 &&
            target.args[1] == :joinpath && Meta.isexpr(target.args[2], :macrocall) &&
            target.args[2].args[1] == Symbol("@__DIR__") &&
            all(argument -> argument isa AbstractString, target.args[3:end])
        push!(inventory.includes, IncludeTarget(
            owner, String[argument for argument in target.args[3:end]], unconditional,
        ))
    else
        error("Unsupported dynamic Julia include at line $line")
    end
end

"""Collect documented syntax and retain the native location of generator calls."""
function scan!(inventory, node, line::Int, owner::String, documented::Bool=false;
               unconditional::Bool=false)
    node isa Expr || return
    head, args = node.head, node.args
    head in (:error, :incomplete) && error("Native Julia parse failure: $node")
    head == :quote && return
    if head == :macrocall
        macro_name = args[1]
        at = args[2] isa LineNumberNode ? Int(args[2].line) : line
        if macro_name == DOC
            length(args) == 4 || error("Unsupported Julia documentation attachment")
            doc = args[3]
            doc isa AbstractString || error("Julia docstring is not a literal")
            scan!(inventory, args[4], at, owner, !isempty(strip(doc)))
        elseif macro_name in (MODEL, ARGUMENT_TABLE)
            require_generator_binding(inventory, owner, macro_name, at)
            push!(inventory.invocations, Invocation(at, owner, macro_name, node, documented))
            if macro_name == MODEL
                length(args) in (3, 4) || error("Unsupported Turing model declaration")
                length(args) == 3 || args[4] isa Bool ||
                    error("Turing model warning flag must be a literal Boolean")
                model = args[3]
                Meta.isexpr(model, :function) ||
                    (Meta.isexpr(model, :(=)) && is_method_signature(first(model.args))) ||
                    error("Unsupported Turing model declaration")
                scan!(inventory, model, at, owner, documented)
            end
        elseif macro_name in SUPPORTED_MACROS
            scan_children!(inventory, args[3:end], at, owner)
        else
            error("Unsupported Julia macro $(string(macro_name)) at line $at")
        end
    elseif head == :module
        child = declare!(inventory, line, :module, owner, reference_name(args[2]), documented)
        scan!(inventory, args[3], line, child; unconditional=true)
    elseif head == :struct
        child = declare!(inventory, line, :struct, owner, signature_name(args[2]), documented)
        scan_fields!(inventory, args[3], line, child, owner, args[1])
    elseif head in (:abstract, :primitive)
        declare!(inventory, line, head, owner, signature_name(args[1]), documented)
    elseif head in (:function, :macro) || (head == :(=) && is_method_signature(args[1]))
        name = signature_name(args[1])
        if head == :macro && name in ("model", "add_arg_table!")
            error("Julia source shadows a qualified generated declaration")
        end
        child = declare!(inventory, line, head == :macro ? :macro : :method, owner, name, documented)
        scan_children!(inventory, args[2:end], line, child)
    elseif head == :const
        length(args) == 1 || error("Unsupported Julia constant declaration")
        assignment = only(args)
        # Both explicit modifier orders lower to the same parser wrapper.
        assignment = Meta.isexpr(assignment, :global, 1) ? only(assignment.args) : assignment
        Meta.isexpr(assignment, :(=), 2) || error("Unsupported Julia constant declaration")
        declare!(inventory, line, :constant, owner, signature_name(assignment.args[1]), documented)
        scan_children!(inventory, assignment.args[2:end], line, owner)
    elseif head in (:using, :import)
        scan_imports!(inventory, args, line, owner, unconditional)
    elseif head in (:export, :public)
        for value in args
            name = reference_name(value)
            name === nothing && error("Unsupported Julia public binding at line $line")
            push!(inventory.public_names, isempty(owner) ? name : string(owner, ".", name))
        end
    elseif head in (:toplevel, :block)
        scan_children!(inventory, args, line, owner, unconditional)
    elseif head == :call
        callee = reference_name(first(args))
        if callee !== nothing && last(split(callee, '.')) in ("eval", "include_string")
            error("Unsupported dynamic Julia declaration at line $line")
        elseif callee == "include"
            scan_include!(inventory, args, line, owner, unconditional)
        elseif callee !== nothing && endswith(callee, ".include")
            error("Unsupported qualified Julia include at line $line")
        end
        scan_children!(inventory, args, line, owner)
    else
        scan_children!(inventory, args, line, owner)
    end
    return nothing
end

"""Walk native sibling locations, retaining the current source line."""
function scan_children!(inventory, children, line, owner, unconditional=false)
    for child in children
        if child isa LineNumberNode
            line = Int(child.line)
        else
            scan!(inventory, child, line, owner; unconditional=unconditional)
        end
    end
end

"""Inspect Julia authored declarations from complete source without executing it."""
function inspect(source::AbstractString, filename::AbstractString; validate_public::Bool=true)
    inventory = SourceInventory(
        Declaration[], Invocation[], Set{Tuple{String,String}}(), Set{String}(),
        IncludeTarget[],
    )
    scan!(inventory, Meta.parseall(source; filename=filename), 1, ""; unconditional=true)
    if validate_public
        defined = Set(item.qualified_name for item in inventory.declarations)
        issubset(inventory.public_names, defined) ||
            error("Julia source has public bindings without qualified declarations")
    end
    return inventory
end

end
