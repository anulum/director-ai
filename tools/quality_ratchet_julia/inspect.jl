# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete native Julia quality evidence

using SHA
using TOML
include("Declarations.jl")

const ENGINE_PATHS = Set([
    "tools/quality_ratchet_julia/Declarations.jl",
    "tools/quality_ratchet_julia/GeneratedDeclarations.jl",
    "tools/quality_ratchet_julia/inspect.jl",
    "tools/quality_ratchet_julia/test/generated.jl",
    "tools/quality_ratchet_julia/test/runtests.jl",
])

"""Determine the complete Git-visible Julia package scope."""
function julia_sources(root)
    realpath(chomp(read(`git -C $root rev-parse --show-toplevel`, String))) == root ||
        error("Julia inspection needs the actual Git root")
    paths = read(`git -C $root ls-files --cached --others --exclude-standard -z`, String)
    files = sort!(unique([
        String(path) for path in split(paths, '\0'; keepempty=false)
        if endswith(path, ".jl") && !(path in ENGINE_PATHS)
    ]))
    !isempty(files) && all(path -> startswith(path, "tools/julia_tuner/"), files) ||
        error("Julia source is outside the qualified package scope")
    return files
end

"""Hash the actual runtime project and manifest selected by Pkg."""
function package_identity()
    project = Base.active_project()
    project === nothing && error("Julia runtime has no active project")
    manifest = Base.project_file_manifest_path(project)
    manifest === nothing && error("Julia runtime has no active manifest")
    return bytes2hex(sha256(read(project))), bytes2hex(sha256(read(manifest)))
end

"""Retain the tuner dependency inputs separately from the analyzer environment."""
function source_configuration(root)
    tuner = joinpath(root, "tools", "julia_tuner")
    project = read(joinpath(tuner, "Project.toml"), String)
    manifest = bytes2hex(sha256(read(joinpath(tuner, "Manifest.toml"))))
    return (; project, manifest)
end

"""Join a declaring file's inherited module with one local lexical name."""
qualified_name(owner, name) = isempty(owner) ? name : isempty(name) ? owner : string(owner, ".", name)

"""Resolve one file's lexical module from the complete set of literal includers."""
function file_owner!(owners, visiting, parents, file)
    haskey(owners, file) && return owners[file]
    file in visiting && error("Cyclic Julia include at $file")
    push!(visiting, file)
    scopes = Set(
        qualified_name(file_owner!(owners, visiting, parents, parent), owner)
        for (parent, owner) in parents[file]
    )
    length(scopes) <= 1 || error("Ambiguous Julia include owner for $file")
    owners[file] = isempty(scopes) ? "" : only(scopes)
    delete!(visiting, file)
    return owners[file]
end

"""Read files, resolve literal include owners and validate exports in the same source graph."""
function inspect_sources(root, files)
    !isempty(files) && length(files) == length(Set(files)) ||
        error("Julia inventory is empty or contains duplicate paths")
    parents = Dict(file => Tuple{String,String}[] for file in files)
    neighbours = Dict(file => String[] for file in files)
    records = map(sort(files)) do file
        path = joinpath(root, file)
        !isabspath(file) && normpath(file) == file &&
            relpath(realpath(path), root) == file &&
            !startswith(file, "..") ||
            error("Unqualified Julia source path")
        source = read(path, String)
        inventory = Declarations.inspect(source, file; validate_public=false)
        for target in inventory.includes
            included = normpath(joinpath(dirname(file), target.parts...))
            haskey(parents, included) ||
                error("Julia include target is absent from the source inventory")
            push!(parents[included], (file, target.owner))
            if target.unconditional
                push!(neighbours[file], included)
                push!(neighbours[included], file)
            end
        end
        return file, bytes2hex(sha256(source)), inventory
    end

    owners = Dict{String,String}()
    visiting = Set{String}()
    for (file, _, _) in records
        file_owner!(owners, visiting, parents, file)
    end

    defined = Dict(file => Set(
        qualified_name(owners[file], item.qualified_name) for item in inventory.declarations
    ) for (file, _, inventory) in records)
    for (file, _, inventory) in records
        isempty(inventory.public_names) && continue
        pending = [file]
        seen = Set{String}()
        bindings = Set{String}()
        while !isempty(pending)
            current = pop!(pending)
            current in seen && continue
            push!(seen, current)
            union!(bindings, defined[current])
            append!(pending, neighbours[current])
        end
        all(name -> qualified_name(owners[file], name) in bindings, inventory.public_names) ||
            error("Julia source has public bindings without qualified declarations at $file")
    end
    return [(file, digest, inventory, owners[file]) for (file, digest, inventory) in records]
end

"""Write a source digest and native declarations in the Julia evidence protocol."""
function write_source!(output, file, digest, records)
    filename = bytes2hex(codeunits(file))
    println(output, "source\t", filename, "\t", digest, "\t", length(records))
    for declaration in records
        println(
            output, "declaration\t", filename, "\t", declaration.line, "\t",
            declaration.kind, "\t", bytes2hex(codeunits(declaration.qualified_name)), "\t",
            declaration.documented,
        )
    end
end

length(ARGS) == 1 || error("Expected the actual Julia source repository root")
root = realpath(only(ARGS))
files = julia_sources(root)
records = inspect_sources(root, files)
project, manifest = package_identity()
tuner_configuration = source_configuration(root)
include("GeneratedDeclarations.jl")
tuner_dependencies = TOML.parse(tuner_configuration.project)["deps"]
output = IOBuffer()
println(output, "julia-quality-v2\t", VERSION, "\t", join(GeneratedDeclarations.versions(), "\t"))
println(output, "binary\t", bytes2hex(sha256(read(first(Base.julia_cmd().exec)))))
println(output, "project\t", project)
println(output, "manifest\t", manifest)
for (file, digest, inventory, owner) in records
    declarations = copy(inventory.declarations)
    for invocation in inventory.invocations
        package = Declarations.GENERATOR_PACKAGES[invocation.name]
        haskey(tuner_dependencies, package) ||
            error("Julia generator $package is not a direct dependency of the source project")
        append!(declarations, GeneratedDeclarations.inspect(invocation))
    end
    qualified = [Declarations.Declaration(
        item.line, item.kind, qualified_name(owner, item.qualified_name), item.documented,
    ) for item in declarations]
    write_source!(output, file, digest, qualified)
end
julia_sources(root) == files && package_identity() == (project, manifest) &&
    all(row -> bytes2hex(sha256(read(joinpath(root, row[1])))) == row[2], records) ||
    error("Julia source inventory or native environment changed during inspection")
source_configuration(root) == tuner_configuration ||
    error("Julia source project or manifest changed during inspection")
println(output, "complete\t", length(records))
write(stdout, take!(output))
