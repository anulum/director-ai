# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Julia API documentation build

using DirectorThresholdTuner
using Documenter

output = get(ENV, "DIRECTOR_JULIA_DOCS_OUTPUT", joinpath(@__DIR__, "build"))

makedocs(
    modules=[DirectorThresholdTuner],
    sitename="Director-AI Julia Threshold Tuner",
    source=joinpath(@__DIR__, "src"),
    build=output,
    remotes=nothing,
    checkdocs=:exports,
    warnonly=false,
    format=Documenter.HTML(
        prettyurls=true,
        repolink="https://github.com/anulum/director-ai",
    ),
    pages=["API Reference" => "index.md"],
)
