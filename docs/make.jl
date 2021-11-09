using FastSpike
using Documenter

DocMeta.setdocmeta!(FastSpike, :DocTestSetup, :(using FastSpike); recursive=true)

makedocs(;
    modules=[FastSpike],
    authors="Mahbod Nouri",
    repo="https://github.com/MahbodNr/FastSpike.jl/blob/{commit}{path}#{line}",
    sitename="FastSpike.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MahbodNr.github.io/FastSpike.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MahbodNr/FastSpike.jl",
    devbranch="main",
)
