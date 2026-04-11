using Documenter
using Lattice2DMonteCarlo

makedocs(;
    sitename="Lattice2DMonteCarlo.jl",
    modules=[Lattice2DMonteCarlo],
    authors="Sota Shimozono",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sotashimozono.github.io/Lattice2DMonteCarlo.jl",
        edit_link="main",
        assets=String[],
    ),
    warnonly=true,
    pages=[
        "Home" => "index.md",
        "API Reference" => [
            "Core & Interfaces" => "core.md",
            "Models" => "models.md",
            "Algorithms" => "algorithms.md",
            "Visualization" => "visualization.md",
        ],
    ],
)

deploydocs(; repo="github.com/sotashimozono/Lattice2DMonteCarlo.jl.git")
