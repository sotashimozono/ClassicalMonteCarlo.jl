using Documenter
using ClassicalMonteCarlo

makedocs(;
    sitename="ClassicalMonteCarlo.jl",
    modules=[ClassicalMonteCarlo],
    authors="Sota Shimozono",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sotashimozono.github.io/ClassicalMonteCarlo.jl",
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

deploydocs(; repo="github.com/sotashimozono/ClassicalMonteCarlo.jl.git")
