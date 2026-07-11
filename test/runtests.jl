ENV["GKSwstype"] = "100"

using Test
using Lattice2D, ClassicalMonteCarlo
using Random, Statistics, Plots

const FIG_BASE = joinpath(pkgdir(ClassicalMonteCarlo), "docs", "src", "assets", "figures")
const FIG_LAT = joinpath(FIG_BASE, "lattice")
const PATHS = Dict(:geometry => joinpath(FIG_LAT, "geometry"))
mkpath.(values(PATHS))

const dirs = ["core", "algorithms", "model", "utils", "validation"]

# ── CI sharding ──────────────────────────────────────────────────────────────
# The heavy physics-validation legs (test/validation/) are split across CI
# shards keyed by (model, parameter): each validation file `include`s
# test/ci/universe.jl and wraps its @testset legs in `run_case(<id>) do … end`,
# which run only when the case id is selected by env CMC_TEST_CASES (a shard
# runs a subset; unset ⇒ ALL, so a local `Pkg.test()` runs everything).
#
# The cheap mechanical unit dirs (core/algorithms/model/utils) are NOT sharded:
# they always run on every shard.  They are fast, and running them everywhere
# keeps each shard self-contained (an empty-`cases` shard still exercises the
# unit suite and passes).  Only the expensive validation MC sweeps are gated.
@testset "tests" begin
    test_args = copy(ARGS)
    println("Passed arguments ARGS = $(test_args) to tests.")
    @time for dir in dirs
        dirpath = joinpath(@__DIR__, dir)
        println("\nTest $(dirpath)")
        # Find all files named test_*.jl in the directory and include them.
        files = sort(
            filter(f -> startswith(f, "test_") && endswith(f, ".jl"), readdir(dirpath))
        )
        if isempty(files)
            println("  No test files found in $(dirpath).")
            @test true
        else
            for f in files
                filepath = joinpath(dirpath, f)
                println("  Including $(filepath)")
                include(filepath)
            end
        end
    end
end
