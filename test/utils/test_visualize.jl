using Test
using Plots
using ColorSchemes
using LinearAlgebra
using Lattice2DMonteCarlo
using Lattice2D

ENV["GKSwstype"] = "100"

@testset "Visualization Methods" begin
    lat = build_lattice(Square, 4, 4)
    N = lat.N

    @testset "Helper Functions" begin
        ms = find_marker_size(lat)
        @test ms isa Float64
        @test ms > 0.0

        p, ms2 = plot_setup(lat)
        @test p isa Plots.Plot
        @test ms2 == ms
    end

    @testset "Ising Model Visualization" begin
        model = IsingModel()
        grids = rand([-1, 1], N)

        colors = Lattice2DMonteCarlo.get_state_colors(model, grids)
        @test length(colors) == N
        @test colors[1] in [:red, :blue]

        p = visualize_snapshot(grids, lat, model)
        @test p isa Plots.Plot
    end

    @testset "Potts Model Visualization" begin
        q = 3
        model = PottsModel(q=q)
        grids = rand(1:q, N)

        colors = Lattice2DMonteCarlo.get_state_colors(model, grids)
        @test length(colors) == N
        @test colors[1] isa ColorTypes.Colorant

        p = visualize_snapshot(grids, lat, model)
        @test p isa Plots.Plot
    end

    @testset "XY Model Visualization" begin
        model = XYModel()
        grids = rand(N) .* 2π

        p = visualize_snapshot(grids, lat, model)
        @test p isa Plots.Plot

        @test length(p.series_list) > 0
    end
end
