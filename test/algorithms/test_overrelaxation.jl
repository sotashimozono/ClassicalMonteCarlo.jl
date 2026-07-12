using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

@testset "Overrelaxation (XY) — microcanonical reflection" begin
    rng = MersenneTwister(11)
    lat = build_lattice(Square, 8, 8)
    model = XYModel(; J=1.0)

    # (1) DEFINING property: over-relaxation conserves the total energy exactly.
    # Each reflection preserves the site's bond energy −J|h|cos(θ−φ), so a whole
    # sweep leaves total_energy unchanged (the microcanonical, ΔE=0 signature).
    grids = 2π .* rand(rng, lat.N)
    E0 = total_energy(grids, lat, model)
    or = LocalUpdate(;
        rule=Metropolis(), selection=SequentialSweep(), proposal=Overrelaxation()
    )
    for _ in 1:25
        ClassicalMonteCarlo.update_step!(rng, grids, lat, model, or; kbT=1.0)
    end
    @test isapprox(total_energy(grids, lat, model), E0; atol=1e-6)
    # it actually MOVES the state (not a no-op): magnetization changes
    @test measure_magnetization(grids, lat, model) !=
          measure_magnetization(2π .* rand(MersenneTwister(11), lat.N), lat, model)

    # (2) CANONICAL correctness: interleaving over-relaxation with Metropolis must
    # give the SAME ⟨E⟩ as pure Metropolis (over-relaxation only speeds
    # decorrelation; it preserves the canonical distribution). Cross-algorithm
    # consistency, not self-consistency.
    function mean_energy(mix::Bool, kbT; nsteps=6000, nburn=1500, seed=7)
        r = MersenneTwister(seed)
        g = 2π .* rand(r, lat.N)
        metro = LocalUpdate(; rule=Metropolis(), proposal=UniformShift(; width=1.5))
        ov = LocalUpdate(; rule=Metropolis(), proposal=Overrelaxation())
        Es = Float64[]
        for step in 1:(nburn + nsteps)
            ClassicalMonteCarlo.update_step!(r, g, lat, model, metro; kbT=kbT)
            mix && ClassicalMonteCarlo.update_step!(r, g, lat, model, ov; kbT=kbT)
            step > nburn && push!(Es, total_energy(g, lat, model))
        end
        return mean(Es)
    end
    kbT = 1.0
    E_metro = mean_energy(false, kbT)
    E_mix = mean_energy(true, kbT)
    @test isapprox(E_metro, E_mix; rtol=0.05)
end
