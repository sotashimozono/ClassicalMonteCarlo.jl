using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Wolff single-cluster algorithm. Independent oracles: (1) it samples the exact
# canonical ensemble — ⟨E⟩, ⟨m²⟩ and the Binder cumulant match 2^N enumeration;
# (2) it cures critical slowing down — the magnetisation autocorrelation time
# near Tc is far smaller than Metropolis's.
@testset "Wolff cluster — canonical correctness vs exact + no critical slowing" begin
    rng = MersenneTwister(46)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)
    c = ClassicalMonteCarlo.get_binder_coeff(model)

    function exact(kbT)
        g = ones(Int, N); Z = 0.0; sE = 0.0; sm2 = 0.0; sm4 = 0.0
        for cfg in 0:(2^N - 1)
            @inbounds for i in 1:N
                g[i] = ((cfg >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            m = measure_magnetization(g, lat, model); E = total_energy(g, lat, model)
            w = exp(-E / kbT); Z += w
            sE += w * E; sm2 += w * m^2; sm4 += w * m^4
        end
        return (energy=sE / Z, m2=sm2 / Z, binder=binder_cumulant(sm2 / Z, sm4 / Z; coeff=c))
    end

    # (1) canonical correctness — Wolff samples ⟨E⟩, ⟨m²⟩, Binder correctly
    for kbT in (2.27, 3.0)
        ex = exact(kbT)
        g = rand(rng, (-1, 1), N)
        sE = 0.0; sm2 = 0.0; sm4 = 0.0; n = 0
        for s in 1:200_000
            ClassicalMonteCarlo.update_step!(rng, g, lat, model, Wolff(); kbT=kbT)
            if s > 20_000
                m = measure_magnetization(g, lat, model)
                sE += total_energy(g, lat, model); sm2 += m^2; sm4 += m^4; n += 1
            end
        end
        @test isapprox(sE / n, ex.energy; rtol=0.02)
        @test isapprox(sm2 / n, ex.m2; rtol=0.03)
        @test isapprox(binder_cumulant(sm2 / n, sm4 / n; coeff=c), ex.binder; atol=0.02)
    end

    # (2) no critical slowing down: τ_int(Wolff) ≪ τ_int(Metropolis) near Tc on 8×8
    lat8 = build_lattice(Square, 8, 8)
    kbT = 2.27
    magseries(alg) = begin
        g = rand(rng, (-1, 1), num_sites(lat8)); m = Float64[]
        for s in 1:40_000
            ClassicalMonteCarlo.update_step!(rng, g, lat8, model, alg; kbT=kbT)
            s > 5_000 && push!(m, measure_magnetization(g, lat8, model))
        end
        m
    end
    τ_wolff = integrated_autocorrelation_time(magseries(Wolff())).tau
    τ_metro = integrated_autocorrelation_time(magseries(LocalUpdate())).tau
    @test τ_wolff < τ_metro

    @test_throws ArgumentError ClassicalMonteCarlo.update_step!(
        rng, rand(rng, (-1, 1), N), lat, IsingModel(; J=-1.0), Wolff(); kbT=2.0
    )
end
