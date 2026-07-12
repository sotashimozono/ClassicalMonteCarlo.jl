using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Swendsen–Wang multi-cluster. Independent oracles: (1) it samples the exact
# canonical ensemble — ⟨E⟩, ⟨m²⟩, Binder match 2^N enumeration; (2) it cures
# critical slowing down — τ_int near Tc is far below Metropolis's; (3) it agrees
# with the independent Wolff cluster algorithm.
@testset "Swendsen–Wang — canonical correctness vs exact + no critical slowing" begin
    rng = MersenneTwister(64)
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

    sample(alg, kbT, nsw, therm) = begin
        g = rand(rng, (-1, 1), N); sE = 0.0; sm2 = 0.0; sm4 = 0.0; n = 0
        for s in 1:nsw
            ClassicalMonteCarlo.update_step!(rng, g, lat, model, alg; kbT=kbT)
            if s > therm
                m = measure_magnetization(g, lat, model)
                sE += total_energy(g, lat, model); sm2 += m^2; sm4 += m^4; n += 1
            end
        end
        (energy=sE / n, m2=sm2 / n, binder=binder_cumulant(sm2 / n, sm4 / n; coeff=c))
    end

    # (1) canonical correctness + (3) agreement with Wolff
    for kbT in (2.27, 3.0)
        ex = exact(kbT)
        sw = sample(SwendsenWang(), kbT, 120_000, 15_000)
        wf = sample(Wolff(), kbT, 120_000, 15_000)
        @test isapprox(sw.energy, ex.energy; rtol=0.02)
        @test isapprox(sw.m2, ex.m2; rtol=0.03)
        @test isapprox(sw.binder, ex.binder; atol=0.02)
        @test isapprox(sw.energy, wf.energy; rtol=0.02)      # SW ≈ Wolff
    end

    # (2) no critical slowing down: τ_int(SW) ≪ τ_int(Metropolis) near Tc on 8×8
    lat8 = build_lattice(Square, 8, 8)
    magseries(alg) = begin
        g = rand(rng, (-1, 1), num_sites(lat8)); m = Float64[]
        for s in 1:40_000
            ClassicalMonteCarlo.update_step!(rng, g, lat8, model, alg; kbT=2.27)
            s > 5_000 && push!(m, measure_magnetization(g, lat8, model))
        end
        m
    end
    @test integrated_autocorrelation_time(magseries(SwendsenWang())).tau <
          integrated_autocorrelation_time(magseries(LocalUpdate())).tau

    @test_throws ArgumentError ClassicalMonteCarlo.update_step!(
        rng, rand(rng, (-1, 1), N), lat, IsingModel(; J=-1.0), SwendsenWang(); kbT=2.0
    )
end
