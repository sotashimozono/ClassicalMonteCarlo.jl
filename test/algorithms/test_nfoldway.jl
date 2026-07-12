using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# N-fold-way / BKL rejection-free kinetic Ising. The Glauber rates obey detailed
# balance w.r.t. the Gibbs weight, so the residence-time-weighted averages ⟨E⟩,
# ⟨M²⟩ must equal the EXACT canonical averages — computed independently by
# enumeration of all 2^N spin configurations. A rejection-free continuous-time
# chain reproducing the equilibrium of the rejection-based ensemble is a genuine
# cross-mechanism check (with a field term to exercise h ≠ 0).
@testset "N-fold-way (BKL) — residence-weighted ⟨E⟩,⟨M²⟩ vs exact enumeration" begin
    rng = MersenneTwister(7)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)

    function exact(lat, model, kbT)
        g = ones(Int, N); Z = 0.0; sE = 0.0; sM2 = 0.0
        for c in 0:(2^N - 1)
            m = 0
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
                m += g[i]
            end
            E = total_energy(g, lat, model); w = exp(-E / kbT)
            Z += w; sE += w * E; sM2 += w * m^2
        end
        return (energy=sE / Z, mag2=sM2 / Z)
    end

    for (model, kbT) in (
        (IsingModel(; J=1.0, h=0.0), 3.0),
        (IsingModel(; J=1.0, h=0.0), 2.5),
        (IsingModel(; J=1.0, h=0.4), 2.5),      # h ≠ 0
    )
        ex = exact(lat, model, kbT)
        g = rand(rng, (-1, 1), N)
        r = nfold_way(rng, g, lat, model, NFoldWay(; steps=3_000_000, therm=50_000); kbT=kbT)
        @test isapprox(r.energy, ex.energy; rtol=0.02)
        @test isapprox(r.mag2, ex.mag2; rtol=0.05)
    end
end
