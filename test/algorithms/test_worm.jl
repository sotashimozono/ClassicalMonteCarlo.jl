using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Worm algorithm (Prokof'ev–Svistunov): the high-temperature bond/graph
# representation of the zero-field Ising model. Its per-spin susceptibility
# estimator S = N_total/N_closed samples S = Σ_r⟨σ_0σ_r⟩ = ⟨M²⟩/N — validated
# against the SAME quantity computed in the completely different SPIN
# representation by exact enumeration of all 2^N configurations. Two disjoint
# representations agreeing is a genuine independent-oracle check.
@testset "Worm (Prokof'ev–Svistunov) — Ising S=Σ⟨σ₀σᵣ⟩ vs exact ⟨M²⟩/N" begin
    rng = MersenneTwister(2027)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)

    function exact_S(lat, model, kbT)
        g = ones(Int, N);
        Z = 0.0;
        M2 = 0.0
        for c in 0:(2 ^ N - 1)
            m = 0
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
                m += g[i]
            end
            E = total_energy(g, lat, model);
            w = exp(-E / kbT)
            Z += w;
            M2 += w * m^2
        end
        return (M2 / Z) / N
    end

    model = IsingModel(; J=1.0, h=0.0)
    for kbT in (3.5, 2.5)                       # paramagnetic → near-critical
        Sx = exact_S(lat, model, kbT)
        alg = WormIsing(; steps=3_000_000, therm=50_000)
        Sw = worm_susceptibility(rng, lat, model, alg; kbT=kbT)
        @test isapprox(Sw, Sx; rtol=0.04)
    end

    # the graph expansion is zero-field only
    @test_throws ArgumentError worm_susceptibility(
        rng, lat, IsingModel(; J=1.0, h=0.3), WormIsing(); kbT=2.5
    )
end
