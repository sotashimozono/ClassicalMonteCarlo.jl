using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Replica exchange / parallel tempering. The swap move preserves the product of
# per-replica canonical distributions, so EVERY replica must sample its own
# temperature's canonical ensemble exactly — the ⟨E⟩(T_k) from the tempered run
# must match the EXACT canonical ⟨E⟩(T_k) from 2^N enumeration, simultaneously
# across the whole ladder. (Independent oracle = exact enumeration; the test
# that swaps don't bias the marginals.)
@testset "Replica exchange — per-replica ⟨E⟩(T) vs exact enumeration" begin
    rng = MersenneTwister(101)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)

    exactE(kbT) = begin
        g = ones(Int, N);
        Z = 0.0;
        sE = 0.0
        for c in 0:(2 ^ N - 1)
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            E = total_energy(g, lat, model);
            w = exp(-E / kbT)
            Z += w;
            sE += w * E
        end
        return sE / Z
    end

    Ts = [1.8, 2.1, 2.4, 2.7, 3.0]
    alg = ReplicaExchange(; temperatures=Ts, sweeps=30_000, therm=3_000)
    g0 = rand(rng, (-1, 1), N)
    res = replica_exchange(rng, g0, lat, model, alg)

    for k in 1:length(Ts)
        @test isapprox(res.energy[k], exactE(Ts[k]); rtol=0.02)
    end
    # swaps are actually occurring across the ladder (non-degenerate exchange)
    @test all(res.swap_acceptance .> 0.05)

    # guards
    @test_throws ArgumentError replica_exchange(
        rng, g0, lat, model, ReplicaExchange(; temperatures=[3.0, 2.0])
    )
    @test_throws ArgumentError replica_exchange(
        rng, g0, lat, model, ReplicaExchange(; temperatures=[2.0])
    )
end
