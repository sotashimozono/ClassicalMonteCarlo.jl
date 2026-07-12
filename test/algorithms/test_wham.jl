using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# WHAM multi-histogram. Combining runs at several temperatures yields a single
# self-consistent density of states g(E). Two independent-oracle checks against
# 2^N enumeration: (1) the recovered ln g(E) matches the EXACT combinatorial DOS
# ln Ω(E) (config multiplicity) up to an additive constant, on the well-sampled
# energies; (2) canonical ⟨E⟩(T) reconstructed from g(E) matches exact ⟨E⟩(T)
# across a WIDE range — wider than any single input run's reliable window.
@testset "WHAM — recovered g(E) vs exact DOS, ⟨E⟩(T) over wide range" begin
    rng = MersenneTwister(24)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)

    Ω = Dict{Int,Int}()
    let g = ones(Int, N)
        for c in 0:(2 ^ N - 1)
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            k = round(Int, total_energy(g, lat, model))
            Ω[k] = get(Ω, k, 0) + 1
        end
    end
    exactE(kbT) = begin
        Z = 0.0;
        sE = 0.0
        for (k, ω) in Ω
            w = ω * exp(-k / kbT);
            Z += w;
            sE += w * k
        end
        return sE / Z
    end

    kbTs = [1.6, 2.1, 2.6, 3.2]
    series = map(kbTs) do T
        g = rand(rng, (-1, 1), N)
        sample_energies(
            rng,
            g,
            lat,
            model,
            LocalUpdate();
            kbT=T,
            sweeps=300_000,
            therm=20_000,
            interval=2,
        )
    end

    res = wham(series, kbTs, WHAM())
    @test res.iters < 10_000                       # converged before the cap

    # (1) recovered ln g(E) vs exact ln Ω(E) up to an additive constant, on the
    # WELL-SAMPLED bins only (reference = the most-sampled bin, so gauge cancels)
    iref = argmax(res.counts)
    kref = round(Int, res.energies[iref])
    ntested = 0
    for i in eachindex(res.energies)
        res.counts[i] >= 500 || continue           # WHAM DOS is meaningful only where sampled
        k = round(Int, res.energies[i])
        @test isapprox(res.logg[i] - res.logg[iref], log(Ω[k]) - log(Ω[kref]); atol=0.15)
        ntested += 1
    end
    @test ntested >= 5                             # the window actually spans several energies

    # (2) reconstructed ⟨E⟩(T) across a wide range vs exact
    for kbT in (1.8, 2.0, 2.27, 2.5, 3.0)
        @test isapprox(wham_mean(res, kbT), exactE(kbT); rtol=0.02)
    end

    @test_throws ArgumentError wham([Float64[]], [2.0])
    @test_throws ArgumentError wham([[1.0], [2.0]], [2.0])
end
