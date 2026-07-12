using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Improved (cluster) susceptibility estimators. Independent oracles: (1) both the
# Wolff ⟨|C|⟩/kbT and the Swendsen–Wang ⟨Σ|C|²⟩/(kbT·N) match the exact
# susceptibility χ = ⟨M²⟩/(kbT·N) from 2^N enumeration; (2) the improved SW
# estimator has SMALLER error bars than the bare spin M² estimator on the same
# run (Rao–Blackwell variance reduction), shown via blocking.
@testset "Improved cluster susceptibility — exact match + variance reduction" begin
    rng = MersenneTwister(1618)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)

    exact_chi(kbT) = begin
        g = ones(Int, N);
        Z = 0.0;
        M2 = 0.0
        for cfg in 0:(2 ^ N - 1)
            m = 0
            @inbounds for i in 1:N
                g[i] = ((cfg >> (i - 1)) & 1) == 1 ? 1 : -1;
                m += g[i]
            end
            w = exp(-total_energy(g, lat, model) / kbT);
            Z += w;
            M2 += w * m^2
        end
        return (M2 / Z) / (kbT * N)
    end

    # (1) both improved estimators match the exact susceptibility
    for kbT in (2.27, 3.0)
        χex = exact_chi(kbT)
        gw = rand(rng, (-1, 1), N)
        χw =
            wolff_susceptibility(rng, gw, lat, model; kbT=kbT, sweeps=150_000, therm=15_000).chi
        gs = rand(rng, (-1, 1), N)
        χs = swendsen_wang_susceptibility(
            rng, gs, lat, model; kbT=kbT, sweeps=150_000, therm=15_000
        ).chi
        @test isapprox(χw, χex; rtol=0.04)
        @test isapprox(χs, χex; rtol=0.04)
    end

    # (2) variance reduction: on a shared SW run collect both the cluster estimator
    # Σ|C|² and the bare spin M², both estimating N²·⟨m²⟩; the cluster estimator's
    # blocking error is smaller.
    kbT = 2.27
    g = rand(rng, (-1, 1), N)
    p = 1.0 - exp(-2.0 / kbT)
    label = zeros(Int, N);
    stack = Int[]
    clust = Float64[];
    spin = Float64[]
    for s in 1:60_000
        fill!(label, 0);
        nlab = 0;
        sizes = Int[]
        for start in 1:N
            label[start] == 0 || continue
            nlab += 1;
            label[start] = nlab;
            push!(stack, start);
            sz = 1
            while !isempty(stack)
                i = pop!(stack)
                for j in Lattice2D.neighbors(lat, i)
                    if label[j] == 0 && g[j] == g[i] && rand(rng) < p
                        label[j] = nlab;
                        push!(stack, j);
                        sz += 1
                    end
                end
            end
            push!(sizes, sz)
        end
        if s > 5_000
            push!(clust, sum(abs2, sizes))          # Σ|C|²  = E[M² | clusters]
            push!(spin, sum(g)^2)                    # bare M²
        end
        flip = [rand(rng) < 0.5 for _ in 1:nlab]
        @inbounds for i in 1:N
            flip[label[i]] && (g[i] = -g[i])
        end
    end
    @test isapprox(mean(clust), mean(spin); rtol=0.03)     # same expectation
    @test blocking_error(clust) < blocking_error(spin)     # smaller variance

    @test_throws ArgumentError wolff_susceptibility(
        rng, rand(rng, (-1, 1), N), lat, IsingModel(; J=-1.0); kbT=2.0, sweeps=10
    )
    @test_throws ArgumentError swendsen_wang_susceptibility(
        rng, rand(rng, (-1, 1), N), lat, IsingModel(; J=-1.0); kbT=2.0, sweeps=10
    )
end
