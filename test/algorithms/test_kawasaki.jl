using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

@testset "Kawasaki (SpinExchange) — conserved-magnetisation dynamics" begin
    rng = MersenneTwister(19)
    lat = build_lattice(Square, 4, 4)
    model = IsingModel(; J=1.0, h=0.0)
    kaw = LocalUpdate(; rule=Metropolis(), proposal=SpinExchange())

    # (1) DEFINING property: SpinExchange conserves the total magnetisation Σs.
    grids = vcat(fill(1, 8), fill(-1, 8))[randperm(rng, 16)]   # an M=0 config
    @test sum(grids) == 0
    for _ in 1:300
        ClassicalMonteCarlo.update_step!(rng, grids, lat, model, kaw; kbT=1.5)
    end
    @test sum(grids) == 0                                       # M invariant

    # (2) CORRECTNESS: within the fixed M=0 sector the Kawasaki ⟨E⟩ matches the
    # EXACT canonical average restricted to Σs=0 configs (independent oracle:
    # enumeration over the C(16,8) constrained states).
    function exact_E_fixedM(lat, model, kbT, M)
        N = lat.N;
        g = ones(Int, N);
        Z = 0.0;
        Es = 0.0
        for c in 0:(2 ^ N - 1)
            s = 0
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
                s += g[i]
            end
            s == M || continue
            E = total_energy(g, lat, model);
            w = exp(-E / kbT)
            Z += w;
            Es += w * E
        end
        return Es / Z
    end
    kbT = 2.0
    Eex = exact_E_fixedM(lat, model, kbT, 0)
    g = vcat(fill(1, 8), fill(-1, 8))[randperm(rng, 16)]
    Emc = Float64[]
    for step in 1:9000
        ClassicalMonteCarlo.update_step!(rng, g, lat, model, kaw; kbT=kbT)
        step > 2500 && push!(Emc, total_energy(g, lat, model))
    end
    @test isapprox(mean(Emc), Eex; rtol=0.03)
end
