using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

@testset "Heat-bath (Gibbs) sampler" begin
    # (1) DEFINING property: heatbath_sample! draws from the exact local Boltzmann
    # conditional P(s) ∝ exp(−E_local(s)/kbT) given the fixed neighbours.
    rng = MersenneTwister(3)
    lat = build_lattice(Square, 3, 3)
    model = PottsModel(; q=3, J=1.0)
    grids = fill(1, lat.N)
    ns = collect(neighbors(lat, 1))
    grids[ns[1]] = 1
    grids[ns[2]] = 1
    grids[ns[3]] = 2
    grids[ns[4]] = 3
    kbT = 0.8
    Eloc = [local_hamiltonian(grids, lat, model, 1; val=s) for s in 1:3]
    Pexact = exp.(-Eloc ./ kbT) ./ sum(exp.(-Eloc ./ kbT))
    counts = zeros(Int, 3)
    Nsamp = 200_000
    for _ in 1:Nsamp
        counts[heatbath_sample!(rng, 1, grids, lat, model; kbT=kbT)] += 1
    end
    @test maximum(abs.(counts ./ Nsamp .- Pexact)) < 0.01

    # (2) CANONICAL correctness: heat-bath ⟨E⟩(T) matches the EXACT canonical
    # average from full enumeration of all 3^9 configs (independent oracle).
    function exact_energy(lat, model, kbT)
        N = lat.N
        q = model.q
        g = fill(1, N)
        Z = 0.0
        Es = 0.0
        for c in 0:(q ^ N - 1)
            x = c
            @inbounds for i in 1:N
                g[i] = (x % q) + 1
                x ÷= q
            end
            E = total_energy(g, lat, model)
            w = exp(-E / kbT)
            Z += w
            Es += w * E
        end
        return Es / Z
    end
    kbT2 = 1.0
    Eex = exact_energy(lat, model, kbT2)
    g = rand(rng, 1:3, lat.N)
    Emc = Float64[]
    for step in 1:6000
        ClassicalMonteCarlo.update_step!(rng, g, lat, model, HeatBath(); kbT=kbT2)
        step > 1500 && push!(Emc, total_energy(g, lat, model))
    end
    @test isapprox(mean(Emc), Eex; rtol=0.03)

    # (3) Ising heat-bath ≡ Glauber dynamics: same canonical ⟨E⟩.
    lat2 = build_lattice(Square, 6, 6)
    im = IsingModel(; J=1.0, h=0.0)
    function meanE(alg; kbT, seed)
        r = MersenneTwister(seed)
        g2 = rand(r, (-1, 1), lat2.N)
        E = Float64[]
        for s in 1:6000
            ClassicalMonteCarlo.update_step!(r, g2, lat2, im, alg; kbT=kbT)
            s > 1500 && push!(E, total_energy(g2, lat2, im))
        end
        return mean(E)
    end
    @test isapprox(
        meanE(HeatBath(); kbT=2.0, seed=5),
        meanE(LocalUpdate(; rule=Glauber(), proposal=SpinFlip()); kbT=2.0, seed=5);
        rtol=0.05,
    )
end
