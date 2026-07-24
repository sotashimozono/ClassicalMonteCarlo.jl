using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# von Mises mean resultant length  R(κ) = ∫cos(t)e^{κcos t}dt / ∫e^{κcos t}dt,
# an INDEPENDENT closed-form (deterministic quadrature — no Monte Carlo) that
# the rejection-free XY heat-bath sampler must reproduce.
function vonmises_R(κ)
    n = 40_000;
    num = 0.0;
    den = 0.0
    for k in 0:(n - 1)
        t = 2π * (k + 0.5) / n
        w = exp(κ * cos(t))
        num += cos(t) * w;
        den += w
    end
    return num / den
end

@testset "XY heat-bath — von Mises local conditional (Best–Fisher)" begin
    rng = MersenneTwister(31)
    lat = build_lattice(Square, 3, 3)
    model = XYModel(; J=1.0)

    # (1) DIRECT conditional: freeze every neighbour at α ⇒ local field points at
    # φ=α with magnitude h=deg(site); draw the site repeatedly and compare the
    # sample circular mean & resultant length to the von Mises closed form.
    α = 0.7
    grids = fill(α, num_sites(lat))
    site = 5                                   # a bulk (degree-4) site on 3×3 PBC
    deg = length(Lattice2D.neighbors(lat, site))
    for kbT in (0.5, 1.5)
        κ = model.J * deg / kbT
        cs = Float64[];
        ss = Float64[]
        for _ in 1:60_000
            θ = ClassicalMonteCarlo.heatbath_sample!(rng, site, grids, lat, model; kbT=kbT)
            push!(cs, cos(θ - α));
            push!(ss, sin(θ - α))
        end
        @test isapprox(mean(cs), vonmises_R(κ); atol=0.01)   # resultant length
        @test isapprox(mean(ss), 0.0; atol=0.01)             # centred at φ=α
    end

    # (2) CROSS-ALGORITHM: heat-bath and Metropolis(UniformShift) must agree on
    # the canonical ⟨E⟩ per site at fixed kbT (independent samplers, same Gibbs
    # measure).
    kbT = 1.0
    hb = HeatBath()
    mp = LocalUpdate(; rule=Metropolis(), proposal=UniformShift(; width=2π))
    function run_E(alg, seed)
        r = MersenneTwister(seed);
        g = 2π .* rand(r, num_sites(lat));
        Es = Float64[]
        for step in 1:15_000
            ClassicalMonteCarlo.update_step!(r, g, lat, model, alg; kbT=kbT)
            step > 4_000 && push!(Es, total_energy(g, lat, model))
        end
        return mean(Es)
    end
    @test isapprox(run_E(hb, 1), run_E(mp, 2); rtol=0.03)
end
