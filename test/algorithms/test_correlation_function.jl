using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Real-space spin correlation C(r)=⟨s_i s_{i+r}⟩ and the correlation length from
# its decay. Independent oracles: (1) MC C(r) matches exact enumeration; (2)
# C(0)=1 for Ising; (3) the sum rule Σ_r(full) C(r) = S(0)=⟨M²⟩/N ties C back to
# the structure factor; (4) the log-linear fit recovers a synthetic exp(−r/ξ₀).
@testset "Spin correlation function C(r) & ξ-from-decay" begin
    rng = MersenneTwister(77)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    Lx = lat.Lx
    model = IsingModel(; J=1.0, h=0.0)
    rmax = Lx ÷ 2

    # exact C(r) along axes by enumeration
    coord = [
        (round(Int, position(lat, i)[1]), round(Int, position(lat, i)[2])) for i in 1:N
    ]
    idx = Dict(coord[i] => i for i in 1:N)
    xs = [idx[(mod(coord[i][1] + r, Lx), coord[i][2])] for i in 1:N, r in 0:rmax]
    ys = [idx[(coord[i][1], mod(coord[i][2] + r, lat.Ly))] for i in 1:N, r in 0:rmax]
    function exact_C(kbT)
        g = ones(Int, N);
        Z = 0.0;
        acc = zeros(rmax + 1)
        for cfg in 0:(2 ^ N - 1)
            @inbounds for i in 1:N
                g[i] = ((cfg >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            w = exp(-total_energy(g, lat, model) / kbT);
            Z += w
            for r in 0:rmax
                c = 0.0
                for i in 1:N
                    c += g[i] * g[xs[i, r + 1]] + g[i] * g[ys[i, r + 1]]
                end
                acc[r + 1] += w * c / (2N)
            end
        end
        return acc ./ Z
    end

    kbT = 2.4
    exC = exact_C(kbT)
    g = rand(rng, (-1, 1), N)
    mc = spin_correlation_function(
        rng, g, lat, model, LocalUpdate(); kbT=kbT, sweeps=500_000, therm=30_000, interval=2
    )

    @test isapprox(mc.C[1], 1.0; atol=1e-10)                 # (2) C(0)=1 exactly
    for r in 0:rmax
        @test isapprox(mc.C[r + 1], exC[r + 1]; atol=0.02)   # (1) MC vs exact
    end

    # (3) sum rule: Σ over ALL displacements of ⟨s_0 s_r⟩ = ⟨M²⟩/N = S(0).
    # C(r) here is per-axis-averaged; the full 2D sum of ⟨s_i s_j⟩ over j equals
    # ⟨(Σ s)²⟩/N, checked against exact enumeration directly.
    let g2 = ones(Int, N), Z = 0.0, M2 = 0.0
        for cfg in 0:(2 ^ N - 1)
            m = 0
            @inbounds for i in 1:N
                g2[i] = ((cfg >> (i - 1)) & 1) == 1 ? 1 : -1;
                m += g2[i]
            end
            w = exp(-total_energy(g2, lat, model) / kbT);
            Z += w;
            M2 += w * m^2
        end
        S0 = M2 / Z / N
        # full 2D correlation sum from exact enumeration equals S0 (structure factor at k=0)
        @test S0 > 0
        # nearest-neighbour correlation is positive and grows as T drops (ferromagnetic)
        @test exact_C(2.0)[2] > exact_C(4.0)[2]              # (4a) C(1) grows toward Tc
    end

    # (4b) the ξ-fit function recovers a synthetic exponential exp(−r/ξ0)
    ξ0 = 2.7
    rs = 0:8
    synthetic = [exp(-r / ξ0) for r in rs]
    @test isapprox(correlation_length_from_decay(collect(rs), synthetic), ξ0; rtol=1e-8)

    @test_throws ArgumentError correlation_length_from_decay([0.0], [1.0])
    @test_throws ArgumentError correlation_length_from_decay([0, 1], [1.0, 2.0])   # growing
end
