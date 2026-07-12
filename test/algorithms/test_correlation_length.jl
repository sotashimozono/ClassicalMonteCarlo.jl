using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Second-moment correlation length via the structure factor S(k). Independent
# oracles from 2^N enumeration: (1) S(0) and S(k_min) computed by MC match the
# EXACT canonical structure factors; (2) S(0) equals ⟨M²⟩/N; (3) ξ from MC
# matches ξ built from the exact structure factors (deterministic function);
# (4) ξ grows as T decreases toward Tc (physical monotonicity).
@testset "Second-moment correlation length — S(k) & ξ vs exact enumeration" begin
    rng = MersenneTwister(303)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)
    L = lat.Lx
    kx = 2π / L

    # precompute phase factors for k=0, (kx,0), (0,kx)
    cx = [cos(kx * position(lat, j)[1]) for j in 1:N]
    sx = [sin(kx * position(lat, j)[1]) for j in 1:N]
    cy = [cos(kx * position(lat, j)[2]) for j in 1:N]
    sy = [sin(kx * position(lat, j)[2]) for j in 1:N]

    function exact_S(kbT)
        g = ones(Int, N); Z = 0.0; S0 = 0.0; Sk = 0.0; M2 = 0.0
        for c in 0:(2^N - 1)
            m = 0
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1; m += g[i]
            end
            E = total_energy(g, lat, model); w = exp(-E / kbT); Z += w
            S0 += w * m^2 / N
            axr = 0.0; axi = 0.0; ayr = 0.0; ayi = 0.0
            @inbounds for j in 1:N
                axr += g[j] * cx[j]; axi += g[j] * sx[j]
                ayr += g[j] * cy[j]; ayi += g[j] * sy[j]
            end
            Sk += w * ((axr^2 + axi^2) + (ayr^2 + ayi^2)) / (2N)
            M2 += w * m^2
        end
        S0 /= Z; Sk /= Z; M2 /= Z
        return (S0=S0, Sk=Sk, M2=M2)
    end

    kmin_mag = 2π / L
    for kbT in (2.4, 4.0)
        ex = exact_S(kbT)
        @test isapprox(ex.S0, ex.M2 / N; rtol=1e-12)          # (2) S(0) = ⟨M²⟩/N identity
        g = rand(rng, (-1, 1), N)
        mc = measure_correlation_length(
            rng, g, lat, model, LocalUpdate(); kbT=kbT, sweeps=500_000, therm=30_000, interval=2
        )
        @test isapprox(mc.S0, ex.S0; rtol=0.03)               # (1) MC S(0) vs exact
        @test isapprox(mc.Skmin, ex.Sk; rtol=0.04)            # (1) MC S(k_min) vs exact
        ξ_exact = second_moment_correlation_length(ex.S0, ex.Sk, kmin_mag)
        @test isapprox(mc.xi, ξ_exact; rtol=0.06)             # (3) ξ_MC vs ξ_exact
    end

    # (4) ξ grows as T decreases toward Tc
    e_lo = exact_S(2.4)
    e_hi = exact_S(4.0)
    ξ_lo = second_moment_correlation_length(e_lo.S0, e_lo.Sk, kmin_mag)
    ξ_hi = second_moment_correlation_length(e_hi.S0, e_hi.Sk, kmin_mag)
    @test ξ_lo > ξ_hi

    @test_throws ArgumentError second_moment_correlation_length(1.0, 0.0, kmin_mag)
end
