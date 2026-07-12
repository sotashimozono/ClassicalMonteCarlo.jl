using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Integrated autocorrelation time. Independent oracles: (1) an AR(1) process has
# ρ(t)=φ^t and the CLOSED FORM τ_int = 1/2 + φ/(1−φ); (2) an i.i.d. series has
# τ_int = 1/2; plus the physical signature (3) critical slowing down — the MC
# autocorrelation time near Tc greatly exceeds that at high temperature.
@testset "Integrated autocorrelation time — AR(1) closed form + critical slowing" begin
    rng = MersenneTwister(1234)

    # (2) i.i.d. white noise → τ_int = 1/2
    x = randn(rng, 100_000)
    @test isapprox(integrated_autocorrelation_time(x).tau, 0.5; atol=0.1)

    # (1) AR(1): x_t = φ x_{t-1} + √(1−φ²) η_t, stationary var 1, ρ(t)=φ^t
    for φ in (0.5, 0.8, 0.9)
        n = 400_000
        y = zeros(n)
        y[1] = randn(rng)
        s = sqrt(1 - φ^2)
        for i in 2:n
            y[i] = φ * y[i - 1] + s * randn(rng)
        end
        τ_exact = 0.5 + φ / (1 - φ)
        @test isapprox(integrated_autocorrelation_time(y).tau, τ_exact; rtol=0.15)
        # ρ(1) ≈ φ, ρ(2) ≈ φ²
        ρ = autocorrelation(y, 3)
        @test isapprox(ρ[2], φ; atol=0.03)
        @test isapprox(ρ[3], φ^2; atol=0.03)
    end

    # (3) critical slowing down: Metropolis magnetisation autocorrelation on 8×8 is
    # much longer near Tc than at high T
    lat = build_lattice(Square, 8, 8)
    model = IsingModel(; J=1.0, h=0.0)
    magseries(kbT) = begin
        g = rand(rng, (-1, 1), num_sites(lat))
        m = Float64[]
        for s in 1:35_000
            ClassicalMonteCarlo.update_step!(rng, g, lat, model, LocalUpdate(); kbT=kbT)
            s > 5_000 && push!(m, measure_magnetization(g, lat, model))
        end
        m
    end
    τ_Tc = integrated_autocorrelation_time(magseries(2.27)).tau
    τ_hi = integrated_autocorrelation_time(magseries(5.0)).tau
    @test τ_Tc > 2 * τ_hi                          # critical slowing down is clearly visible

    @test_throws ArgumentError autocorrelation([1.0, 2.0], 5)
end
