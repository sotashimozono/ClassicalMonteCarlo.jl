using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Blocking / binning analysis. Independent oracles: (1) i.i.d. data — the blocking
# SE stays flat at σ/√n (no correlation to expose); (2) AR(1) — the plateau error
# matches the CLOSED-FORM asymptotic variance of the sample mean,
# Var(x̄) = (σ²/n)(1+φ)/(1−φ); (3) it is consistent with the integrated
# autocorrelation time: blocking_error² · n / σ² ≈ 2 τ_int.
@testset "Blocking analysis — plateau error vs AR(1) closed form & τ_int" begin
    rng = MersenneTwister(31459)

    # (1) i.i.d. → flat: blocking error ≈ naive σ/√n
    x = randn(rng, 1 << 16) .+ 1.0
    n = length(x)
    naive = std(x; corrected=true) / sqrt(n)
    @test isapprox(blocking_error(x), naive; rtol=0.15)

    # (2) AR(1): plateau error² → (σ²/n)(1+φ)/(1−φ)
    for φ in (0.6, 0.8)
        m = 1 << 18
        y = zeros(m); y[1] = randn(rng); s = sqrt(1 - φ^2)
        for i in 2:m
            y[i] = φ * y[i - 1] + s * randn(rng)
        end
        σ2 = var(y; corrected=true)
        err_exact = sqrt(σ2 / m * (1 + φ) / (1 - φ))
        be = blocking_error(y)
        @test isapprox(be, err_exact; rtol=0.15)
        # the naive SE badly underestimates for correlated data
        @test be > 1.5 * (sqrt(σ2) / sqrt(m))

        # (3) cross-check with τ_int:  be² · m / σ² ≈ 2 τ_int
        τ = integrated_autocorrelation_time(y).tau
        @test isapprox(be^2 * m / σ2, 2τ; rtol=0.2)
    end

    # blocking levels: block size doubles, block count halves
    lv = blocking(x)
    @test lv[1].block_size == 1 && lv[1].n_blocks == n
    @test lv[2].block_size == 2 && lv[2].n_blocks == n ÷ 2
    @test isapprox(lv[1].mean, mean(x); rtol=1e-12)

    @test_throws ArgumentError blocking([1.0])
end
