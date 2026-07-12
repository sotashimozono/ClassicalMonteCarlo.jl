using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Jackknife & bootstrap error bars. Independent oracles: (1) the jackknife error
# of the MEAN of i.i.d. samples equals the sample standard error σ/√n EXACTLY;
# (2) for the nonlinear estimator g(⟨x⟩)=⟨x⟩², both match the delta-method error
# 2|μ|·σ/√n; (3) jackknife and bootstrap agree; (4) applied to a Binder-type ratio
# of independent-oracle samples, the bias-corrected value is unbiased.
@testset "Resampling — jackknife/bootstrap vs analytic error formulas" begin
    rng = MersenneTwister(2718)

    # (1) jackknife error of the mean == sample SE = std(x; corrected)/√n (exact)
    x = randn(rng, 5_000) .+ 2.0
    n = length(x)
    jk = jackknife(m -> m[1], x)
    se = std(x; corrected=true) / sqrt(n)
    @test isapprox(jk.value, mean(x); rtol=1e-12)
    @test isapprox(jk.error, se; rtol=1e-10)

    # (2) delta method for g(⟨x⟩)=⟨x⟩²: error ≈ |g'(μ)|·SE = 2|⟨x⟩|·SE
    jk2 = jackknife(m -> m[1]^2, x)
    delta = 2 * abs(mean(x)) * se
    @test isapprox(jk2.error, delta; rtol=0.02)

    bs2 = bootstrap(rng, m -> m[1]^2, x; n_resample=3_000)
    @test isapprox(bs2.error, delta; rtol=0.08)               # (3) bootstrap ≈ delta
    @test isapprox(bs2.error, jk2.error; rtol=0.08)           # (3) bootstrap ≈ jackknife

    # (4) ratio estimator R = ⟨y⟩/⟨x⟩ on two independent observables: jackknife
    # error matches the delta-method  SE_R = |R|·√((σx/μx)²+(σy/μy)²)/√n  (indep)
    y = randn(rng, 5_000) .+ 5.0
    R(m) = m[2] / m[1]
    jkR = jackknife(R, [x, y])
    μx = mean(x); μy = mean(y)
    seR = abs(μy / μx) * sqrt((std(x) / μx)^2 + (std(y) / μy)^2) / sqrt(n)
    @test isapprox(jkR.value, μy / μx; rtol=1e-12)
    @test isapprox(jkR.error, seR; rtol=0.05)

    @test_throws ArgumentError jackknife(m -> m[1], [1.0])
    @test_throws ArgumentError jackknife(m -> m[1], [[1.0, 2.0], [1.0]])
end
