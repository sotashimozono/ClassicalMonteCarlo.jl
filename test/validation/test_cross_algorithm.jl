# ─────────────────────────────────────────────────────────────────────────────
# Layer A — cross-algorithm agreement.
#
# At a fixed (L, T) away from Tc, different update schemes must sample the SAME
# equilibrium distribution, so their ⟨E⟩ and ⟨M²⟩ estimates must agree within
# combined statistical error. This needs no external oracle: disagreement pins
# an algorithm-specific bug (acceptance rule, sweep ordering, ΔE bookkeeping).
#
# NOTE: the package currently ships only LOCAL updaters (wolff.jl and
# swendsen-wang.jl are empty stubs), so the task's Wolff / Swendsen-Wang legs
# are realised by the distinct local schemes actually available:
#   • Metropolis + random-site        • Glauber + random-site
#   • Metropolis + sequential sweep
# These differ in acceptance rule AND site-selection order, so agreement is a
# genuine independent check. When cluster updaters land, add them here.
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "mc_helpers.jl"))
include(joinpath(@__DIR__, "..", "ci", "universe.jl"))

run_case("cross_algo_ising") do
    @testset "cross-algorithm equilibrium agreement" begin
        L = 8
    T = 3.0    # comfortably above Tc ≈ 2.269 ⇒ short autocorrelation, fast mixing

    schemes = [
        (
            "Metropolis/random",
            LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection()),
        ),
        ("Glauber/random", LocalUpdate(; rule=Glauber(), selection=RandomSiteSelection())),
        ("Metropolis/sweep", LocalUpdate(; rule=Metropolis(), selection=SequentialSweep())),
    ]

    ests = Dict(
        name => mc_estimate(L, T, alg; R=8, burn=2000, nsteps=4000, seed0=5000) for
        (name, alg) in schemes
    )

    names = first.(schemes)
    for i in 1:length(names), j in (i + 1):length(names)
        a = ests[names[i]]
        b = ests[names[j]]
        # Combined SEM of the difference of two independent estimates.
        semE = sqrt(a.semE^2 + b.semE^2)
        semM2 = sqrt(a.semM2^2 + b.semM2^2)
        @testset "$(names[i]) vs $(names[j])" begin
            @test abs(a.mE - b.mE) ≤ KSIGMA * semE
            @test abs(a.mM2 - b.mM2) ≤ KSIGMA * semM2
        end
    end
    end
end
