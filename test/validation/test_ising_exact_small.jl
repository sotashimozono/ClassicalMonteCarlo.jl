# ─────────────────────────────────────────────────────────────────────────────
# Layer A — engine self-test: MC on a small lattice vs an INDEPENDENT
# brute-force Boltzmann enumeration, plus an exact-vs-exact QAtlas cross-check.
#
# Independent expectations checked here:
#   1. Brute-force partition function  ==  QAtlas Kaufman/transfer-matrix Z
#      (exact vs exact, rtol 1e-8) — corroborates QAtlas's declared exact value.
#   2. MC ⟨E⟩, ⟨|M|⟩, ⟨M²⟩  ==  brute-force exact, within k·SEM (k = 4),
#      where SEM comes from independent chains (no hand-tuned tolerance).
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "mc_helpers.jl"))
include(joinpath(@__DIR__, "..", "ci", "universe.jl"))

# QAtlas is only needed by the exact-Z cross-check leg; a shard that does not
# select it never pays the QAtlas precompile/load cost.
if case_selected("exact_ising_bruteZ")
    using QAtlas
end

@testset "Ising exact-small (engine self-test + QAtlas Z cross-check)" begin

    # `build_lattice(Square, …)` is fully periodic (constructor default
    # `boundary = PeriodicAxis()`), which is exactly QAtlas's torus convention
    # for the finite-lattice partition function. Verified below at rtol 1e-8.

    run_case("exact_ising_bruteZ") do
        @testset "brute-force Z == QAtlas exact Z" begin
        # Lx,Ly ≥ 3 count each physical bond once; the 2×2 case wraps a
        # length-2 ring (each such bond enters twice), and Lattice2D's
        # `neighbors` returns the wrapped site twice as well — the same
        # double-counting QAtlas documents — so the two Z's still coincide.
        for (Lx, Ly, β) in ((2, 2, 0.5), (3, 3, 0.3), (4, 4, 0.4), (4, 4, 0.6))
            Z_brute = exact_ising(Lx, Ly, β).Z
            Z_qatlas = QAtlas.fetch(
                IsingSquare(), PartitionFunction(); Lx=Lx, Ly=Ly, β=β, J=1.0
            )
            # Both are exact; only floating-point summation error separates them.
            # The 4×4 sum over 2^16 states carries ~1e-13 relative round-off,
            # comfortably inside rtol = 1e-8. A larger gap would flag a genuine
            # convention or QAtlas bug.
            @test Z_brute ≈ Z_qatlas rtol = 1e-8
        end
        end
    end

    run_case("exact_ising_mc") do
        @testset "MC == brute-force exact within $(KSIGMA)·SEM" begin
        L = 4                       # 4×4 = 16 sites: 2^16 exact states
        # Two acceptance rules exercised (the package ships no cluster updater,
        # so Wolff/Swendsen-Wang from the task spec are substituted by the two
        # available local rules Metropolis and Glauber — see PR notes).
        rules = [("Metropolis", Metropolis()), ("Glauber", Glauber())]
        for T in (2.5, 4.0)           # moderate T ⇒ good single-flip mixing on 4×4
            β = 1 / T
            ex = exact_ising(L, L, β)
            for (rname, rule) in rules
                alg = LocalUpdate(; rule=rule, selection=RandomSiteSelection())
                est = mc_estimate(L, T, alg; R=10, burn=1500, nsteps=4000, seed0=4000)
                @testset "$rname T=$T" begin
                    @test abs(est.mE - ex.E) ≤ KSIGMA * est.semE
                    @test abs(est.mAbsM - ex.absM) ≤ KSIGMA * est.semAbsM
                    @test abs(est.mM2 - ex.M2) ≤ KSIGMA * est.semM2
                end
            end
        end
        end
    end
end
