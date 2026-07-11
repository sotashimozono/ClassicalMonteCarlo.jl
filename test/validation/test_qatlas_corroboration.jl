# ─────────────────────────────────────────────────────────────────────────────
# Layer B — QAtlas corroboration.
#
# This package is the independent CLASSICAL numerical oracle in QAtlas's
# dual-oracle scheme: here the MC engine corroborates QAtlas's declared
# thermodynamic-limit closed forms (Onsager Tc, Yang M, critical exponent β).
#
# Heavy legs (large L, temperature sweeps) run only when CMC_TEST_FULL=1; the
# light lane still fetches every oracle value and runs a small-L MC so the
# QAtlas path is exercised on every PR.
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "mc_helpers.jl"))
include(joinpath(@__DIR__, "..", "ci", "universe.jl"))

const CMC_FULL = get(ENV, "CMC_TEST_FULL", "0") == "1"

# All four legs of this file are QAtlas-corroboration cases; a shard that
# selects none of them never loads QAtlas at all.
const _QATLAS_CASES = (
    "qatlas_oracle_sanity",
    "qatlas_yang_mag",
    "qatlas_binder_crossing",
    "qatlas_beta_consistency",
)

if any(case_selected, _QATLAS_CASES)
    using QAtlas

    if !CMC_FULL
        @info "test_qatlas_corroboration: running LIGHT lane (set CMC_TEST_FULL=1 for the heavy large-L sweeps)"
    end

    @testset "QAtlas corroboration (layer B)" begin
        Tc = QAtlas.fetch(IsingSquare(), CriticalTemperature())

        run_case("qatlas_oracle_sanity") do
            @testset "oracle sanity (exact, always run)" begin
                # QAtlas Tc must equal the Onsager closed form 2/log(1+√2).
                @test Tc ≈ 2 / log(1 + sqrt(2)) atol = 1e-12
                # Critical exponents NamedTuple; boundary-condition tag is Infinite()
                # (verified in QAtlas src/models/classical/IsingSquare/IsingSquare.jl).
                ex = QAtlas.fetch(IsingSquare(), CriticalExponents(), QAtlas.Infinite())
                @test float(ex.β) == 0.125            # β = 1//8, Onsager universality
                @test float(ex.ν) == 1.0
            end
        end

        run_case("qatlas_yang_mag") do
            @testset "Yang spontaneous magnetization vs MC (below Tc)" begin
                # Finite-lattice ⟨|M|⟩ is biased slightly ABOVE the TDL Yang value
                # (|M| cannot vanish on a finite system); well below Tc this bias is
                # tiny. Measured L=8/T=1.8 bias ≈ +9e-4, L=16 ≈ +1e-4 — so atol below
                # bounds finite-size + statistical error with large margin.
                L = CMC_FULL ? 16 : 8
                Ts = CMC_FULL ? (1.8, 1.5) : (1.8,)
                for T in Ts
                    β = 1 / T
                    M_yang = QAtlas.fetch(
                        IsingSquare(), SpontaneousMagnetization(); β=β, J=1.0
                    )
                    @test M_yang > 0                  # T < Tc ⇒ ordered
                    alg = LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection())
                    est = mc_estimate(L, T, alg; R=6, burn=3000, nsteps=6000, seed0=7000)
                    @testset "L=$L T=$T" begin
                        # atol = 0.02: finite-size upward bias (≲ 1e-3 here) + a few·SEM.
                        @test est.mAbsM ≈ M_yang atol = 0.02
                    end
                end
            end
        end

        run_case("qatlas_binder_crossing") do
            @testset "Onsager Tc via Binder-cumulant crossing" begin
                # U₄(L) curves for two sizes cross at ≈ Tc. Below Tc the larger lattice
                # is more ordered (higher U₄ ⇒ D = U₄(L2) − U₄(L1) > 0); above Tc the
                # ordering collapses fastest on the larger lattice (D < 0). The sign
                # flip localizes the crossing — hence Tc — to the bracket [Tc−δ, Tc+δ].
                #
                # Below Tc both curves saturate toward 2/3, so D is only weakly positive
                # there; the discriminating, robust signal is the strong D < 0 above Tc.
                L1, L2 = CMC_FULL ? (8, 16) : (6, 12)
                δ = 0.5
                nseed = CMC_FULL ? 8 : 4
                burn1, ns1 = 2000, 6000
                burn2, ns2 = 2500, 6000

                Dlo =
                    binder_mean(L2, Tc - δ; burn=burn2, nsteps=ns2, nseed=nseed) -
                    binder_mean(L1, Tc - δ; burn=burn1, nsteps=ns1, nseed=nseed)
                Dhi =
                    binder_mean(L2, Tc + δ; burn=burn2, nsteps=ns2, nseed=nseed) -
                    binder_mean(L1, Tc + δ; burn=burn1, nsteps=ns1, nseed=nseed)

                @test Dlo > 0        # larger lattice more ordered below Tc
                @test Dhi < 0        # larger lattice less ordered above Tc (robust signal)
                @test Dlo > Dhi      # crossing bracketed ⇒ Tc within [Tc−δ, Tc+δ]

                if CMC_FULL
                    # Sharper localization on the strong side in the heavy lane.
                    @test Dhi < -0.05
                end
            end
        end

        run_case("qatlas_beta_consistency") do
            @testset "critical exponent β consistency" begin
                # Oracle value (exact rational) — exercised every PR.
                ex = QAtlas.fetch(IsingSquare(), CriticalExponents(), QAtlas.Infinite())
                @test float(ex.β) ≈ 1 / 8

                # Independent physical consistency: the order parameter must GROW as T
                # decreases below Tc (β > 0). Compare MC ⟨|M|⟩ at two sub-Tc temperatures
                # against the QAtlas Yang curve — MC must reproduce the same ordering.
                L = CMC_FULL ? 16 : 8
                Thi, Tlo = 2.0, 1.6            # both < Tc
                alg = LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection())
                m_hi =
                    mc_estimate(L, Thi, alg; R=6, burn=2500, nsteps=5000, seed0=8000).mAbsM
                m_lo =
                    mc_estimate(L, Tlo, alg; R=6, burn=2500, nsteps=5000, seed0=8100).mAbsM
                y_hi = QAtlas.fetch(IsingSquare(), SpontaneousMagnetization(); β=1 / Thi)
                y_lo = QAtlas.fetch(IsingSquare(), SpontaneousMagnetization(); β=1 / Tlo)
                @test m_lo > m_hi              # MC: order grows as T falls (β > 0)
                @test y_lo > y_hi             # Yang closed form: same monotonicity
                @test m_hi ≈ y_hi atol = 0.02  # and MC tracks the Yang magnitude
                @test m_lo ≈ y_lo atol = 0.02
            end
        end
    end
end
