# ─────────────────────────────────────────────────────────────────────────────
# Regression test for the observe! dispatch bug (the PART-1 fix).
#
# Before the fix the concrete observe! methods carried a stale 6-positional
# signature (obs, grids, lat, kbT, model, step) that never matched run!'s call
# observe!(obs, grids, lat, model, step; kbT=…), so every real observer fell
# through to the generic method and threw
#     "observe! not implemented for (typeof(obs))".
# This test drives a full run! with a ThermodynamicObserver end-to-end; it
# THROWS on the unfixed code and only passes once dispatch is repaired.
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "mc_helpers.jl"))
include(joinpath(@__DIR__, "..", "ci", "universe.jl"))

run_case("run_observer_e2e") do
    @testset "run! + ThermodynamicObserver end-to-end (dispatch regression)" begin
        L = 8
        lat = build_lattice(Square, L, L)
        N = num_sites(lat)
        model = IsingModel(; J=1.0, h=0.0)
        alg = LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection())
        T = 2.5

        grids = rand(MersenneTwister(1), [-1, 1], N)
        obs = ThermodynamicObserver(; interval=10)

        # Must not throw (this is the actual bug surface).
        run!(
            MersenneTwister(2),
            grids,
            lat,
            model,
            alg,
            AbstractObserver[obs];
            kbT=T,
            nsteps=200,
        )

        @test obs.n_samples > 0

        d = get_thermodynamics(obs, T, N, model)
        @test !isempty(d)
        @test d["Samples"] == obs.n_samples

        # Physical sanity (independent bounds, not re-computations of MC output):
        #  energy density of a ±J Ising model with |neighbors| = 4 lies in [-2, 0]
        #  for any finite T ≥ 0 with h = 0 (paramagnet → 0, ferromagnet → −2J).
        @test -2.0 - 1e-9 ≤ d["Energy"] ≤ 1e-9
        @test 0.0 ≤ d["Magnetization"] ≤ 1.0 + 1e-9   # |M| density is in [0,1]
        @test isfinite(d["Magnetization"])
        @test d["Susceptibility"] ≥ 0.0               # χ = N·var(|M|)/T ≥ 0
        @test d["SpecificHeat"] ≥ 0.0                 # C = var(E)/(T²N) ≥ 0
        @test isfinite(d["SpecificHeat"])
        @test isfinite(d["BinderParam"])

        # FunctionObserver goes through the same dispatch seam.
        fobs = FunctionObserver(
            "absM", (g, l, m) -> measure_magnetization(g, l, m); interval=10
        )
        run!(
            MersenneTwister(3),
            grids,
            lat,
            model,
            alg,
            AbstractObserver[fobs];
            kbT=T,
            nsteps=100,
        )
        @test length(fobs.history) > 0
        @test all(isfinite, fobs.history)
    end
end
