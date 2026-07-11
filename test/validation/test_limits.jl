# ─────────────────────────────────────────────────────────────────────────────
# Layer A — physical limits (closed-form endpoints of the MC dynamics).
#
#   T → 0  : the ferromagnetic ground state is fully ordered, E/site = −2J
#            (square lattice, 4 neighbours, PBC) and |M| = 1.
#   T → ∞  : spins decorrelate, ⟨E⟩/N → 0 and ⟨M²⟩ (density) → 1/N
#            (⟨(Σs)²⟩ = N for independent ±1 spins ⇒ ⟨(Σs/N)²⟩ = 1/N).
# ─────────────────────────────────────────────────────────────────────────────
include(joinpath(@__DIR__, "mc_helpers.jl"))
include(joinpath(@__DIR__, "..", "ci", "universe.jl"))

@testset "physical limits" begin
    L = 8
    lat = build_lattice(Square, L, L)
    N = lat.N
    model = IsingModel(; J=1.0, h=0.0)

    run_case("limits_ising_lowT") do
        @testset "T → 0 ground state" begin
            # Closed-form endpoint, independent of MC: the ordered config has
            # E/site = −2J exactly (each of 4 PBC bonds per site shared by 2 sites).
            ordered = ones(Int, N)
            @test total_energy(ordered, lat, model) / N ≈ -2.0
            @test measure_magnetization(ordered, lat, model) ≈ 1.0

            # MC started ordered at very low T must stay frozen in the ground state:
            # a single flip costs ΔE = +8J, accepted with prob e^{-8/0.1} ≈ e^{-80}.
            alg = LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection())
            rng = MersenneTwister(7)
            grids = ones(Int, N)
            obs = ThermodynamicObserver(; interval=5)
            run!(rng, grids, lat, model, alg, AbstractObserver[obs]; kbT=0.1, nsteps=200)
            d = get_thermodynamics(obs, 0.1, N, model)
            @test d["Energy"] ≈ -2.0 atol = 1e-9
            @test d["Magnetization"] ≈ 1.0 atol = 1e-9
        end
    end

    run_case("limits_ising_highT") do
        @testset "T → ∞ paramagnet" begin
            Thot = 1.0e6
            alg = LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection())
            est = mc_estimate(L, Thot, alg; R=8, burn=500, nsteps=3000, seed0=6000)

            # ⟨E⟩/N → 0.  |E/N| bound is a few·SEM plus the O(J/T) = 1e-6 residual;
            # 0.02 is a safe absolute tolerance (measured deviation ≈ 3e-3).
            @test abs(est.mE) ≤ 0.02

            # ⟨M²⟩ (density) → 1/N.  Check within k·SEM of the exact infinite-T value.
            @test abs(est.mM2 - 1 / N) ≤ KSIGMA * est.semM2
        end
    end
end
