# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for the physics-validation suite (test/validation/).
#
# This file is NOT auto-discovered by runtests.jl (it is not named `test_*.jl`);
# each validation test file `include`s it. The `isdefined` guard makes repeated
# includes idempotent (the files are included sequentially into the same scope).
#
# Nothing here re-derives a quantity from the MC code's own answer: the exact
# routines are an INDEPENDENT brute-force Boltzmann enumeration, and the MC
# estimators return honest statistical error bars from independent chains.
# ─────────────────────────────────────────────────────────────────────────────
if !isdefined(Main, :MC_VALIDATION_HELPERS_LOADED)
    const MC_VALIDATION_HELPERS_LOADED = true

    using ClassicalMonteCarlo, Lattice2D
    using Random, Statistics

    """
        exact_ising(Lx, Ly, β; J=1.0) -> (Z, E, absM, M2)

    Brute-force exact enumeration of the 2D Ising model on the `Lx × Ly`
    `build_lattice(Square, …)` lattice (full periodic boundaries — see the
    constructor default `boundary = PeriodicAxis()`).

    Every configuration's energy is taken from the package primitive
    `total_energy` (which is itself covered by the mechanical unit tests), then
    Boltzmann-weighted here **independently** of any MC machinery. Returns:
    - `Z`     : partition function Σ exp(-βE)
    - `E`     : energy density ⟨E⟩ / N
    - `absM`  : ⟨|M|⟩ with M = |Σ sᵢ| / N   (magnetization density)
    - `M2`    : ⟨M²⟩ with M = (Σ sᵢ) / N     (|M|² = M², so abs is irrelevant)

    Only feasible for N = Lx·Ly ≲ 20 (2^N states).
    """
    function exact_ising(Lx::Int, Ly::Int, β::Float64; J::Float64=1.0)
        lat = build_lattice(Square, Lx, Ly)
        N = num_sites(lat)
        @assert N ≤ 24 "exact enumeration only for N ≤ 24 (got $N)"
        model = IsingModel(; J=J, h=0.0)
        Z = 0.0
        sE = 0.0
        sAM = 0.0
        sM2 = 0.0
        g = Vector{Int}(undef, N)
        for c in 0:(2 ^ N - 1)
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            E = total_energy(g, lat, model)
            w = exp(-β * E)
            s = sum(g)
            Mden = abs(s) / N
            Z += w
            sE += w * E
            sAM += w * Mden
            sM2 += w * Mden^2
        end
        return (Z=Z, E=sE / Z / N, absM=sAM / Z, M2=sM2 / Z)
    end

    """
        mc_chain(seed, lat, model, alg, T; burn, nsteps, interval=10) -> (E, absM, M2)

    One equilibrated Monte Carlo chain: `burn` sweeps of burn-in with no
    observers, then `nsteps` sweeps accumulating a `ThermodynamicObserver`.
    Returns the chain-mean energy density, ⟨|M|⟩ density and ⟨M²⟩ density.
    """
    function mc_chain(
        seed::Int, lat, model, alg, T::Float64; burn::Int, nsteps::Int, interval::Int=10
    )
        N = num_sites(lat)
        rng = MersenneTwister(seed)
        grids = rand(rng, [-1, 1], N)
        run!(rng, grids, lat, model, alg, AbstractObserver[]; kbT=T, nsteps=burn)
        obs = ThermodynamicObserver(; interval=interval)
        run!(rng, grids, lat, model, alg, AbstractObserver[obs]; kbT=T, nsteps=nsteps)
        res = get_thermodynamics(obs, T, N, model)
        return (E=res["Energy"], absM=res["Magnetization"], M2=obs.sum_M2 / obs.n_samples)
    end

    """
        mc_estimate(L, T, alg; R, burn, nsteps, interval=10, seed0=1000)

    Run `R` **independent** MC chains (distinct seeds) and return the grand mean
    of each observable together with the standard error of the mean estimated
    from the spread ACROSS chains:  SEM = std(chain means) / √R.

    Using the across-chain spread sidesteps within-chain autocorrelation: each
    chain mean is one quasi-independent draw, so `mean ± k·SEM` is an honest
    confidence band with no hand-tuning.
    """
    function mc_estimate(
        L::Int,
        T::Float64,
        alg;
        R::Int,
        burn::Int,
        nsteps::Int,
        interval::Int=10,
        seed0::Int=1000,
        J::Float64=1.0,
    )
        lat = build_lattice(Square, L, L)
        model = IsingModel(; J=J, h=0.0)
        Es = Float64[]
        AMs = Float64[]
        M2s = Float64[]
        for r in 1:R
            c = mc_chain(
                seed0 + 17r, lat, model, alg, T; burn=burn, nsteps=nsteps, interval=interval
            )
            push!(Es, c.E)
            push!(AMs, c.absM)
            push!(M2s, c.M2)
        end
        sem(v) = R > 1 ? std(v) / sqrt(R) : Inf
        return (
            mE=mean(Es),
            semE=sem(Es),
            mAbsM=mean(AMs),
            semAbsM=sem(AMs),
            mM2=mean(M2s),
            semM2=sem(M2s),
        )
    end

    """
        binder_mean(L, T; burn, nsteps, nseed=4, interval=10, seed0=900)

    Seed-averaged Binder cumulant U₄ = 1 − ⟨M⁴⟩/(3⟨M²⟩²) at (L, T).
    """
    function binder_mean(
        L::Int,
        T::Float64;
        burn::Int,
        nsteps::Int,
        nseed::Int=4,
        interval::Int=10,
        seed0::Int=900,
    )
        lat = build_lattice(Square, L, L)
        model = IsingModel(; J=1.0, h=0.0)
        us = Float64[]
        for s in 1:nseed
            rng = MersenneTwister(seed0 * s + L)
            grids = rand(rng, [-1, 1], num_sites(lat))
            run!(
                rng,
                grids,
                lat,
                model,
                LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection()),
                AbstractObserver[];
                kbT=T,
                nsteps=burn,
            )
            obs = ThermodynamicObserver(; interval=interval)
            run!(
                rng,
                grids,
                lat,
                model,
                LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection()),
                AbstractObserver[obs];
                kbT=T,
                nsteps=nsteps,
            )
            push!(us, get_thermodynamics(obs, T, num_sites(lat), model)["BinderParam"])
        end
        return mean(us)
    end

    # k-sigma band constant used throughout the MC-vs-independent-expectation
    # comparisons.  k = 4 ⇒ two-sided Gaussian false-alarm ≈ 6e-5 per test;
    # with the (fatter-tailed) t-distribution from R≈8–12 chains still ≲ 1e-3.
    const KSIGMA = 4
end
