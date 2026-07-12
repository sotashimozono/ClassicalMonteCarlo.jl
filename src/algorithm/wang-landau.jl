# Wang–Landau flat-histogram algorithm — a density-of-states sampler, a
# fundamentally different paradigm from the equilibrium (Metropolis / Glauber /
# cluster) samplers here: a single run yields the log density of states
# log g(E) over the whole spectrum, and from it the *full* temperature
# dependence of every thermodynamic quantity follows (Z(T), ⟨E⟩, C, F, S).
# Reference: F. Wang & D. P. Landau, PRL 86, 2050 (2001).
#
# Scoped to DISCRETE-spectrum models (Ising, Potts): the energy is binned by
# rounding `E / energy_quantum` to an integer, so each reachable level gets its
# own histogram bin and the unreachable gaps simply never appear.

"""
    WangLandau(; flatness=0.8, ln_f_final=1e-8, check_interval=10_000,
               max_iters=10^9, energy_quantum=1.0, proposal=SpinFlip())

Wang–Landau density-of-states estimator. It is an `UpdateAlgorithm`, but unlike
the equilibrium samplers it does **not** run through the fixed-`kbT` [`run!`]
loop — drive it with [`wang_landau`](@ref). The modification factor `f` starts
at `e` (`ln f = 1`) and is refined `f → √f` each time the energy histogram is
`flatness`-flat, until `ln f < ln_f_final`.
"""
@kwdef struct WangLandau{P<:ProposalMethod} <: UpdateAlgorithm
    flatness::Float64 = 0.8
    ln_f_final::Float64 = 1e-8
    check_interval::Int = 10_000
    max_iters::Int = 10^9
    energy_quantum::Float64 = 1.0
    proposal::P = SpinFlip()
end
export WangLandau

# numerically stable log(Σ exp(xs))
_logsumexp(xs) = (m=maximum(xs); m + log(sum(x -> exp(x - m), xs)))

"""
    wang_landau(rng, grids, lat, model, alg::WangLandau) -> (energies, log_g)

Estimate the log density of states. `grids` is evolved in place (its final
configuration is arbitrary). Returns the sorted reachable energy levels
`energies::Vector{Float64}` and `log_g::Vector{Float64}` (natural log of g(E)),
shifted so `minimum(log_g) == 0`. WL leaves an overall additive constant in
`log g` undetermined; thermodynamic *ratios* (⟨E⟩, C — see
[`wl_thermodynamics`](@ref)) are independent of it, while absolute F / S require
normalising so that `Σ_E g(E)` equals the total number of states (`q^N`).
"""
function wang_landau(
    rng::AbstractRNG,
    grids::AbstractVector,
    lat::AbstractLattice,
    model::AbstractModel,
    alg::WangLandau,
)
    N = num_sites(lat)
    q = alg.energy_quantum
    ekey(E) = round(Int, E / q)

    E = total_energy(grids, lat, model)
    log_g = Dict{Int,Float64}(ekey(E) => 0.0)
    H = Dict{Int,Int}(ekey(E) => 0)
    ln_f = 1.0
    iters = 0

    while ln_f > alg.ln_f_final && iters < alg.max_iters
        for _ in 1:(alg.check_interval)
            iters += 1
            site = rand(rng, 1:N)
            changes = propose(rng, alg.proposal, grids, lat, model, site)
            isempty(changes) && continue
            dE = calculate_diff_energy(grids, lat, model, changes)
            knew = ekey(E + dE)
            get!(log_g, knew, 0.0)
            get!(H, knew, 0)
            # WL acceptance: min(1, g(E_old)/g(E_new)) = min(1, exp(logg_old - logg_new))
            if rand(rng) < exp(log_g[ekey(E)] - log_g[knew])
                for c in changes
                    grids[c.index] = c.new_val
                end
                E += dE
            end
            k = ekey(E)
            log_g[k] += ln_f
            H[k] += 1
        end
        hv = values(H)
        if minimum(hv) >= alg.flatness * (sum(hv) / length(hv))
            ln_f /= 2
            for kk in keys(H)
                H[kk] = 0
            end
        end
    end

    ks = sort!(collect(keys(log_g)))
    energies = Float64[k * q for k in ks]
    lg = Float64[log_g[k] for k in ks]
    lg .-= minimum(lg)
    return energies, lg
end
export wang_landau

"""
    wl_thermodynamics(energies, log_g, kbT) -> (; energy, energy2, C, logZ)

Thermodynamic averages at temperature `kbT` from a Wang–Landau density of
states, via log-sum-exp over the energy levels. `energy` (⟨E⟩) and `C`
(= (⟨E²⟩−⟨E⟩²)/kbT²) are independent of the additive constant left free in
`log_g`; `logZ` carries it (fix it by normalising `log_g` to `Σ g = q^N` first).
"""
function wl_thermodynamics(energies::AbstractVector, log_g::AbstractVector, kbT::Float64)
    β = 1.0 / kbT
    w = log_g .- β .* energies                 # log( g(E) e^{-βE} )
    lZ = _logsumexp(w)
    p = exp.(w .- lZ)                            # normalised Boltzmann weight per level
    E = sum(p .* energies)
    E2 = sum(p .* energies .^ 2)
    return (; energy=E, energy2=E2, C=(E2 - E^2) / kbT^2, logZ=lZ)
end
export wl_thermodynamics
