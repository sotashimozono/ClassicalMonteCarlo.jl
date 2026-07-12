# Heat-bath (Gibbs) single-site sampler — rejection-free: instead of
# propose→accept, each visited site is redrawn *directly* from its local
# Boltzmann conditional P(s) ∝ exp(−E_local(s)/kbT) over the site's state space,
# given its current neighbours. Every update resamples the site, so it mixes
# faster than Metropolis near equilibrium — the gain is largest for q-state
# Potts, where Metropolis wastes proposals on the q−1 other colours. For the
# Ising model heat-bath coincides with Glauber dynamics.

"""
    HeatBath(; selection=RandomSiteSelection())

Rejection-free single-site heat-bath (Gibbs) update. Requires the model to
implement [`heatbath_sample!`](@ref)`(rng, site, grids, lat, model; kbT)`
(provided here for `IsingModel` and `PottsModel`).
"""
@kwdef struct HeatBath{S<:SiteSelectionMethod} <: LocalUpdateAlgorithm
    selection::S = RandomSiteSelection()
end
export HeatBath

function update_step!(
    rng::AbstractRNG,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::HeatBath;
    kbT::Float64=1.0,
    kwargs...,
) where {T}
    return process_site_selection!(
        rng, alg.selection, grids, lat, model, alg; kwargs..., kbT=kbT
    )
end

# reuse the generic site sweep in localupdates.jl (dispatches on selection);
# it calls update_single_site!(…, alg::HeatBath), which draws from the conditional.
function update_single_site!(
    rng::AbstractRNG,
    site::Int,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::HeatBath;
    kbT::Float64=1.0,
    kwargs...,
) where {T}
    grids[site] = heatbath_sample!(rng, site, grids, lat, model; kbT=kbT)
    return nothing
end

"""
    heatbath_sample!(rng, site, grids, lat, model; kbT) -> new state

Draw the site's new state from the exact local Boltzmann conditional
P(s) ∝ exp(−E_local(s)/kbT), given the fixed neighbour configuration.
"""
function heatbath_sample! end
export heatbath_sample!

# Ising: two states {+1,−1}; P(+1) = 1/(1 + exp((E₊ − E₋)/kbT)) (= Glauber).
function heatbath_sample!(
    rng::AbstractRNG,
    site::Int,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel;
    kbT::Float64,
)
    Ep = local_hamiltonian(grids, lat, model, site; val=1)
    Em = local_hamiltonian(grids, lat, model, site; val=-1)
    if kbT <= 0
        return Ep <= Em ? 1 : -1
    end
    return rand(rng) < 1.0 / (1.0 + exp((Ep - Em) / kbT)) ? 1 : -1
end

# q-state Potts: sample s ∝ exp(−E_local(s)/kbT) via a stable cumulative draw.
function heatbath_sample!(
    rng::AbstractRNG,
    site::Int,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::PottsModel;
    kbT::Float64,
)
    q = model.q
    logw = Vector{Float64}(undef, q)
    for s in 1:q
        logw[s] = -local_hamiltonian(grids, lat, model, site; val=s)  # = +J·n_s
    end
    if kbT <= 0
        return argmax(logw)                       # T=0: pick a lowest-energy state
    end
    logw ./= kbT
    m = maximum(logw)
    total = sum(x -> exp(x - m), logw)
    r = rand(rng) * total
    acc = 0.0
    for s in 1:q
        acc += exp(logw[s] - m)
        r <= acc && return s
    end
    return q
end
