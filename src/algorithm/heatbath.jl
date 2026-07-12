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

# Continuous XY spin: the local energy is −J|h|cos(θ − φ) with local field
# h = Σ_{n∈neighbours} e^{iθ_n}, φ = arg h, so the exact conditional is the
# von Mises distribution  P(θ) ∝ exp(κ cos(θ − φ)),  κ = J|h|/kbT — sampled
# rejection-free by the Best & Fisher (1979) algorithm. This is the continuous
# analogue of the Ising/Potts heat-bath and the rejection-free counterpart of
# the microcanonical `Overrelaxation` reflection on the same local field.
function heatbath_sample!(
    rng::AbstractRNG,
    site::Int,
    grids::AbstractVector{Float64},
    lat::AbstractLattice,
    model::XYModel;
    kbT::Float64,
)
    hx = 0.0
    hy = 0.0
    for n in neighbors(lat, site)
        hx += cos(grids[n])
        hy += sin(grids[n])
    end
    h = sqrt(hx^2 + hy^2)
    h < 1e-12 && return 2π * rand(rng)          # free spin: uniform on the circle
    φ = atan(hy, hx)
    if kbT <= 0                                 # T=0: align to (or anti-align from) φ
        return model.J >= 0 ? mod2pi(φ) : mod2pi(φ + π)
    end
    κ = model.J * h / kbT
    return κ >= 0 ? _sample_vonmises(rng, φ, κ) : _sample_vonmises(rng, φ + π, -κ)
end

# Best & Fisher (1979) rejection sampler for the von Mises distribution
# P(θ) ∝ exp(κ cos(θ − μ)), κ ≥ 0. Reduces to uniform as κ → 0.
function _sample_vonmises(rng::AbstractRNG, μ::Float64, κ::Float64)
    κ < 1e-10 && return 2π * rand(rng)
    a = 1.0 + sqrt(1.0 + 4.0 * κ^2)
    b = (a - sqrt(2.0 * a)) / (2.0 * κ)
    r = (1.0 + b^2) / (2.0 * b)
    while true
        z = cos(π * rand(rng))
        f = (1.0 + r * z) / (r + z)
        c = κ * (r - f)
        u2 = rand(rng)
        if c * (2.0 - c) - u2 > 0.0 || log(c / u2) + 1.0 - c >= 0.0
            θ = sign(rand(rng) - 0.5) * acos(clamp(f, -1.0, 1.0))
            return mod2pi(μ + θ)
        end
    end
end
