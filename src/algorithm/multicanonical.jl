# Multicanonical (MUCA) sampling (Berg–Neuhaus 1991). A production run with a
# FIXED multicanonical weight W(E) = 1/g(E) = e^{−S(E)}, S(E) = ln g(E), makes
# the energy histogram flat: configs are drawn ∝ e^{−S(E)}, so the sampled
# energy density ∝ g(E)e^{−S(E)} = const. Unlike Wang–Landau — which keeps
# modifying its weight and is only asymptotically correct — the MUCA weight is
# fixed, so detailed balance holds exactly and the run is unbiased. Canonical
# averages at ANY temperature come from a single flat run by reweighting each
# sample by e^{−βE + S(E)}:
#
#   ⟨A⟩_β = ( Σ_i A(E_i) e^{−βE_i + S(E_i)} ) / ( Σ_i e^{−βE_i + S(E_i)} ).
#
# The weight S(E) is normally the ln g(E) learned by a prior Wang–Landau run.

"""
    Multicanonical(; sweeps=500_000, therm=50_000)

Parameters for a [`multicanonical`](@ref) production run: `sweeps` measurement
sweeps after `therm` thermalisation sweeps.
"""
@kwdef struct Multicanonical
    sweeps::Int = 500_000
    therm::Int = 50_000
end
export Multicanonical

"""
    muca_logweight(energies, logg; energy_quantum=1.0) -> Function

Build the multicanonical log-weight S(E) = ln g(E) as a lookup over the tabulated
`(energies, logg)` (e.g. a Wang–Landau result). Energies outside the tabulated
support map to `-Inf`, so multicanonical moves never leave the known range.
"""
function muca_logweight(energies, logg; energy_quantum::Float64=1.0)
    tab = Dict(
        round(Int, energies[i] / energy_quantum) => float(logg[i]) for
        i in eachindex(energies)
    )
    return E -> get(tab, round(Int, E / energy_quantum), -Inf)
end
export muca_logweight

"""
    multicanonical(rng, grids, lat, model::IsingModel, logweight, alg::Multicanonical)
        -> Vector{Float64}

Run a fixed-weight multicanonical simulation (single-spin flips accepted with
min(1, e^{S(E)−S(E')})) and return the sampled total energies. `logweight` is the
S(E) = ln g(E) callable (see [`muca_logweight`](@ref)); `grids` is mutated.
"""
function multicanonical(
    rng::AbstractRNG,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    logweight,
    alg::Multicanonical,
)
    N = num_sites(lat)
    E = total_energy(grids, lat, model)
    isfinite(logweight(E)) ||
        throw(ArgumentError("initial energy $E lies outside the weight support"))
    energies = Float64[]
    for sweep in 1:(alg.therm + alg.sweeps)
        for _ in 1:N
            site = rand(rng, 1:N)
            s_old = grids[site]
            hs = 0
            for j in neighbors(lat, site)
                hs += grids[j]
            end
            Enew = E + 2.0 * s_old * (model.J * hs + model.h)
            Snew = logweight(Enew)
            isfinite(Snew) || continue                 # reject moves leaving the support
            Sold = logweight(E)
            if Snew <= Sold || rand(rng) < exp(Sold - Snew)
                grids[site] = -s_old
                E = Enew
            end
        end
        sweep > alg.therm && push!(energies, E)
    end
    return energies
end
export multicanonical

"""
    muca_mean(energies, logweight, kbT, f=identity) -> Float64

Canonical ⟨f(E)⟩ at temperature `kbT` from multicanonical energy samples,
reweighting each by e^{−βE + S(E)} (log-sum-exp-stabilised).
"""
function muca_mean(energies::AbstractVector{<:Real}, logweight, kbT::Real, f=identity)
    isempty(energies) && throw(ArgumentError("no samples to reweight"))
    β = 1.0 / kbT
    xs = [-β * E + logweight(E) for E in energies]
    m = maximum(xs)
    num = 0.0
    den = 0.0
    for (E, x) in zip(energies, xs)
        w = exp(x - m)
        num += f(E) * w
        den += w
    end
    return num / den
end
export muca_mean
