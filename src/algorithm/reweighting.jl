# Single-histogram reweighting (Ferrenberg–Swendsen 1988). A Monte Carlo run at
# one temperature kbT₀ already contains, in its sampled energy histogram, enough
# information to compute canonical averages at NEARBY temperatures — no new
# simulation needed. A configuration sampled with weight ∝ e^{−β₀E} is reweighted
# to temperature β by the factor e^{−(β−β₀)E}, so
#
#   ⟨A⟩_β = ( Σ_i A(E_i) e^{−(β−β₀)E_i} ) / ( Σ_i e^{−(β−β₀)E_i} ),   β = 1/kbT.
#
# The estimate is trustworthy only where the β-histogram overlaps the sampled
# β₀-histogram (a window of width ~1/√fluctuations around kbT₀); it is the basis
# of multi-histogram/WHAM. Sums use a log-sum-exp shift for numerical stability.

"""
    sample_energies(rng, grids, lat, model, updater; kbT, sweeps,
                    therm=sweeps÷10, interval=1) -> Vector{Float64}

Advance `grids` with `updater` at temperature `kbT` and return the total energy
recorded every `interval` sweeps after `therm` thermalisation sweeps. `grids` is
mutated in place.
"""
function sample_energies(
    rng::AbstractRNG,
    grids::AbstractVector,
    lat::AbstractLattice,
    model::AbstractModel,
    updater::UpdateAlgorithm;
    kbT::Float64,
    sweeps::Int,
    therm::Int=sweeps ÷ 10,
    interval::Int=1,
)
    energies = Float64[]
    for s in 1:(therm + sweeps)
        update_step!(rng, grids, lat, model, updater; kbT=kbT)
        if s > therm && (s - therm) % interval == 0
            push!(energies, total_energy(grids, lat, model))
        end
    end
    return energies
end
export sample_energies

"""
    reweight_mean(energies, kbT0, kbT, f=identity) -> Float64

Ferrenberg–Swendsen single-histogram estimate of ⟨f(E)⟩ at `kbT` from total
energies sampled at `kbT0`.
"""
function reweight_mean(
    energies::AbstractVector{<:Real}, kbT0::Float64, kbT::Float64, f=identity
)
    isempty(energies) && throw(ArgumentError("no sampled energies to reweight"))
    Δβ = 1.0 / kbT - 1.0 / kbT0
    m = maximum(E -> -Δβ * E, energies)                  # log-sum-exp shift = max exponent
    num = 0.0
    den = 0.0
    for E in energies
        w = exp(-Δβ * E - m)
        num += f(E) * w
        den += w
    end
    return num / den
end
export reweight_mean

"""
    reweight_specific_heat(energies, kbT0, kbT, N) -> Float64

Per-spin specific heat C = (⟨E²⟩−⟨E⟩²)/(kbT²·N) at `kbT` reweighted from a run
at `kbT0` (`N` = number of spins).
"""
function reweight_specific_heat(
    energies::AbstractVector{<:Real}, kbT0::Float64, kbT::Float64, N::Int
)
    Em = reweight_mean(energies, kbT0, kbT)
    E2 = reweight_mean(energies, kbT0, kbT, abs2)
    return (E2 - Em^2) / (kbT^2 * N)
end
export reweight_specific_heat
