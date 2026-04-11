# =========================================================
# definition of Abstract Types
# =========================================================
"""
AbstractMonteCarlo: Abstract type for Monte Carlo simulation components.
"""
abstract type AbstractMonteCarlo end

"""
AbstractModel{T}: Abstract type for lattice models in Monte Carlo simulations.
T: The state type of the model (e.g., Int, Float64, Vector, etc.).
if you want to define a new model, subtype this abstract type.
"""
abstract type AbstractModel{T} <: AbstractMonteCarlo end
# T (Int, Float64, Vector etc.) represents the state type of the model
"""
AbstractObserver: Abstract type for observers in Monte Carlo simulations.
it is used to monitor physical quantities during the simulation.
"""
abstract type AbstractObserver <: AbstractMonteCarlo end
export AbstractObserver
"""
UpdateAlgorithm: Abstract type for update algorithms in Monte Carlo simulations.
"""
abstract type UpdateAlgorithm <: AbstractMonteCarlo end
"""
LocalUpdateAlgorithm: Abstract type for local update algorithms.
it is used for kawaski dynamics, metropolis updates, etc.
"""
abstract type LocalUpdateAlgorithm <: UpdateAlgorithm end
"""
ClusterUpdateAlgorithm: Abstract type for cluster update algorithms.
it is used for wolff algorithm, swendsen-wang algorithm, etc.
"""
abstract type ClusterUpdateAlgorithm <: UpdateAlgorithm end

"""
AcceptanceRule: Abstract type for acceptance rules in Monte Carlo simulations.
SiteSelectionMethod: Abstract type for site selection methods in Monte Carlo simulations.
"""
abstract type AcceptanceRule <: AbstractMonteCarlo end
"""
SiteSelectionMethod: Abstract type for site selection methods in Monte Carlo simulations.
"""
abstract type SiteSelectionMethod <: AbstractMonteCarlo end
"""
ProposalMethod: Abstract type for proposal methods in Monte Carlo simulations.
"""
abstract type ProposalMethod <: AbstractMonteCarlo end

export AcceptanceRule, SiteSelectionMethod, ProposalMethod
export AbstractModel, UpdateAlgorithm, LocalChange
# --- Local Changes ---
# Represents a local change at a single site, used in energy difference calculations
"""
LocalChange{T}: Represents a local change at a single site.
T: The state type of the model (e.g., Int, Float64, Vector, etc.).
"""
struct LocalChange{T}
    index::Int
    new_val::T
    old_val::T
end
# --- Proposals ---
"""
Spinfilp: Proposal method for flipping spins (e.g., in Ising model).
non conserved dynamics.
"""
struct SpinFlip <: ProposalMethod end
export SpinFlip
"""
SpinExchange: Proposal method for exchanging spins between two sites (e.g., in Kawasaki dynamics).
conserved dynamics.
"""
struct SpinExchange <: ProposalMethod end
export SpinExchange
"""
UniformShift(width::Float64 = 0.1)
A proposal method for continuous spin models (e.g., XY, Heisenberg).
Shifts the current value by a random amount drawn uniformly from `[-width/2, width/2]`.
"""
@kwdef struct UniformShift <: ProposalMethod
    width::Float64 = 0.1
end
export UniformShift

# --- Rules ---
"""
Metropolis: Acceptance rule based on the Metropolis criterion.
`e^(-dE/kbT)`
"""
struct Metropolis <: AcceptanceRule end
export Metropolis
"""
Glauber: Acceptance rule based on the Glauber dynamics.
`1 / (1 + e^(dE/kbT))`
"""
struct Glauber <: AcceptanceRule end
export Glauber

# --- Selections ---
"""
RandomSiteSelection: Site selection method that selects sites randomly.
when you consider non equilibrium dynamics, this method is recommended.
"""
struct RandomSiteSelection <: SiteSelectionMethod end
export RandomSiteSelection
"""
SequentialSweep: Site selection method that sweeps through sites sequentially.
"""
struct SequentialSweep <: SiteSelectionMethod end
export SequentialSweep

# --- Algorithms ---
"""
LocalUpdate{R<:AcceptanceRule,S<:SiteSelectionMethod,P<:ProposalMethod}: Local update algorithm combining acceptance rule R, site selection method S, and proposal method P.
this is a general structure for local update algorithms in Monte Carlo simulations.
"""
@kwdef struct LocalUpdate{R<:AcceptanceRule,S<:SiteSelectionMethod,P<:ProposalMethod} <:
              LocalUpdateAlgorithm
    rule::R = Metropolis()
    selection::S = RandomSiteSelection()
    proposal::P = SpinFlip()
end
export LocalUpdate
