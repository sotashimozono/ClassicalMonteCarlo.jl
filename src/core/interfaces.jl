# Order of arguments:
# rng, (dispatch target), grids, lat, model, alg, observers
# others will be passed with kwargs... = kbT, nsteps
# =========================================================
# common interfaces
# =========================================================
"""
    run!(rng, grids, lat, model, alg, observers; kbT=1.0, nsteps=1000, kwargs...)
Run a Monte Carlo simulation with the specified parameters, updating the spin configuration `grids` in place.

# Arguments
- `rng::AbstractRNG`: random number generator for stochastic processes.
- `grids::AbstractVector{T}`: it represents the spin configuration of the lattice, where `T` is the type of spin (e.g., `Int` for Ising spins).
- `lat::AbstractLattice`: lattice structure (topology, size, adjacency information).
- `model::AbstractModel{T}`: definition of the physical model (Ising, Potts, XY, etc.).
- `alg::UpdateAlgorithm`: update algorithm (e.g., `LocalUpdate`).
- `observers::AbstractVector`: list of `AbstractObserver`s for measurements.
# Keyword Arguments
- `kbT::Float64`:  k_BT value for the simulation. Default is `1.0`.
- `nsteps::Int`: total number of simulation steps (MCS). Default is `1000`.
- `kwargs...`: other algorithm-specific parameters.

# Execution Flow
1. **Initial Observation**: Record the state at step 0 using `observe!`.
2. **Simulation Loop**: Loop for `nsteps` iterations.
    - `update_step!`: Perform one Monte Carlo step update.
    - **Interval Observation**: Call `observe!` when `step % obs.interval == 0`.
"""
function run!(
    rng::AbstractRNG,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::UpdateAlgorithm,
    observers::AbstractVector{<:AbstractObserver};
    kbT::Float64=1.0,
    nsteps::Int=1000,
    kwargs...,
) where {T}
    for obs in observers
        observe!(obs, grids, lat, model, 0; kwargs..., kbT=kbT)
    end

    for step in 1:nsteps
        update_step!(rng, grids, lat, model, alg; kwargs..., kbT=kbT)

        # observation
        for obs in observers
            if step % obs.interval == 0
                observe!(obs, grids, lat, model, step; kwargs..., kbT=kbT)
            end
        end
    end
end
"""
    run(rng, grids, lat, model, alg, observers; kwargs...) -> Vector{T}

[`run!`](@ref) that returns a new spin configuration without modifying the original `grids`.
"""
function run(
    rng::AbstractRNG,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::UpdateAlgorithm,
    observers::AbstractVector{<:AbstractObserver};
    kwargs...,
) where {T}
    grids_copy = copy(grids)
    run!(rng, grids_copy, lat, model, alg, observers; kwargs...)
    return grids_copy
end
export run!, run
"""
    propose(rng, alg, grids, lat, model, site) -> Tuple{LocalChange, ...}

Proposes a new candidate state for a local update algorithm.
This function does not actually modify `grids`, but returns a tuple of [`LocalChange`](@ref) representing the proposed modifications.

# Arguments
- `alg::ProposalMethod`: The proposal method (e.g., [`SpinFlip`](@ref), [`SpinExchange`](@ref)).
- `site::Int`: The index of the site to be updated.

# Returns
- `Tuple{Vararg{LocalChange{T}}}`: A tuple containing change information for one or more sites.
"""
function propose(
    rng::AbstractRNG,
    alg::ProposalMethod,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    site::Int,
) where {T}
    return error("propose is not implemented for (typeof(alg)) and (typeof(model))")
end
export propose
"""
    local_hamiltonian(grids, lat, model, site; val=grids[site]) -> Float64

Calculates the local interaction energy around the specified `site`.
A virtual value `val` can be specified instead of the current state `grids[site]` for calculating energy differences.

# Arguments
- `site::Int`: The site index for calculation.
- `val::T`: The state value of the site. Defaults to the current `grids[site]`.

# Returns
- `Float64`: The local energy value.
"""
function local_hamiltonian(
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    site::Int;
    val::T=grids[site],
) where {T}
    return error("local_hamiltonian not implemented for (typeof(model))")
end
export local_hamiltonian

"""
    total_energy(grids, lat, model) -> Float64

Calculates the total energy of the system.
The default implementation sums up [`local_hamiltonian`](@ref) for all sites and divides by `2.0` to correct for double counting of interactions.

!!! warning
    For models containing terms that do not result in double counting (e.g., external field terms), this default implementation may be inaccurate.
    In such cases, please override this function for the specific model.
"""
function total_energy(
    grids::AbstractVector{T}, lat::AbstractLattice, model::AbstractModel{T}
) where {T}
    energy = 0.0
    for site in 1:(num_sites(lat))
        energy += local_hamiltonian(grids, lat, model, site)
    end
    return energy / 2.0
end
export total_energy
"""
    calculate_diff_energy(grids, lat, model, changes) -> Float64

Efficiently calculates the energy difference \\Delta E associated with the proposed `changes`.
Instead of recalculating the total energy, it considers only the changed sites and their neighbors.

# Arguments
- `changes`: A tuple of [`LocalChange`](@ref).
"""
function calculate_diff_energy(
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    changes::Tuple{LocalChange{T},LocalChange{T},Vararg{LocalChange{T}}},
) where {T}
    return error(
        "Generic calculate_diff_energy is not implemented for multi-site updates on (typeof(model)).",
    )
end
export calculate_diff_energy
"""
    update_step!(rng, grids, lat, model, alg; kbT=1.0)

Performs updates for one Monte Carlo step (1 MCS).
The specific process is dispatched to the type of `alg` (e.g., [`LocalUpdateAlgorithm`](@ref) or [`ClusterUpdateAlgorithm`](@ref)).
"""
function update_step!(
    rng::AbstractRNG,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::UpdateAlgorithm;
    kbT::Float64=1.0,
) where {T}
    return error("update_step! not implemented for algorithm (typeof(alg))")
end
export update_step!
"""
    process_site_selection!(rng, S, grids, lat, model, alg; kbT=1.0)

Called as part of the [`LocalUpdate`](@ref) algorithm.
Selects a site based on the [`SiteSelectionMethod`](@ref) `S`, then executes `propose` and `check_acceptance`.

# Arguments
- `S::SiteSelectionMethod`: The site selection strategy (e.g., [`RandomSiteSelection`](@ref)).
"""
function process_site_selection!(
    rng::AbstractRNG,
    S::SiteSelectionMethod, # dispatch target
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::UpdateAlgorithm;
    kbT::Float64=1.0,
) where {T}
    return error("Unsupported site selection method (S) for algorithm.")
end
export process_site_selection!
"""
    check_acceptance(rng, rule, dE; kbT) -> Bool

Determines whether to accept the state transition based on the energy difference `dE` and temperature `kbT`.

# Arguments
- `rule::AcceptanceRule`: The acceptance rule (e.g., [`Metropolis`](@ref), [`Glauber`](@ref)).
- `dE::Float64`: The energy difference (E_{new} - E_{old}).

# Returns
- `Bool`: `true` if the transition is accepted.
"""
function check_acceptance(rng::AbstractRNG, rule::AcceptanceRule, dE::Float64; kbT::Float64)
    return error("check_acceptance not implemented for (typeof(rule))")
end
export check_acceptance
"""
    observe!(obs, grids, lat, kbT, model, step)

Measures and records physical quantities from the current state.
This function is called periodically within the [`run!`](@ref) loop.

# Arguments
- `obs::AbstractObserver`: The specific observer instance (e.g., [`FunctionObserver`](@ref)).
- `step::Int`: The current Monte Carlo step number.
"""
function observe!(
    obs::AbstractObserver,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    step::Int;
    kwargs...,
) where {T}
    return error("observe! not implemented for (typeof(obs))")
end
export observe!
