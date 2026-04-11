"""
    process_site_selection!(rng, selection_method, grids, lat, model, alg; kwargs...)

Iterates through lattice sites based on the specified `selection_method` (Sweep vs Random)
and triggers single-site updates.

# Methods
- `::SequentialSweep`: Iterates from site `1` to `num_sites(lat)` in order.
- `::RandomSiteSelection`: Performs `num_sites(lat)` updates, choosing a random site index each time.
"""
function process_site_selection! end

function process_site_selection!(
    rng::AbstractRNG,
    ::SequentialSweep, # dispatch target
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::UpdateAlgorithm;
    kwargs...,
) where {T}
    for site in 1:(num_sites(lat))
        update_single_site!(rng, site, grids, lat, model, alg; kwargs...)
    end
end

function process_site_selection!(
    rng::AbstractRNG,
    ::RandomSiteSelection,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::UpdateAlgorithm;
    kwargs...,
) where {T}
    for _ in 1:(num_sites(lat))
        site = rand(rng, 1:(num_sites(lat)))
        update_single_site!(rng, site, grids, lat, model, alg; kwargs...)
    end
end

"""
    calculate_diff_energy(grids, lat, model, changes; kwargs...) -> Float64

Calculates the energy difference \\Delta E = E_{new} - E_{old} caused by the proposed `changes`.

# Arguments
- `changes`: A tuple of `LocalChange` structs.
    - If a single change is passed, it uses `local_hamiltonian` difference.
    - If multiple changes are passed, a specialized method must be implemented for the `model`.
"""
function calculate_diff_energy(
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    changes::Tuple{LocalChange{T}};
    kwargs...,
) where {T}
    c = changes[1]
    E_old = local_hamiltonian(grids, lat, model, c.index; val=c.old_val, kwargs...)
    E_new = local_hamiltonian(grids, lat, model, c.index; val=c.new_val, kwargs...)
    return E_new - E_old
end

function calculate_diff_energy(
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    changes::Tuple{Vararg{LocalChange{T}}};
    kwargs...,
) where {T}
    return error(
        "Generic calculate_diff_energy is not implemented for multi-site updates on (typeof(model)). Please implement a specific method in models/(typeof(model)).jl.",
    )
end

"""
    check_acceptance(rng, rule, dE, kbT) -> Bool

Determines if a proposed move with energy difference `dE` should be accepted at temperature `kbT`.

# Strategies
- `::Metropolis`: Accepts if r < e^{-\\Delta E / k_B T}. Always accepts if \\Delta E \\leq 0.
- `::Glauber`: Accepts if r < \\frac{1}{1 + e^{\\Delta E / k_B T}}.
"""
function check_acceptance(
    rng::AbstractRNG, ::Metropolis, dE::Float64, kbT::Float64; kwargs...
)
    if dE <= 0
        return true
    elseif kbT <= 0
        return false
    else
        return rand(rng) < exp(-dE / kbT)
    end
end

function check_acceptance(rng::AbstractRNG, ::Glauber, dE::Float64, kbT::Float64; kwargs...)
    if kbT <= 0
        return dE < 0
    else
        return rand(rng) < 1.0 / (1.0 + exp(dE / kbT))
    end
end
"""
    update_step!(rng, grids, lat, model, alg::LocalUpdate; kwargs...)

Performs one full Monte Carlo step (MCS).
For a local update algorithm, this typically consists of N single-site update attempts,
where N is the system size.
"""
function update_step!(
    rng::AbstractRNG,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::LocalUpdate;
    kwargs...,
) where {T}
    return process_site_selection!(rng, alg.selection, grids, lat, model, alg; kwargs...)
end
"""
    update_single_site!(rng, site, grids, lat, model, alg; kbT, kwargs...)

The core logic for a local Monte Carlo update at a specific `site`.

1. **Propose**: Generates a set of `LocalChange` candidates using `alg.proposal`.
2. **Energy Diff**: Calculates \\Delta E resulting from the proposed changes.
3. **Accept/Reject**: Decides whether to accept the change based on `alg.rule` (e.g., Metropolis) and temperature `kbT`.
4. **Update**: If accepted, applies the changes to `grids`.
"""
function update_single_site!(
    rng::AbstractRNG,
    site::Int,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::LocalUpdate;
    kbT::Float64=1.0,
    kwargs...,
) where {T}
    changes = propose(rng, alg.proposal, grids, lat, model, site)

    if isempty(changes)
        return nothing
    end

    dE = calculate_diff_energy(grids, lat, model, changes; kwargs...)

    if check_acceptance(rng, alg.rule, dE, kbT; kwargs...)
        for c in changes
            grids[c.index] = c.new_val
        end
    end
end
