"""
    PottsModel(q::Int=3, J::Float64=1.0)

Represents the q-state Potts model. Spins take values s_i \\in \\{1, 2, \\dots, q\\}.

# Hamiltonian
H = -J \\sum_{\\langle i, j \\rangle} \\delta(s_i, s_j)
where \\delta is the Kronecker delta (1 if equal, 0 otherwise).
"""
@kwdef struct PottsModel <: AbstractModel{Int}
    q::Int = 3
    J::Float64 = 1.0
end
export PottsModel

function local_hamiltonian(
    grids::AbstractVector{Int},
    lat::Lattice,
    model::PottsModel,
    site::Int;
    val::Int=grids[site],
    kwargs...,
)
    energy = 0.0
    for neighbor in lat.nearest_neighbors[site]
        if val == grids[neighbor]
            energy -= model.J
        end
    end
    return energy
end
"""
    propose(rng, ::SpinFlip, ..., model::PottsModel, site)

Proposes a new state for the Potts model.
The new state is chosen uniformly at random from the q-1 states that are **not** the current state.
"""
function propose(
    rng::AbstractRNG,
    ::SpinFlip,
    grids::AbstractVector{Int},
    lat::Lattice,
    model::PottsModel,
    site::Int;
    kwargs...,
)
    current = grids[site]
    shift = rand(rng, 1:(model.q - 1))
    new_val = mod1(current + shift, model.q)
    return (LocalChange(site, new_val, current),)
end
get_binder_coeff(::PottsModel) = 3.0
"""
    measure_magnetization(grids, lat, model::PottsModel)

Calculates the order parameter for the Potts model based on the density of the majority spin.

# Formula
M = \\frac{q \\rho_{max} - 1}{q - 1}
where \\rho_{max} is the fraction of sites occupied by the most common spin state.
This ensures M=1 for a fully ordered state and M=0 for a disordered state.
"""
function measure_magnetization(grids, lat, model::PottsModel)
    counts = zeros(Int, model.q)
    for s in grids
        counts[s] += 1
    end
    m_frac = maximum(counts) / length(grids)
    return (model.q * m_frac - 1.0) / (model.q - 1.0)
end
