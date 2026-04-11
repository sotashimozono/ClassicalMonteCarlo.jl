"""
    IsingModel(J::Float64=1.0, h::Float64=0.0)

Represents the classical Ising model with spins s_i \\in \\{-1, 1\\}.

# Hamiltonian
H = -J \\sum_{\\langle i, j \\rangle} s_i s_j - h \\sum_i s_i
- `J`: Interaction constant (ferromagnetic if J>0).
- `h`: External magnetic field.
"""
@kwdef struct IsingModel <: AbstractModel{Int}
    J::Float64 = 1.0
    h::Float64 = 0.0
end
export IsingModel
"""
    local_hamiltonian(grids, lat, model::IsingModel, site; val)

Calculates the local energy contribution of a single site.
E_{local} = -J \\sum_{k \\in \\text{neighbors}} s_{\\text{site}} s_k - h s_{\\text{site}}
"""
function local_hamiltonian(
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    site::Int;
    val::Int=grids[site],
    kwargs...,
)
    energy = 0.0
    for neighbor in neighbors(lat, site)
        energy -= model.J * val * grids[neighbor]
    end
    energy -= model.h * val
    return energy
end
"""
    total_energy(grids, lat, model::IsingModel)

Calculates the total energy of the Ising system.
Includes a factor of `1/2` for the bond sum to avoid double counting edges.
"""
function Lattice2DMonteCarlo.total_energy(
    grids::AbstractVector{Int}, lat::AbstractLattice, model::IsingModel
)
    E_bond_sum = 0.0
    E_field_sum = 0.0

    for site in 1:(num_sites(lat))
        s = grids[site]
        for n in neighbors(lat, site)
            E_bond_sum -= model.J * s * grids[n]
        end
        E_field_sum -= model.h * s
    end

    return (E_bond_sum / 2.0) + E_field_sum
end
"""
    propose(rng, ::SpinFlip, ..., model::IsingModel, site)

Proposes a standard Glauber/Metropolis flip: s_{new} = -s_{old}.
"""
function propose(
    rng::AbstractRNG,
    ::SpinFlip,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    site::Int;
    kwargs...,
)
    current = grids[site]
    proposed = -current
    return (LocalChange(site, proposed, current),)
end

function propose(
    rng::AbstractRNG,
    ::SpinExchange,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    site1::Int;
    kwargs...,
)
    site2 = rand(rng, neighbors(lat, site1))

    v1 = grids[site1]
    v2 = grids[site2]

    if v1 == v2
        return ()
    end

    return (LocalChange(site1, v2, v1), LocalChange(site2, v1, v2))
end
"""
    calculate_diff_energy(grids, lat, model::IsingModel, changes::Tuple{LocalChange, LocalChange})

Calculates the energy difference for a two-site update (Spin Exchange).

# Logic
It sums the local energy differences of both sites.
**Crucially**, if the swapped sites are nearest neighbors, it subtracts the bond energy correction
to prevent double-counting the interaction between `site1` and `site2`.
"""
function Lattice2DMonteCarlo.calculate_diff_energy(
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    changes::Tuple{LocalChange{Int},LocalChange{Int}};
    kwargs...,
)
    c1 = changes[1]
    c2 = changes[2]

    is_neighbor = c2.index in neighbors(lat, c1.index)

    dE1 =
        local_hamiltonian(grids, lat, model, c1.index; val=c1.new_val) -
        local_hamiltonian(grids, lat, model, c1.index; val=c1.old_val)

    dE2 =
        local_hamiltonian(grids, lat, model, c2.index; val=c2.new_val) -
        local_hamiltonian(grids, lat, model, c2.index; val=c2.old_val)
    total_dE = dE1 + dE2
    if is_neighbor
        term1 = -model.J * (c1.new_val - c1.old_val) * c2.old_val
        term2 = -model.J * (c2.new_val - c2.old_val) * c1.old_val
        total_dE -= (term1 + term2)
    end

    return total_dE
end
get_binder_coeff(::IsingModel) = 3.0
"""
    measure_magnetization(grids, lat, ::IsingModel)

Calculates the magnetization per site: M = \\frac{1}{N} |\\sum s_i|.
"""
measure_magnetization(grids, lat, ::IsingModel) = abs(sum(grids)) / length(grids)
