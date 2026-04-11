"""
    XYModel(J::Float64=1.0)

Represents the classical XY model with continuous planar spins \\theta_i \\in [0, 2\\pi).

# Hamiltonian
H = -J \\sum_{\\langle i, j \\rangle} \\cos(\\theta_i - \theta_j)
"""
@kwdef struct XYModel <: AbstractModel{Float64}
    J::Float64 = 1.0
end
export XYModel

function local_hamiltonian(
    grids::AbstractVector{Float64},
    lat::AbstractLattice,
    model::XYModel,
    site::Int;
    val::Float64=grids[site],
    kwargs...,
)
    energy = 0.0
    for neighbor in neighbors(lat, site)
        energy -= model.J * cos(val - grids[neighbor])
    end
    return energy
end
"""
    propose(rng, alg::UniformShift, ..., model::XYModel, site)

Proposes a change to the angle \\theta by shifting it by a random amount \\delta.
\\theta_{new} = \\theta_{old} + \\delta, where \\delta \\in [-\\text{width}/2, \\text{width}/2].
"""
function propose(
    rng::AbstractRNG,
    alg::UniformShift,
    grids::AbstractVector{Float64},
    lat::AbstractLattice,
    model::XYModel,
    site::Int;
    kwargs...,
)
    current = grids[site]
    delta = (rand(rng) - 0.5) * alg.width
    new_val = current + delta
    return (LocalChange(site, new_val, current),)
end
"""
    get_binder_coeff(model) -> Float64

Returns the coefficient used in the denominator of the Binder Cumulant calculation (3 \\langle M^2 \\rangle^2).

- `IsingModel`: 3.0 (Scalar order parameter, Gaussian universality).
- `PottsModel`: 3.0.
- `XYModel`: 2.0 (Vector order parameter with 2 components).
"""
get_binder_coeff(::XYModel) = 2.0
"""
    measure_magnetization(grids, lat, ::XYModel)

Calculates the magnetization magnitude for the XY model.
It treats the spins as unit vectors \\vec{S}_i = (\\cos \\theta_i, \\sin \\theta_i).

# Formula
M = \\frac{1}{N} \\left| \\sum_i \\vec{S}_i \\right|
"""
function measure_magnetization(grids, lat, ::XYModel)
    M_x = sum(cos, grids)
    M_y = sum(sin, grids)
    return sqrt(M_x^2 + M_y^2) / length(grids)
end
