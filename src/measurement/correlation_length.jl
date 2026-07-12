# Second-moment correlation length. The structure factor (static, per config)
#
#   S(k) = (1/N) |Σ_j s_j e^{i k·r_j}|²
#
# has S(0) = ⟨M²⟩/N, and its decay to the first non-zero wavevector k_min gives
# the standard finite-lattice second-moment estimator of the correlation length
# (Cooper–Freedman–Preston):
#
#   ξ = (1 / (2 sin(|k_min|/2))) · sqrt( ⟨S(0)⟩/⟨S(k_min)⟩ − 1 ).
#
# On an L×L lattice |k_min| = 2π/L, so the prefactor is 1/(2 sin(π/L)). This ξ is
# the quantity used in finite-size-scaling collapses (ξ/L crossings locate Tc).

# Linear extents (Lx, Ly) of a regular-grid lattice, derived from the abstract
# `position` interface (LatticeCore) so no concrete lattice fields are needed:
# integer coordinates 0…L−1 along each axis give L = max coordinate + 1.
function _grid_extents(lat::AbstractLattice)
    mx = 0
    my = 0
    for i in 1:num_sites(lat)
        r = position(lat, i)
        mx = max(mx, round(Int, r[1]))
        my = max(my, round(Int, r[2]))
    end
    return (mx + 1, my + 1)
end

"""
    structure_factor(config, lat, kx, ky) -> Float64

Static structure factor S(k) = (1/N)|Σ_j s_j e^{i k·r_j}|² of a single scalar
(Ising/Potts-value) configuration at wavevector k = (kx, ky).
"""
function structure_factor(
    config::AbstractVector{<:Real}, lat::AbstractLattice, kx::Real, ky::Real
)
    re = 0.0
    im = 0.0
    for j in 1:num_sites(lat)
        r = position(lat, j)
        ϕ = kx * r[1] + ky * r[2]
        re += config[j] * cos(ϕ)
        im += config[j] * sin(ϕ)
    end
    return (re^2 + im^2) / num_sites(lat)
end
export structure_factor

"""
    second_moment_correlation_length(S0, Skmin, kmin_mag) -> Float64

Second-moment correlation length ξ = sqrt(max(S0/Skmin − 1, 0)) / (2 sin(kmin_mag/2))
from the k=0 and k=k_min structure factors (`kmin_mag` = |k_min|).
"""
function second_moment_correlation_length(S0::Real, Skmin::Real, kmin_mag::Real)
    Skmin > 0 || throw(ArgumentError("S(k_min) must be positive"))
    ratio = S0 / Skmin - 1.0
    return sqrt(max(ratio, 0.0)) / (2.0 * sin(kmin_mag / 2.0))
end
export second_moment_correlation_length

"""
    measure_correlation_length(rng, grids, lat, model, updater; kbT, sweeps,
                               therm=sweeps÷10, interval=1) -> (; xi, S0, Skmin)

Monte-Carlo estimate of the second-moment correlation length: averages S(0) and
S(k_min) (over the two axis-aligned minimal wavevectors) and combines them.
`grids` is mutated.
"""
function measure_correlation_length(
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
    Lx, Ly = _grid_extents(lat)
    kx = 2π / Lx
    ky = 2π / Ly
    S0 = 0.0
    Sk = 0.0
    n = 0
    for s in 1:(therm + sweeps)
        update_step!(rng, grids, lat, model, updater; kbT=kbT)
        if s > therm && (s - therm) % interval == 0
            S0 += structure_factor(grids, lat, 0.0, 0.0)
            Sk +=
                (
                    structure_factor(grids, lat, kx, 0.0) +
                    structure_factor(grids, lat, 0.0, ky)
                ) / 2
            n += 1
        end
    end
    S0 /= n
    Sk /= n
    kmin_mag = 2π / Lx                       # |k_min| (assumes Lx == Ly)
    return (; xi=second_moment_correlation_length(S0, Sk, kmin_mag), S0=S0, Skmin=Sk)
end
export measure_correlation_length
