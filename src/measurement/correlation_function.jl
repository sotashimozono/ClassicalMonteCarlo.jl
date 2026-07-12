# Real-space spin–spin correlation function C(r) = ⟨s_0 s_r⟩ measured along the
# lattice axes, and the correlation length extracted from its exponential decay
# C(r) ~ e^{−r/ξ}. This is the direct real-space counterpart of the reciprocal-
# space second-moment estimator (see correlation_length.jl); the two ξ's agree
# for a well-behaved correlator. The Wiener–Khinchin sum rule Σ_r C(r) = S(0) =
# ⟨M²⟩/N ties C(r) back to the structure factor.

"""
    spin_correlation_function(rng, grids, lat, model, updater; kbT, sweeps,
                              therm=sweeps÷10, interval=1, rmax=Lx÷2)
        -> (; r, C)

Monte-Carlo estimate of C(r) = ⟨s_i s_{i+r}⟩ for r = 0…rmax, averaged over both
axes and all origins. `grids` is mutated. (Scalar Ising/Potts-value spins.)
"""
function spin_correlation_function(
    rng::AbstractRNG,
    grids::AbstractVector{<:Real},
    lat::AbstractLattice,
    model::AbstractModel,
    updater::UpdateAlgorithm;
    kbT::Float64,
    sweeps::Int,
    therm::Int=sweeps ÷ 10,
    interval::Int=1,
    rmax::Int=_grid_extents(lat)[1] ÷ 2,
)
    N = num_sites(lat)
    Lx, Ly = _grid_extents(lat)
    coord = [
        (round(Int, position(lat, i)[1]), round(Int, position(lat, i)[2])) for i in 1:N
    ]
    idx = Dict(coord[i] => i for i in 1:N)
    xshift = [idx[(mod(coord[i][1] + r, Lx), coord[i][2])] for i in 1:N, r in 0:rmax]
    yshift = [idx[(coord[i][1], mod(coord[i][2] + r, Ly))] for i in 1:N, r in 0:rmax]

    acc = zeros(Float64, rmax + 1)
    nsamp = 0
    for s in 1:(therm + sweeps)
        update_step!(rng, grids, lat, model, updater; kbT=kbT)
        if s > therm && (s - therm) % interval == 0
            @inbounds for r in 0:rmax
                c = 0.0
                for i in 1:N
                    c +=
                        grids[i] * grids[xshift[i, r + 1]] +
                        grids[i] * grids[yshift[i, r + 1]]
                end
                acc[r + 1] += c / (2N)
            end
            nsamp += 1
        end
    end
    return (; r=collect(0:rmax), C=acc ./ nsamp)
end
export spin_correlation_function

"""
    correlation_length_from_decay(rs, Cr) -> Float64

Correlation length ξ from a log-linear least-squares fit of C(r) ~ e^{−r/ξ} over
the given lags `rs` (only strictly-positive `Cr` entries are used); ξ = −1/slope.
"""
function correlation_length_from_decay(
    rs::AbstractVector{<:Real}, Cr::AbstractVector{<:Real}
)
    x = Float64[]
    y = Float64[]
    for (r, c) in zip(rs, Cr)
        c > 0 || continue
        push!(x, r)
        push!(y, log(c))
    end
    length(x) >= 2 || throw(ArgumentError("need ≥ 2 positive C(r) values to fit"))
    x̄ = sum(x) / length(x)
    ȳ = sum(y) / length(y)
    slope = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    slope < 0 || throw(ArgumentError("C(r) is not decaying (non-negative slope)"))
    return -1.0 / slope
end
export correlation_length_from_decay
