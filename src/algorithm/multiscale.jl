# Multiscale block-spin (multigrid) update. Instead of — or on top of — flipping
# one spin at a time, it flips *whole blocks* of a coarse-graining hierarchy.
# Flipping every spin in a block leaves the block's internal bonds unchanged, so
# the energy change comes only from the bonds crossing the block boundary:
#
#     ΔE = 2J Σ_{⟨ij⟩ ∈ ∂B} s_i s_j  +  2h Σ_{i∈B} s_i,
#
# and a Metropolis accept/reject on ΔE keeps detailed balance for the Ising Gibbs
# weight. Sweeping the block flips across the levels of the hierarchy injects
# large-scale moves that single-spin updates reach only after ~L^z sweeps; the
# top level (a block spanning the sample) is the global Z₂ flip.
#
# The hierarchy is supplied by the caller as, for each level, a partition of the
# sites into blocks — any block-spin coarsening works. `square_block_levels`
# builds the 2^ℓ × 2^ℓ hierarchy for a square lattice.

"""
    MultiscaleBlockFlip(levels; single_spin = true)

Multiscale block-spin update built from a coarse-graining hierarchy. `levels[ℓ]`
partitions the sites into blocks, `levels[ℓ][b]::Vector{Int}` being the site
list of block `b` at level `ℓ` (finest first). One [`update_step!`](@ref) does an
optional single-spin Metropolis sweep (`single_spin`) followed by a block-flip
attempt on every block of every level.

Ferromagnetic or antiferromagnetic Ising; the acceptance uses only the block's
boundary bonds. See [`square_block_levels`](@ref) for the square-lattice
hierarchy.
"""
struct MultiscaleBlockFlip <: UpdateAlgorithm
    blocks::Vector{Vector{Vector{Int}}}
    blockid::Vector{Vector{Int}}
    single_spin::Bool
end
export MultiscaleBlockFlip

function MultiscaleBlockFlip(levels::Vector{Vector{Vector{Int}}}; single_spin::Bool=true)
    isempty(levels) &&
        throw(ArgumentError("MultiscaleBlockFlip needs at least one hierarchy level"))
    N = maximum(i for lvl in levels for blk in lvl for i in blk)
    blockid = Vector{Vector{Int}}(undef, length(levels))
    for (ℓ, lvl) in enumerate(levels)
        ids = zeros(Int, N)
        for (b, blk) in enumerate(lvl), i in blk
            ids[i] = b
        end
        any(iszero, ids) &&
            throw(ArgumentError("level $ℓ block partition does not cover every site"))
        blockid[ℓ] = ids
    end
    return MultiscaleBlockFlip(levels, blockid, single_spin)
end

"""
    square_block_levels(lat::AbstractLattice) -> Vector{Vector{Vector{Int}}}

Build the `2^ℓ × 2^ℓ` block hierarchy of an `L × L` square lattice (`L` a power
of two) from the site positions, ready for [`MultiscaleBlockFlip`](@ref). Level
`ℓ` groups the sites into `(L/2^ℓ)²` square blocks of `2^ℓ × 2^ℓ` sites; the
coarsest level is the single block spanning the sample.
"""
function square_block_levels(lat::AbstractLattice)
    N = num_sites(lat)
    P = [position(lat, i) for i in 1:N]
    x0 = minimum(p -> p[1], P)
    y0 = minimum(p -> p[2], P)
    L = round(Int, maximum(p -> p[1], P) - x0) + 1
    Ly = round(Int, maximum(p -> p[2], P) - y0) + 1
    (L == Ly && ispow2(L)) || throw(
        ArgumentError(
            "square_block_levels expects a square L×L lattice with L a power of two, " *
            "got $(L)×$(Ly)",
        ),
    )
    coord(i) = (round(Int, P[i][1] - x0), round(Int, P[i][2] - y0))   # 0-based
    nlev = trailing_zeros(L)
    levels = Vector{Vector{Vector{Int}}}(undef, nlev)
    for ℓ in 1:nlev
        b = 1 << ℓ
        nb = L ÷ b
        groups = [Int[] for _ in 1:(nb * nb)]
        for i in 1:N
            x, y = coord(i)
            push!(groups[(y ÷ b) * nb + (x ÷ b) + 1], i)
        end
        levels[ℓ] = groups
    end
    return levels
end
export square_block_levels

function update_step!(
    rng::AbstractRNG,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    alg::MultiscaleBlockFlip;
    kbT::Float64,
    kwargs...,
)
    β = 1.0 / kbT
    J = model.J
    h = model.h

    if alg.single_spin
        @inbounds for i in 1:num_sites(lat)
            f = 0
            for j in neighbors(lat, i)
                f += grids[j]
            end
            dE = 2 * grids[i] * (J * f + h)
            (dE <= 0 || rand(rng) < exp(-β * dE)) && (grids[i] = -grids[i])
        end
    end

    @inbounds for ℓ in eachindex(alg.blocks)
        bid = alg.blockid[ℓ]
        for blk in alg.blocks[ℓ]
            id = bid[blk[1]]
            bond = 0
            fsum = 0
            for i in blk
                si = grids[i]
                fsum += si
                for j in neighbors(lat, i)
                    bid[j] != id && (bond += si * grids[j])   # boundary bond, counted once
                end
            end
            dE = 2 * (J * bond + h * fsum)
            if dE <= 0 || rand(rng) < exp(-β * dE)
                for i in blk
                    grids[i] = -grids[i]
                end
            end
        end
    end
    return nothing
end
