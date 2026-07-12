# Swendsen–Wang multi-cluster algorithm (Swendsen–Wang 1987). The whole lattice
# is decomposed into Fortuin–Kasteleyn clusters at once — every bond between
# aligned spins is activated with p = 1 − e^{−2βJ}, the connected components are
# identified, and EACH cluster is flipped independently with probability ½.
# Like Wolff it beats critical slowing down; unlike Wolff it updates all spins per
# sweep (one whole partition) rather than a single cluster.
#
# The clusters are grown one-at-a-time to completion, so every aligned bond is
# tested exactly once with probability p — reproducing the FK bond ensemble
# exactly. Ferromagnetic (J > 0) Ising only.

"""
    SwendsenWang()

Swendsen–Wang multi-cluster update for the ferromagnetic [`IsingModel`](@ref).
One [`update_step!`](@ref) rebuilds the full Fortuin–Kasteleyn cluster partition
and flips each cluster with probability ½.
"""
struct SwendsenWang <: UpdateAlgorithm end
export SwendsenWang

function update_step!(
    rng::AbstractRNG,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    ::SwendsenWang;
    kbT::Float64,
    kwargs...,
)
    model.J > 0 ||
        throw(ArgumentError("Swendsen–Wang cluster update requires a ferromagnet (J > 0)"))
    N = num_sites(lat)
    p_add = 1.0 - exp(-2.0 * model.J / kbT)
    label = zeros(Int, N)
    nlab = 0
    stack = Int[]
    for start in 1:N
        label[start] == 0 || continue
        nlab += 1
        label[start] = nlab
        push!(stack, start)
        while !isempty(stack)
            i = pop!(stack)
            for j in neighbors(lat, i)
                if label[j] == 0 && grids[j] == grids[i] && rand(rng) < p_add
                    label[j] = nlab
                    push!(stack, j)
                end
            end
        end
    end
    flip = [rand(rng) < 0.5 for _ in 1:nlab]           # each cluster flips with prob ½
    @inbounds for i in 1:N
        flip[label[i]] && (grids[i] = -grids[i])
    end
    return nothing
end
