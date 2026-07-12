# Wolff single-cluster algorithm (Wolff 1989). The canonical cure for critical
# slowing down: instead of flipping one spin at a time, it grows a cluster of
# aligned spins and flips it whole. From a random seed, an aligned neighbour is
# added to the cluster with the bond probability p = 1 − e^{−2βJ}; the completed
# cluster is flipped. This satisfies detailed balance for the Ising Gibbs weight
# (the Fortuin–Kasteleyn construction) while decorrelating the configuration in
# O(1) cluster moves near Tc, where local updates need ~L^z sweeps.
#
# Ferromagnetic (J > 0) Ising only — the cluster bond probability assumes aligned
# spins lower the energy.

"""
    Wolff()

Wolff single-cluster update for the ferromagnetic [`IsingModel`](@ref). One
[`update_step!`](@ref) grows and flips a single cluster.
"""
struct Wolff <: UpdateAlgorithm end
export Wolff

function update_step!(
    rng::AbstractRNG,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    ::Wolff;
    kbT::Float64,
    kwargs...,
)
    model.J > 0 ||
        throw(ArgumentError("Wolff cluster update requires a ferromagnet (J > 0)"))
    p_add = 1.0 - exp(-2.0 * model.J / kbT)
    seed = rand(rng, 1:num_sites(lat))
    s0 = grids[seed]
    grids[seed] = -s0                              # flip the seed on inclusion
    stack = [seed]
    while !isempty(stack)
        i = pop!(stack)
        for j in neighbors(lat, i)
            if grids[j] == s0 && rand(rng) < p_add # aligned with the seed's original spin
                grids[j] = -s0
                push!(stack, j)
            end
        end
    end
    return nothing
end
