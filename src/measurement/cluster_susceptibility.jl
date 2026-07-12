# Improved (cluster) susceptibility estimators. In the Fortuin–Kasteleyn
# representation ⟨s_i s_j⟩ = P(i ↔ j in the same cluster), so
#
#   ⟨M²⟩ = Σ_{i,j} ⟨s_i s_j⟩ = ⟨ Σ_clusters |C|² ⟩,
#
# i.e. Σ_c |C|² is the conditional expectation E[M² | cluster config]. By
# Rao–Blackwell it has variance ≤ that of the bare spin estimator M², so cluster
# algorithms measure the susceptibility χ = ⟨M²⟩/(kbT·N) with smaller error:
#   • Swendsen–Wang:  χ = ⟨ Σ_c |C|² ⟩ / (kbT·N),
#   • Wolff:          χ = ⟨|C|⟩ / kbT       (⟨|C|⟩ = ⟨Σ_c|C|²⟩/N, since Σ_c|C| = N).

"""
    wolff_susceptibility(rng, grids, lat, model::IsingModel; kbT, sweeps,
                         therm=sweeps÷10) -> (; chi, mean_cluster_size)

Improved susceptibility χ = ⟨|C|⟩/kbT from the mean Wolff single-cluster size.
`grids` is evolved by the Wolff dynamics. Ferromagnet (J>0) only.
"""
function wolff_susceptibility(
    rng::AbstractRNG, grids::AbstractVector{Int}, lat::AbstractLattice,
    model::IsingModel; kbT::Float64, sweeps::Int, therm::Int=sweeps ÷ 10,
)
    model.J > 0 || throw(ArgumentError("Wolff requires a ferromagnet (J > 0)"))
    N = num_sites(lat)
    p = 1.0 - exp(-2.0 * model.J / kbT)
    tot = 0.0
    n = 0
    stack = Int[]
    for s in 1:(therm + sweeps)
        seed = rand(rng, 1:N)
        s0 = grids[seed]
        grids[seed] = -s0
        empty!(stack)
        push!(stack, seed)
        sz = 1
        while !isempty(stack)
            i = pop!(stack)
            for j in neighbors(lat, i)
                if grids[j] == s0 && rand(rng) < p
                    grids[j] = -s0
                    push!(stack, j)
                    sz += 1
                end
            end
        end
        if s > therm
            tot += sz
            n += 1
        end
    end
    mcs = tot / n
    return (; chi=mcs / kbT, mean_cluster_size=mcs)
end
export wolff_susceptibility

"""
    swendsen_wang_susceptibility(rng, grids, lat, model::IsingModel; kbT, sweeps,
                                 therm=sweeps÷10) -> (; chi)

Improved susceptibility χ = ⟨Σ_c |C|²⟩/(kbT·N) from the Swendsen–Wang cluster
decomposition. `grids` is evolved by the SW dynamics. Ferromagnet (J>0) only.
"""
function swendsen_wang_susceptibility(
    rng::AbstractRNG, grids::AbstractVector{Int}, lat::AbstractLattice,
    model::IsingModel; kbT::Float64, sweeps::Int, therm::Int=sweeps ÷ 10,
)
    model.J > 0 || throw(ArgumentError("Swendsen–Wang requires a ferromagnet (J > 0)"))
    N = num_sites(lat)
    p = 1.0 - exp(-2.0 * model.J / kbT)
    label = zeros(Int, N)
    stack = Int[]
    tot = 0.0
    n = 0
    for s in 1:(therm + sweeps)
        fill!(label, 0)
        nlab = 0
        sizes = Int[]
        for start in 1:N
            label[start] == 0 || continue
            nlab += 1
            label[start] = nlab
            push!(stack, start)
            sz = 1
            while !isempty(stack)
                i = pop!(stack)
                for j in neighbors(lat, i)
                    if label[j] == 0 && grids[j] == grids[i] && rand(rng) < p
                        label[j] = nlab
                        push!(stack, j)
                        sz += 1
                    end
                end
            end
            push!(sizes, sz)
        end
        s > therm && (tot += sum(abs2, sizes); n += 1)
        flip = [rand(rng) < 0.5 for _ in 1:nlab]        # evolve: flip each cluster w.p. ½
        @inbounds for i in 1:N
            flip[label[i]] && (grids[i] = -grids[i])
        end
    end
    return (; chi=(tot / n) / (kbT * N))
end
export swendsen_wang_susceptibility
