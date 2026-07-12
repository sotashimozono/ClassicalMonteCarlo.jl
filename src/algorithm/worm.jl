# Worm algorithm (Prokof'ev–Svistunov, 2001) for the zero-field Ising model in
# the high-temperature / bond (graph) representation. Instead of sampling spin
# configurations, it samples the closed-loop graphs of the high-T expansion
#
#   ⟨σ_x σ_y⟩ = ( Σ_{∂g = {x,y}} t^{|g|} ) / ( Σ_{∂g = ∅} t^{|g|} ),   t = tanh(βJ),
#
# where g is a bond subset and ∂g the set of odd-degree ("defect") sites. A pair
# of worm ends — Ira (head) and Masha (tail) — random-walks through graph space:
# each step flips the bond between Ira and a random neighbour (Metropolis weight
# ratio t^{±1}, with the neighbour-degree ratio for detailed balance on
# irregular lattices) and moves Ira there, so the odd-degree sites stay exactly
# {Ira, Masha}. When Ira returns to Masha the graph is closed (the Z sector).
#
# Because it never has to flip large clusters, the worm has NO critical slowing
# down. The per-spin magnetic susceptibility
#
#   S = Σ_r ⟨σ_0 σ_r⟩ = ⟨M²⟩/N       (zero field, so ⟨M⟩ = 0)
#
# is the ratio estimator S = N_total / N_closed — the average number of worm
# steps between successive closed (Ira == Masha) events.

"""
    WormIsing(; steps=1_000_000, therm=10_000)

Prokof'ev–Svistunov worm sampler for the zero-field [`IsingModel`](@ref) in the
high-temperature bond representation. `steps` measurement steps follow `therm`
thermalisation steps. See [`worm_susceptibility`](@ref).
"""
@kwdef struct WormIsing
    steps::Int = 1_000_000
    therm::Int = 10_000
end
export WormIsing

_bondkey(a::Int, b::Int) = a < b ? (a, b) : (b, a)

"""
    worm_susceptibility(rng, lat, model::IsingModel, alg::WormIsing; kbT) -> S

Estimate the per-spin susceptibility `S = Σ_r ⟨σ_0 σ_r⟩ = ⟨M²⟩/N` of the
zero-field Ising model with the worm ratio estimator `S = N_total / N_closed`.
Requires `model.h == 0` (the high-temperature graph expansion is zero-field).
"""
function worm_susceptibility(
    rng::AbstractRNG,
    lat::AbstractLattice,
    model::IsingModel,
    alg::WormIsing;
    kbT::Float64,
)
    iszero(model.h) ||
        throw(ArgumentError("worm algorithm requires zero field (model.h == 0)"))
    t = tanh(model.J / kbT)
    N = num_sites(lat)
    bonds = Dict{Tuple{Int,Int},Int}()
    ira = rand(rng, 1:N)
    masha = ira
    ntot = 0
    nclosed = 0
    for step in 1:(alg.therm + alg.steps)
        nbrs = neighbors(lat, ira)
        j = rand(rng, nbrs)
        key = _bondkey(ira, j)
        occ = get(bonds, key, 0)
        # target weight ∝ t^{|g|}: adding a bond ×t, removing ÷t; the degree
        # ratio deg(ira)/deg(j) makes the asymmetric proposal detailed-balanced.
        ratio = (occ == 0 ? t : inv(t)) * length(nbrs) / length(neighbors(lat, j))
        if rand(rng) < ratio
            bonds[key] = 1 - occ
            ira = j
        end
        if step > alg.therm
            ntot += 1
            if ira == masha
                nclosed += 1
                s = rand(rng, 1:N)          # relocate the closed worm to decorrelate
                ira = s
                masha = s
            end
        end
    end
    return ntot / nclosed
end
export worm_susceptibility
