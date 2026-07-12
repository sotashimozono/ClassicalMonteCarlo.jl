# N-fold-way / BKL rejection-free kinetic Monte Carlo (Bortz–Kalos–Lebowitz 1975)
# for the Ising model. Ordinary Metropolis wastes proposals as rejections when
# the acceptance is small (low T); the N-fold-way instead groups every spin into
# a small number of CLASSES by its local flip probability and, each step, flips
# exactly one spin — no rejection ever. A spin's Glauber flip rate
#
#   p(s_i, hsum_i) = 1 / (1 + exp(ΔE/kbT)),  ΔE = 2 s_i (J·hsum_i + h),
#
# depends only on its sign s_i and the sum hsum_i = Σ_{j∈nbr} s_j of its
# neighbours, so the class key (s_i, hsum_i) takes just ~2(deg+1) values. Each
# step: pick a class ∝ (#members · p_class), pick a uniform member, flip it, and
# advance simulated time by the mean residence Δt = 1/Q with Q = Σ_classes #·p —
# so equilibrium averages are the residence-time-weighted sums
#
#   ⟨A⟩ = ( Σ_steps A · Δt ) / ( Σ_steps Δt ).
#
# The Glauber rates obey detailed balance w.r.t. the Gibbs weight, so this
# continuous-time chain samples the SAME canonical distribution as Metropolis,
# by a completely different (rejection-free, time-weighted) mechanism.

"""
    NFoldWay(; steps=2_000_000, therm=20_000)

Bortz–Kalos–Lebowitz rejection-free kinetic sampler for the [`IsingModel`](@ref).
`steps` residence-weighted measurement steps follow `therm` thermalisation steps.
See [`nfold_way`](@ref).
"""
@kwdef struct NFoldWay
    steps::Int = 2_000_000
    therm::Int = 20_000
end
export NFoldWay

"""
    nfold_way(rng, grids, lat, model::IsingModel, alg::NFoldWay; kbT)
        -> (; energy, energy2, mag2)

Evolve `grids` by the rejection-free N-fold-way kinetic Ising dynamics and return
the residence-time-weighted canonical averages ⟨E⟩, ⟨E²⟩ (total energy) and
⟨M²⟩ (total magnetisation squared). `grids` is mutated in place.
"""
function nfold_way(
    rng::AbstractRNG,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::IsingModel,
    alg::NFoldWay;
    kbT::Float64,
)
    N = num_sites(lat)
    β = 1.0 / kbT
    J = model.J
    h = model.h
    pflip(s::Int, hsum::Int) = 1.0 / (1.0 + exp(2.0 * s * (J * hsum + h) * β))

    hsum(i::Int) = (t = 0; for j in neighbors(lat, i); t += grids[j]; end; t)

    key_of = Vector{Tuple{Int,Int}}(undef, N)
    pos = Vector{Int}(undef, N)
    members = Dict{Tuple{Int,Int},Vector{Int}}()
    add_site!(i) = begin
        k = (grids[i], hsum(i))
        key_of[i] = k
        v = get!(members, k, Int[])
        push!(v, i)
        pos[i] = length(v)
    end
    remove_site!(i) = begin                    # O(1) swap-remove keeping pos[] valid
        v = members[key_of[i]]
        p = pos[i]
        moved = v[end]
        v[p] = moved
        pos[moved] = p
        pop!(v)
    end
    for i in 1:N
        add_site!(i)
    end

    E = total_energy(grids, lat, model)
    M = sum(grids)
    sumE = 0.0
    sumE2 = 0.0
    sumM2 = 0.0
    sumw = 0.0

    for step in 1:(alg.therm + alg.steps)
        Q = 0.0
        for (k, v) in members
            isempty(v) && continue
            Q += pflip(k[1], k[2]) * length(v)
        end
        target = rand(rng) * Q
        acc = 0.0
        sel = 0
        for (k, v) in members
            isempty(v) && continue
            acc += pflip(k[1], k[2]) * length(v)
            if target <= acc
                sel = v[rand(rng, 1:length(v))]
                break
            end
        end
        sel == 0 && (sel = rand(rng, 1:N))     # floating-point tail guard

        if step > alg.therm                    # measure BEFORE the flip, weight Δt=1/Q
            w = 1.0 / Q
            sumE += E * w
            sumE2 += E * E * w
            sumM2 += (M * M) * w
            sumw += w
        end

        s_old = grids[sel]
        E += 2.0 * s_old * (J * hsum(sel) + h)
        M -= 2 * s_old
        remove_site!(sel)
        for j in neighbors(lat, sel)
            remove_site!(j)
        end
        grids[sel] = -s_old
        add_site!(sel)
        for j in neighbors(lat, sel)
            add_site!(j)
        end
    end
    return (; energy=sumE / sumw, energy2=sumE2 / sumw, mag2=sumM2 / sumw)
end
export nfold_way
