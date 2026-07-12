# Replica exchange / parallel tempering. R replicas of the system are simulated
# at an ascending temperature ladder T_1 < … < T_R; each does ordinary local MC
# at its own temperature, and periodically adjacent replicas attempt to swap
# their whole configurations with the Metropolis acceptance
#
#   p_swap = min(1, exp[(β_i − β_j)(E_i − E_j)]),   β = 1/kbT,
#
# which leaves the joint product distribution ∏_k exp(−β_k E_k) invariant. Low-T
# replicas thus borrow the high-T replicas' mobility to escape metastable basins,
# curing the ergodicity problems of a single low-T chain — while every replica
# still samples its own canonical ensemble exactly. Swaps use the even/odd
# (non-overlapping adjacent pairs, alternating offset) schedule.

"""
    ReplicaExchange(; temperatures, updater=LocalUpdate(), sweeps=20_000,
                    therm=2_000, swap_interval=1)

Parallel-tempering driver over an ascending `temperatures` ladder. Each replica
is advanced by `updater` at its own temperature; adjacent replicas attempt a
configuration swap every `swap_interval` sweeps. See [`replica_exchange`](@ref).
"""
@kwdef struct ReplicaExchange{U<:UpdateAlgorithm}
    temperatures::Vector{Float64}
    updater::U = LocalUpdate()
    sweeps::Int = 20_000
    therm::Int = 2_000
    swap_interval::Int = 1
end
export ReplicaExchange

"""
    replica_exchange(rng, grids, lat, model, alg::ReplicaExchange)
        -> (; temperatures, energy, energy2, swap_acceptance)

Run parallel tempering starting every replica from a copy of `grids`. Returns the
per-temperature canonical averages ⟨E⟩, ⟨E²⟩ and the adjacent-pair swap
acceptance ratios. `grids` itself is not modified.
"""
function replica_exchange(
    rng::AbstractRNG,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    alg::ReplicaExchange,
) where {T}
    Ts = alg.temperatures
    R = length(Ts)
    issorted(Ts) || throw(ArgumentError("temperatures must be sorted ascending"))
    R >= 2 || throw(ArgumentError("replica exchange needs at least 2 temperatures"))

    configs = [copy(grids) for _ in 1:R]
    E = [total_energy(configs[k], lat, model) for k in 1:R]
    sumE = zeros(R)
    sumE2 = zeros(R)
    nacc = zeros(Int, R - 1)
    natt = zeros(Int, R - 1)
    count = 0

    for sweep in 1:(alg.therm + alg.sweeps)
        for k in 1:R
            update_step!(rng, configs[k], lat, model, alg.updater; kbT=Ts[k])
            E[k] = total_energy(configs[k], lat, model)
        end

        if sweep % alg.swap_interval == 0
            round = sweep ÷ alg.swap_interval
            offset = isodd(round) ? 1 : 2               # even/odd non-overlapping pairs
            for k in offset:2:(R - 1)
                Δ = (1.0 / Ts[k] - 1.0 / Ts[k + 1]) * (E[k] - E[k + 1])
                natt[k] += 1
                if Δ >= 0.0 || rand(rng) < exp(Δ)
                    configs[k], configs[k + 1] = configs[k + 1], configs[k]
                    E[k], E[k + 1] = E[k + 1], E[k]
                    nacc[k] += 1
                end
            end
        end

        if sweep > alg.therm
            count += 1
            @inbounds for k in 1:R
                sumE[k] += E[k]
                sumE2[k] += E[k]^2
            end
        end
    end

    return (;
        temperatures=Ts,
        energy=sumE ./ count,
        energy2=sumE2 ./ count,
        swap_acceptance=[natt[k] == 0 ? 0.0 : nacc[k] / natt[k] for k in 1:(R - 1)],
    )
end
export replica_exchange
