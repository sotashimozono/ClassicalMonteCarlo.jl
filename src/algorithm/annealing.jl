# Simulated annealing (Kirkpatrick–Gelatt–Vecchi 1983). Monte Carlo used as a
# global OPTIMISER rather than a sampler: run local updates while cooling the
# temperature slowly along a geometric schedule T0 → Tf. At high T the chain
# roams freely over barriers; as T → 0 the Boltzmann weight concentrates on the
# lowest-energy states, so the walk settles into a ground state. The best
# (lowest-energy) configuration visited along the schedule is returned.

"""
    SimulatedAnnealing(; updater=LocalUpdate(), T0=5.0, Tf=0.05, steps=200,
                       sweeps_per_step=20)

Geometric-schedule annealing: `steps` temperature stages from `T0` down to `Tf`,
`sweeps_per_step` sweeps of `updater` at each stage. See [`simulated_anneal`](@ref).
"""
@kwdef struct SimulatedAnnealing{U<:UpdateAlgorithm}
    updater::U = LocalUpdate()
    T0::Float64 = 5.0
    Tf::Float64 = 0.05
    steps::Int = 200
    sweeps_per_step::Int = 20
end
export SimulatedAnnealing

"""
    simulated_anneal(rng, grids, lat, model, alg::SimulatedAnnealing)
        -> (; energy, config)

Anneal `grids` (mutated) along the geometric schedule and return the lowest total
energy visited together with the configuration achieving it.
"""
function simulated_anneal(
    rng::AbstractRNG,
    grids::AbstractVector,
    lat::AbstractLattice,
    model::AbstractModel,
    alg::SimulatedAnnealing,
)
    alg.steps >= 2 || throw(ArgumentError("steps must be ≥ 2"))
    alg.T0 > alg.Tf > 0 || throw(ArgumentError("require T0 > Tf > 0"))
    ratio = (alg.Tf / alg.T0)^(1 / (alg.steps - 1))
    T = alg.T0
    Ebest = total_energy(grids, lat, model)
    best = copy(grids)
    for _ in 1:(alg.steps)
        for _ in 1:(alg.sweeps_per_step)
            update_step!(rng, grids, lat, model, alg.updater; kbT=T)
        end
        E = total_energy(grids, lat, model)
        if E < Ebest
            Ebest = E
            best .= grids
        end
        T *= ratio
    end
    return (; energy=Ebest, config=best)
end
export simulated_anneal
