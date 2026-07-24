"""
    measure_energy(grids, lat, model) -> Float64

Calculates the total energy of the system.
This is a wrapper around [`total_energy`](@ref) to provide a consistent interface for observers.
"""
measure_energy(grids, lat, model) = total_energy(grids, lat, model)
"""
    measure_magnetization(grids, lat, model) -> Float64

Calculates the magnetization (order parameter) of the system.
This function must be implemented for each specific `model` type.

# Returns
- The magnetization per site (usually normalized between 0 and 1).

# Throws
- `ErrorException`: If not implemented for the specific model type.
"""
function measure_magnetization(grids, lat, model)
    return error("measure_magnetization not implemented for (typeof(model))")
end
"""
    FunctionObserver(name::String, func::Function; interval::Int=100)

A generic observer that tracks a scalar value returned by a custom function `func`.

# Fields
- `name::String`: The name of the observable (used as a column name in DataFrames).
- `measure_func::Function`: A function with signature `f(grids, lat, model)` returning a `Float64`.
- `interval::Int`: The frequency of observation (in Monte Carlo steps).
- `steps::Vector{Int}`: History of time steps where measurements were taken.
- `history::Vector{Float64}`: History of measured values.
"""
mutable struct FunctionObserver <: AbstractObserver
    name::String
    measure_func::Function
    interval::Int
    steps::Vector{Int}
    history::Vector{Float64}
end
export FunctionObserver, measure_magnetization, measure_energy

function FunctionObserver(name::String, func::Function; interval::Int=100)
    return FunctionObserver(name, func, interval, Int[], Float64[])
end
"""
    observe!(obs, grids, lat, model, step; kbT=1.0, kwargs...)

Records the state of the system into the observer.

- For `FunctionObserver`: Executes the user-defined measurement function.
- For `ThermodynamicObserver`: Accumulates Energy (E, E^2) and Magnetization (M, M^2, M^4) statistics for post-processing.

The signature matches the call convention used inside [`run!`](@ref):
five positional arguments `(obs, grids, lat, model, step)` plus `kbT` and any
other algorithm keyword arguments. `kbT` is accepted-and-ignored here (it is
only consumed later in [`get_thermodynamics`](@ref)).
"""
function observe!(
    obs::FunctionObserver,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    step::Int;
    kbT::Float64=1.0,
    kwargs...,
) where {T}
    val = obs.measure_func(grids, lat, model)
    push!(obs.steps, step)
    push!(obs.history, val)
    return nothing
end
"""
    to_dataframe(obs::FunctionObserver) -> DataFrame

Converts the recorded history of a `FunctionObserver` into a DataFrame.
Returns columns `:Step` and `Symbol(obs.name)`.

Requires `DataFrames` to be loaded; the method is provided by the
`ClassicalMonteCarloDataFramesExt` package extension.
"""
function to_dataframe end
"""
    ThermodynamicObserver(; interval::Int=10)

An observer designed to accumulate statistics required to calculate thermodynamic quantities
such as Specific Heat and Susceptibility.

# accumulated Statistics
- `sum_E`, `sum_E2`: Sum of Energy and Energy squared.
- `sum_M`, `sum_M2`, `sum_M4`: Sum of Magnetization moments.
- `n_samples`: Total number of samples taken.
"""
mutable struct ThermodynamicObserver <: AbstractObserver
    interval::Int
    n_samples::Int
    sum_E::Float64
    sum_E2::Float64
    sum_M::Float64
    sum_M2::Float64
    sum_M4::Float64
end

function ThermodynamicObserver(; interval::Int=10)
    return ThermodynamicObserver(interval, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
end
function observe!(
    obs::ThermodynamicObserver,
    grids::AbstractVector{T},
    lat::AbstractLattice,
    model::AbstractModel{T},
    step::Int;
    kbT::Float64=1.0,
    kwargs...,
) where {T}
    E = measure_energy(grids, lat, model)
    M = measure_magnetization(grids, lat, model)

    obs.n_samples += 1
    obs.sum_E += E
    obs.sum_E2 += E^2
    obs.sum_M += M

    M2 = M^2
    obs.sum_M2 += M2
    return obs.sum_M4 += M2^2 # M^4
end
"""
    get_thermodynamics(obs::ThermodynamicObserver, kbT::Float64, N::Int, model::AbstractModel) -> Dict

Computes physical quantities using the accumulated statistics in the observer.

# Returns
A Dictionary containing:
- `"Energy"`: Internal energy density (E/N).
- `"Magnetization"`: Magnetization density (M).
- `"SpecificHeat"`: Heat capacity per site C_v = \\frac{\\text{var}(E)}{k_B T^2 N}.
- `"Susceptibility"`: Magnetic susceptibility \\chi = \\frac{\\text{var}(M)}{k_B T} N.
- `"BinderParam"`: Binder cumulant U_4 = 1 - \\frac{\\langle M^4 \\rangle}{3 \\langle M^2 \\rangle^2}.
- `"Samples"`: Number of samples used for the calculation.
"""
function get_thermodynamics(
    obs::ThermodynamicObserver, kbT::Float64, N::Int, model::AbstractModel
)
    if obs.n_samples == 0
        return Dict{String,Float64}()
    end
    mean_E = obs.sum_E / obs.n_samples
    mean_E2 = obs.sum_E2 / obs.n_samples
    mean_M = obs.sum_M / obs.n_samples
    mean_M2 = obs.sum_M2 / obs.n_samples
    mean_M4 = obs.sum_M4 / obs.n_samples

    var_E = mean_E2 - mean_E^2
    Cv = var_E / (kbT^2) / N

    var_M = mean_M2 - mean_M^2
    Chi = var_M / kbT * N

    bindercoeff = get_binder_coeff(model)
    denominator = bindercoeff * (mean_M2^2)
    U4 = (denominator == 0.0) ? 0.0 : (1.0 - mean_M4 / denominator)

    return Dict(
        "Energy" => mean_E / N,
        "Magnetization" => mean_M,
        "SpecificHeat" => Cv,
        "Susceptibility" => Chi,
        "BinderParam" => U4,
        "Samples" => Float64(obs.n_samples),
    )
end
export ThermodynamicObserver, observe!, get_thermodynamics, to_dataframe
