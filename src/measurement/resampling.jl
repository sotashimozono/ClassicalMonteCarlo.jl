# Resampling error bars — jackknife and bootstrap. The naive standard error σ/√n
# is only valid for the mean of independent samples; Monte-Carlo observables of
# interest (Binder cumulant, susceptibility, any ratio g(⟨a⟩,⟨b⟩)) are NONLINEAR
# functions of averages, whose error must be propagated. Jackknife (leave-one-out)
# and bootstrap (resample-with-replacement) do this without an analytic error
# formula, and reduce to σ/√n for the linear case.

"""
    jackknife(f, data) -> (; value, error, bias_corrected)

Delete-1 jackknife for an estimator `f(means)` that is a function of the
per-observable sample means. `data` is a vector of equal-length sample vectors
(one per observable); a single vector is treated as one observable. Returns the
full-sample estimate, its jackknife standard error, and the bias-corrected value.
"""
function jackknife(f, data::AbstractVector{<:AbstractVector{<:Real}})
    n = length(first(data))
    all(length(d) == n for d in data) ||
        throw(ArgumentError("sample vectors must be equal length"))
    n >= 2 || throw(ArgumentError("need ≥ 2 samples"))
    totals = [sum(d) for d in data]
    full = f([t / n for t in totals])
    jack = Vector{Float64}(undef, n)
    means_i = Vector{Float64}(undef, length(data))
    for i in 1:n
        @inbounds for k in eachindex(data)
            means_i[k] = (totals[k] - data[k][i]) / (n - 1)
        end
        jack[i] = f(means_i)
    end
    θbar = sum(jack) / n
    err = sqrt((n - 1) / n * sum(x -> (x - θbar)^2, jack))
    return (; value=full, error=err, bias_corrected=n * full - (n - 1) * θbar)
end
jackknife(f, data::AbstractVector{<:Real}) = jackknife(f, [data])
export jackknife

"""
    bootstrap(rng, f, data; n_resample=1000) -> (; value, error, samples)

Bootstrap error for `f(means)`: `n_resample` resamples with replacement, the
standard deviation of the resampled estimates being the error. `data` as in
[`jackknife`](@ref).
"""
function bootstrap(
    rng::AbstractRNG,
    f,
    data::AbstractVector{<:AbstractVector{<:Real}};
    n_resample::Int=1000,
)
    n = length(first(data))
    all(length(d) == n for d in data) ||
        throw(ArgumentError("sample vectors must be equal length"))
    K = length(data)
    full = f([sum(d) / n for d in data])
    ests = Vector{Float64}(undef, n_resample)
    means = Vector{Float64}(undef, K)
    for b in 1:n_resample
        fill!(means, 0.0)
        for _ in 1:n
            j = rand(rng, 1:n)
            @inbounds for k in 1:K
                means[k] += data[k][j]
            end
        end
        means ./= n
        ests[b] = f(copy(means))
    end
    θbar = sum(ests) / n_resample
    return (;
        value=full,
        error=sqrt(sum(x -> (x - θbar)^2, ests) / (n_resample - 1)),
        samples=ests,
    )
end
function bootstrap(rng::AbstractRNG, f, data::AbstractVector{<:Real}; kwargs...)
    return bootstrap(rng, f, [data]; kwargs...)
end
export bootstrap
