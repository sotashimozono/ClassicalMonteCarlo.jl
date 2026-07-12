# Integrated autocorrelation time. Successive Monte-Carlo samples are correlated,
# so the true statistical error of an average is √(2τ_int/n)·σ, not √(1/n)·σ. For
# a stationary series with normalised autocorrelation ρ(t) = C(t)/C(0),
#
#   τ_int = 1/2 + Σ_{t≥1} ρ(t),
#
# truncated by the automatic self-consistent window W (Sokal/Madras): W is the
# smallest lag with W ≥ c·τ_int(W) (c≈6), which balances bias and variance. An
# uncorrelated series gives τ_int = 1/2; τ_int diverges at a critical point
# (critical slowing down), which is exactly what cluster algorithms cure.

"""
    autocorrelation(series, tmax) -> Vector{Float64}

Normalised autocorrelation ρ(t) = C(t)/C(0) for lags t = 0…tmax (so the result
has length `tmax+1` with `ρ[1] = 1`).
"""
function autocorrelation(series::AbstractVector{<:Real}, tmax::Int)
    n = length(series)
    tmax < n || throw(ArgumentError("tmax must be < length(series)"))
    μ = sum(series) / n
    c0 = sum(x -> (x - μ)^2, series) / n
    ρ = ones(Float64, tmax + 1)
    c0 == 0 && return ρ
    for t in 1:tmax
        s = 0.0
        @inbounds for i in 1:(n - t)
            s += (series[i] - μ) * (series[i + t] - μ)
        end
        ρ[t + 1] = (s / (n - t)) / c0
    end
    return ρ
end
export autocorrelation

"""
    integrated_autocorrelation_time(series; c=6.0, tmax=min(length(series)÷2, 2000))
        -> (; tau, window)

Integrated autocorrelation time τ_int = 1/2 + Σ_{t=1}^{W} ρ(t) with the automatic
window W (smallest lag with W ≥ c·τ_int). Returns τ and the chosen window.
"""
function integrated_autocorrelation_time(
    series::AbstractVector{<:Real};
    c::Float64=6.0,
    tmax::Int=min(length(series) ÷ 2, 2000),
)
    n = length(series)
    μ = sum(series) / n
    c0 = sum(x -> (x - μ)^2, series) / n
    c0 == 0 && return (; tau=0.5, window=0)
    τ = 0.5
    W = tmax
    for t in 1:tmax
        s = 0.0
        @inbounds for i in 1:(n - t)
            s += (series[i] - μ) * (series[i + t] - μ)
        end
        τ += (s / (n - t)) / c0
        if t >= c * τ
            W = t
            break
        end
    end
    return (; tau=τ, window=W)
end
export integrated_autocorrelation_time
