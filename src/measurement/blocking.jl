# Blocking / binning analysis (Flyvbjerg–Petersen 1989). The naive standard error
# √(var/n) underestimates the error of CORRELATED data. Blocking repeatedly
# replaces the series by the averages of adjacent pairs; each transform halves
# the length and halves the correlation, while the true error of the mean is
# invariant. The naive SE therefore GROWS with the block size and plateaus at the
# correct value √(2τ_int)·σ/√n once the blocks are effectively independent.

"""
    blocking(series) -> Vector{NamedTuple}

Flyvbjerg–Petersen blocking transform. Returns, for each level (block size 2^ℓ),
`(; level, block_size, n_blocks, mean, stderr)` where `stderr` = √(var/n_blocks)
is the naive standard error of the mean of the blocked series.
"""
function blocking(series::AbstractVector{<:Real})
    x = collect(float.(series))
    length(x) >= 2 || throw(ArgumentError("need ≥ 2 samples"))
    levels = NamedTuple[]
    ℓ = 0
    while length(x) >= 2
        n = length(x)
        μ = sum(x) / n
        v = sum(xi -> (xi - μ)^2, x) / (n - 1)
        push!(
            levels, (; level=ℓ, block_size=1 << ℓ, n_blocks=n, mean=μ, stderr=sqrt(v / n))
        )
        m = n ÷ 2
        @inbounds for i in 1:m
            x[i] = (x[2i - 1] + x[2i]) / 2
        end
        resize!(x, m)
        ℓ += 1
    end
    return levels
end
export blocking

"""
    blocking_error(series; min_blocks=64) -> Float64

Plateau standard error from the blocking analysis: the largest naive SE over the
levels that still have at least `min_blocks` blocks (where the estimate is
statistically reliable) — the correlation-corrected error bar of the mean.
"""
function blocking_error(series::AbstractVector{<:Real}; min_blocks::Int=64)
    lv = blocking(series)
    reliable = [l.stderr for l in lv if l.n_blocks >= min_blocks]
    return isempty(reliable) ? lv[1].stderr : maximum(reliable)
end
export blocking_error
