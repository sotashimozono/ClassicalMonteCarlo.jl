# Multi-histogram reweighting — WHAM (Weighted Histogram Analysis Method,
# Ferrenberg–Swendsen 1989 / Kumar et al. 1992). Single-histogram reweighting is
# reliable only in a narrow window; WHAM optimally COMBINES energy histograms
# from several runs at different temperatures into one self-consistent estimate
# of the density of states g(E), valid across the union of their windows. Given
# runs r=1…R at inverse temperatures β_r with n_r samples and histograms H_r(E),
#
#   g(E) = ( Σ_r H_r(E) ) / ( Σ_r n_r exp(f_r − β_r E) ),
#   exp(−f_r) = Σ_E g(E) exp(−β_r E)          (dimensionless free energies f_r),
#
# solved by iterating the two equations to a fixed point (gauge-fixed f_1 = 0).
# Any canonical average then follows at ARBITRARY β from the single recovered
# g(E):  ⟨A⟩_β = Σ_E A(E) g(E) e^{−βE} / Σ_E g(E) e^{−βE}. All sums are done in
# log space with a log-sum-exp shift.

# `_logsumexp` is shared with the Wang–Landau module (defined in wang-landau.jl).

"""
    WHAM(; energy_quantum=1.0, tol=1e-10, maxiter=10_000)

Parameters for the [`wham`](@ref) multi-histogram solver: energies are binned at
resolution `energy_quantum`; the self-consistent iteration stops when the largest
free-energy change falls below `tol` (or after `maxiter` sweeps).
"""
@kwdef struct WHAM
    energy_quantum::Float64 = 1.0
    tol::Float64 = 1e-10
    maxiter::Int = 10_000
end
export WHAM

"""
    wham(energy_series, kbTs, alg::WHAM=WHAM()) -> (; energies, logg, counts, f, iters)

Solve the WHAM equations for energy samples `energy_series[r]` taken at
temperatures `kbTs[r]`. Returns the sorted bin `energies`, the log density of
states `logg` (defined up to an additive constant), the combined per-bin sample
`counts`, the free energies `f`, and the iteration count.
"""
function wham(
    energy_series::AbstractVector{<:AbstractVector{<:Real}},
    kbTs::AbstractVector{<:Real},
    alg::WHAM=WHAM(),
)
    R = length(kbTs)
    R == length(energy_series) ||
        throw(ArgumentError("energy_series and kbTs must have equal length"))
    β = [1.0 / T for T in kbTs]
    q = alg.energy_quantum
    ekey(E) = round(Int, E / q)

    Hr = [Dict{Int,Int}() for _ in 1:R]
    Htot = Dict{Int,Int}()
    for r in 1:R, E in energy_series[r]
        k = ekey(E)
        Hr[r][k] = get(Hr[r], k, 0) + 1
        Htot[k] = get(Htot, k, 0) + 1
    end
    isempty(Htot) && throw(ArgumentError("no samples provided"))

    keys_sorted = sort!(collect(keys(Htot)))
    Evals = [k * q for k in keys_sorted]
    counts = [Htot[k] for k in keys_sorted]
    logHtot = [log(Htot[k]) for k in keys_sorted]
    logn = [log(length(energy_series[r])) for r in 1:R]

    f = zeros(R)
    logg = zeros(length(keys_sorted))
    iters = 0
    for it in 1:(alg.maxiter)
        iters = it
        for i in eachindex(keys_sorted)
            E = Evals[i]
            logden = _logsumexp([logn[r] + f[r] - β[r] * E for r in 1:R])
            logg[i] = logHtot[i] - logden
        end
        newf = [-_logsumexp([logg[i] - β[r] * Evals[i] for i in eachindex(Evals)]) for r in 1:R]
        newf .-= newf[1]                              # gauge fix f_1 = 0
        Δ = maximum(abs.(newf .- f))
        f = newf
        Δ < alg.tol && break
    end
    return (; energies=Evals, logg=logg, counts=counts, f=f, iters=iters)
end
export wham

"""
    wham_mean(result, kbT, f=identity) -> Float64

⟨f(E)⟩ at temperature `kbT` from a recovered density of states `result` (a
[`wham`](@ref) return value).
"""
function wham_mean(result, kbT::Real, f=identity)
    β = 1.0 / kbT
    E = result.energies
    xs = [result.logg[i] - β * E[i] for i in eachindex(E)]
    m = maximum(xs)
    w = exp.(xs .- m)
    return sum(f.(E) .* w) / sum(w)
end
export wham_mean
