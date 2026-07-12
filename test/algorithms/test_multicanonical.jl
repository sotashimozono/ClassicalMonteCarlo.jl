using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Multicanonical (Berg–Neuhaus). With the fixed weight S(E)=ln g(E) from a prior
# Wang–Landau run, the production run (1) makes the energy histogram FLAT (its
# defining property) and (2) reweighting e^{−βE+S(E)} recovers canonical ⟨E⟩(T)
# across a WIDE temperature range from a SINGLE run — both checked against exact
# 2^N enumeration.
@testset "Multicanonical — flat histogram + canonical ⟨E⟩(T) vs exact" begin
    rng = MersenneTwister(88)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)

    Ω = Dict{Int,Int}()
    let g = ones(Int, N)
        for c in 0:(2 ^ N - 1)
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            k = round(Int, total_energy(g, lat, model))
            Ω[k] = get(Ω, k, 0) + 1
        end
    end
    exactE(kbT) = begin
        Z = 0.0;
        sE = 0.0
        for (k, ω) in Ω
            w = ω * exp(-k / kbT);
            Z += w;
            sE += w * k
        end
        return sE / Z
    end

    # learn g(E) with Wang–Landau, then use it as the fixed multicanonical weight
    g0 = rand(rng, (-1, 1), N)
    wl_energies, wl_logg = wang_landau(
        rng, g0, lat, model, WangLandau(; flatness=0.9, ln_f_final=1e-6)
    )
    S = muca_logweight(wl_energies, wl_logg)

    g1 = fill(1, N)                                   # start inside the support (E=-32)
    Es = multicanonical(
        rng, g1, lat, model, S, Multicanonical(; sweeps=800_000, therm=80_000)
    )

    # (1) flat histogram over the interior of the WL support (drop the two extreme
    # levels, which the reflecting boundary under-samples)
    lo, hi = extrema(round.(Int, wl_energies))
    hist = Dict{Int,Int}()
    for E in Es
        k = round(Int, E)
        hist[k] = get(hist, k, 0) + 1
    end
    interior = [get(hist, k, 0) for k in (lo + 4):4:(hi - 4) if haskey(Ω, k)]
    @test !isempty(interior)
    @test minimum(interior) > 0
    @test maximum(interior) / (sum(interior) / length(interior)) < 2.0   # roughly flat

    # (2) canonical ⟨E⟩(T) from the single MUCA run, wide range, vs exact
    for kbT in (1.5, 2.0, 2.27, 2.8, 3.5)
        @test isapprox(muca_mean(Es, S, kbT), exactE(kbT); rtol=0.03)
    end

    @test_throws ArgumentError muca_mean(Float64[], S, 2.0)
    @test_throws ArgumentError multicanonical(
        rng, fill(1, N), lat, model, (E -> -Inf), Multicanonical(; sweeps=10, therm=0)
    )
end
