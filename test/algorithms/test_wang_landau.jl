using Test
using Random
using ClassicalMonteCarlo
using Lattice2D

# Independent oracle: the EXACT density of states by brute-force enumeration of
# all 2^N Ising configurations (feasible for L=4 ⇒ 2^16). Everything below is
# checked against this closed-form reference, never against another MC run.
function exact_ising_dos(lat, model)
    N = num_sites(lat)
    N <= 20 || error("enumeration too large")
    g = Dict{Int,Int}()
    grids = ones(Int, N)
    for c in 0:(2 ^ N - 1)
        @inbounds for i in 1:N
            grids[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
        end
        E = round(Int, total_energy(grids, lat, model))
        g[E] = get(g, E, 0) + 1
    end
    return g
end

@testset "Wang–Landau vs exact DOS (2D Ising L=4)" begin
    rng = MersenneTwister(2026)
    lat = build_lattice(Square, 4, 4)      # 16 spins, PBC
    model = IsingModel(; J=1.0, h=0.0)

    gex = exact_ising_dos(lat, model)
    Es = sort(collect(keys(gex)))
    lg_ex = Float64[log(gex[E]) for E in Es]
    lg_ex .-= minimum(lg_ex)               # same gauge WL uses (min log g = 0)
    @test sum(values(gex)) == 2^16         # enumeration sanity: all states counted
    @test gex[-32] == 2                     # two ferromagnetic ground states

    grids = rand(rng, (-1, 1), num_sites(lat))
    energies, lg_wl = wang_landau(
        rng,
        grids,
        lat,
        model,
        WangLandau(; flatness=0.8, ln_f_final=1e-5, check_interval=20_000),
    )

    # (1) WL discovers exactly the reachable spectrum — no spurious/missing levels
    @test Set(round.(Int, energies)) == Set(Es)

    # (2) log g(E) shape matches the exact DOS within WL statistical accuracy
    idx = Dict(round(Int, e) => i for (i, e) in enumerate(energies))
    lg_wl_ord = Float64[lg_wl[idx[E]] for E in Es]
    lg_wl_ord .-= minimum(lg_wl_ord)
    @test maximum(abs.(lg_wl_ord .- lg_ex)) < 0.5

    # (3) derived thermodynamics ⟨E⟩(T), C(T) vs exact (normalisation-independent)
    for kbT in (1.5, 2.269, 3.0)            # incl. the 2D-Ising T_c ≈ 2.269
        β = 1.0 / kbT
        wex = Float64[log(gex[E]) - β * E for E in Es]
        m = maximum(wex);
        Z = sum(exp.(wex .- m));
        pex = exp.(wex .- m) ./ Z
        Eex = sum(pex .* Es);
        Cex = (sum(pex .* Es .^ 2) - Eex^2) / kbT^2

        th = wl_thermodynamics(energies, lg_wl, kbT)
        @test isapprox(th.energy, Eex; rtol=0.03)
        @test isapprox(th.C, Cex; rtol=0.12, atol=0.5)
    end
end
