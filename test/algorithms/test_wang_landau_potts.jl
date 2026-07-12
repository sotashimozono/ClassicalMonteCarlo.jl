using Test
using Random
using ClassicalMonteCarlo
using Lattice2D

# Wang–Landau is model-generic: it drives any discrete-spectrum model through
# `propose`/`calculate_diff_energy`. Here we confirm it works on the q=3 Potts
# model, validated against the EXACT density of states (all 3^9 configs of a 3×3
# lattice), never another MC run.
function exact_potts_dos(lat, model)
    N = lat.N
    q = model.q
    g = fill(1, N)
    gE = Dict{Int,Int}()
    for c in 0:(q ^ N - 1)
        x = c
        @inbounds for i in 1:N
            g[i] = (x % q) + 1
            x ÷= q
        end
        E = round(Int, total_energy(g, lat, model))
        gE[E] = get(gE, E, 0) + 1
    end
    return gE
end

@testset "Wang–Landau on q=3 Potts (vs exact DOS, 3×3)" begin
    rng = MersenneTwister(77)
    lat = build_lattice(Square, 3, 3)
    model = PottsModel(; q=3, J=1.0)

    gex = exact_potts_dos(lat, model)
    Es = sort(collect(keys(gex)))
    @test sum(values(gex)) == 3^9
    lg_ex = Float64[log(gex[E]) for E in Es]
    lg_ex .-= minimum(lg_ex)

    grids = rand(rng, 1:3, lat.N)
    energies, lg_wl = wang_landau(
        rng,
        grids,
        lat,
        model,
        WangLandau(; flatness=0.8, ln_f_final=1e-5, check_interval=20_000),
    )

    @test Set(round.(Int, energies)) == Set(Es)          # exact reachable spectrum

    idx = Dict(round(Int, e) => i for (i, e) in enumerate(energies))
    lg_wl_ord = Float64[lg_wl[idx[E]] for E in Es]
    lg_wl_ord .-= minimum(lg_wl_ord)
    @test maximum(abs.(lg_wl_ord .- lg_ex)) < 0.6         # log g(E) shape

    for kbT in (0.8, 1.5)
        β = 1.0 / kbT
        wex = Float64[log(gex[E]) - β * E for E in Es]
        m = maximum(wex);
        Z = sum(exp.(wex .- m));
        pex = exp.(wex .- m) ./ Z
        Eex = sum(pex .* Es);
        Cex = (sum(pex .* Es .^ 2) - Eex^2) / kbT^2
        th = wl_thermodynamics(energies, lg_wl, kbT)
        @test isapprox(th.energy, Eex; rtol=0.03)
        @test isapprox(th.C, Cex; rtol=0.15, atol=0.5)
    end
end
