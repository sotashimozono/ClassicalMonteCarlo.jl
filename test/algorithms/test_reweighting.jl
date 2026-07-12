using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Single-histogram (Ferrenberg–Swendsen) reweighting. Energies sampled at ONE
# temperature kbT0 are reweighted to nearby temperatures; inside the histogram-
# overlap window the reweighted ⟨E⟩(kbT) and specific heat C(kbT) must match the
# EXACT canonical values from 2^N enumeration. Independent oracle = exact
# enumeration; the check is that the reweighting identity is implemented right.
@testset "Single-histogram reweighting — ⟨E⟩(T),C(T) vs exact enumeration" begin
    rng = MersenneTwister(555)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)

    function exact(kbT)
        g = ones(Int, N);
        Z = 0.0;
        sE = 0.0;
        sE2 = 0.0
        for c in 0:(2 ^ N - 1)
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            E = total_energy(g, lat, model);
            w = exp(-E / kbT)
            Z += w;
            sE += w * E;
            sE2 += w * E^2
        end
        Em = sE / Z
        return (energy=Em, C=(sE2 / Z - Em^2) / (kbT^2 * N))
    end

    # sample once at kbT0 near Tc (broad energy histogram ⇒ wide overlap window)
    kbT0 = 2.4
    g = rand(rng, (-1, 1), N)
    energies = sample_energies(
        rng,
        g,
        lat,
        model,
        LocalUpdate();
        kbT=kbT0,
        sweeps=400_000,
        therm=20_000,
        interval=2,
    )

    # reweighting back to kbT0 reproduces the plain sample mean (sanity)
    @test isapprox(reweight_mean(energies, kbT0, kbT0), mean(energies); rtol=1e-12)

    # reweight across a window around kbT0 and compare to exact
    for kbT in (2.25, 2.35, 2.45, 2.55)
        ex = exact(kbT)
        @test isapprox(reweight_mean(energies, kbT0, kbT), ex.energy; rtol=0.02)
        @test isapprox(reweight_specific_heat(energies, kbT0, kbT, N), ex.C; rtol=0.06)
    end

    @test_throws ArgumentError reweight_mean(Float64[], kbT0, 2.3)
end
