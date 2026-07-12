using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Thermodynamic observables (Binder cumulant, susceptibility, specific heat) from
# fluctuations, all validated against the SAME formulas evaluated on the EXACT
# canonical moments from 2^N enumeration (independent oracle). Uses the same
# order parameter m=|Σs|/N (= measure_magnetization) in MC and in the oracle.
@testset "Thermodynamics — Binder/χ/C vs exact enumeration" begin
    rng = MersenneTwister(909)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)
    c = ClassicalMonteCarlo.get_binder_coeff(model)

    function exact(kbT)
        g = ones(Int, N);
        Z = 0.0
        sm = 0.0;
        sm2 = 0.0;
        sm4 = 0.0;
        sE = 0.0;
        sE2 = 0.0
        for cfg in 0:(2 ^ N - 1)
            @inbounds for i in 1:N
                g[i] = ((cfg >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            m = measure_magnetization(g, lat, model)     # |Σs|/N, identical to MC
            E = total_energy(g, lat, model)
            w = exp(-E / kbT);
            Z += w
            sm += w * m;
            sm2 += w * m^2;
            sm4 += w * m^4;
            sE += w * E;
            sE2 += w * E^2
        end
        m1 = sm / Z;
        m2 = sm2 / Z;
        m4 = sm4 / Z;
        e1 = sE / Z;
        e2 = sE2 / Z
        return (
            binder=binder_cumulant(m2, m4; coeff=c),
            chi=susceptibility(m1, m2, kbT, N),
            C=specific_heat(e1, e2, kbT, N),
            energy=e1,
        )
    end

    for kbT in (2.0, 2.4, 3.2)
        ex = exact(kbT)
        g = rand(rng, (-1, 1), N)
        mc = measure_thermodynamics(
            rng,
            g,
            lat,
            model,
            LocalUpdate();
            kbT=kbT,
            sweeps=600_000,
            therm=40_000,
            interval=2,
        )
        @test isapprox(mc.energy, ex.energy; rtol=0.02)
        @test isapprox(mc.binder, ex.binder; atol=0.02)
        @test isapprox(mc.susceptibility, ex.chi; rtol=0.05)
        @test isapprox(mc.specific_heat, ex.C; rtol=0.06)
    end

    # T→∞ closed form: independent ±1 spins give ⟨M⁴⟩/⟨M²⟩² = 3 − 2/N exactly,
    # so U₄ = 1 − (3−2/N)/3 = 2/(3N) — a hard analytic check on the Binder formula.
    @test isapprox(exact(1.0e6).binder, 2 / (3N); atol=1.0e-3)

    @test_throws ArgumentError binder_cumulant(0.0, 1.0)
end
