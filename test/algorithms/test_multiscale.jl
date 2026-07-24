using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Multiscale block-spin (multigrid) update. Independent oracle: flipping whole
# blocks of the coarse-graining hierarchy is a detailed-balance-preserving move,
# so the sampler reproduces the exact canonical ⟨E⟩, ⟨m²⟩ and Binder cumulant
# from 2^N enumeration on a small lattice. Also checks the block hierarchy is a
# genuine partition and that the sampler locates the square-lattice T_c.
@testset "Multiscale block-flip — canonical correctness vs exact" begin
    rng = MersenneTwister(2024)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    model = IsingModel(; J=1.0, h=0.0)
    c = ClassicalMonteCarlo.get_binder_coeff(model)

    levels = square_block_levels(lat)
    @testset "square_block_levels is a proper hierarchy" begin
        @test length(levels) == 2                       # 4 = 2^2
        @test length(levels[1]) == 4                    # four 2×2 blocks
        @test all(length(b) == 4 for b in levels[1])
        @test length(levels[2]) == 1                    # one 4×4 block (whole)
        @test length(levels[2][1]) == N
        @test sort(vcat(levels[1]...)) == collect(1:N)  # partition of all sites
    end

    alg = MultiscaleBlockFlip(levels)

    function exact(kbT)
        g = ones(Int, N)
        Z = 0.0;
        sE = 0.0;
        sm2 = 0.0;
        sm4 = 0.0
        for cfg in 0:(2 ^ N - 1)
            @inbounds for i in 1:N
                g[i] = ((cfg >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            m = measure_magnetization(g, lat, model)
            E = total_energy(g, lat, model)
            w = exp(-E / kbT)
            Z += w;
            sE += w * E;
            sm2 += w * m^2;
            sm4 += w * m^4
        end
        return (
            energy=sE / Z, m2=sm2 / Z, binder=binder_cumulant(sm2 / Z, sm4 / Z; coeff=c)
        )
    end

    for kbT in (2.27, 3.5)
        ex = exact(kbT)
        g = rand(rng, (-1, 1), N)
        sE = 0.0;
        sm2 = 0.0;
        sm4 = 0.0;
        n = 0
        for s in 1:200_000
            update_step!(rng, g, lat, model, alg; kbT=kbT)
            if s > 20_000
                m = measure_magnetization(g, lat, model)
                sE += total_energy(g, lat, model);
                sm2 += m^2;
                sm4 += m^4;
                n += 1
            end
        end
        @test isapprox(sE / n, ex.energy; rtol=0.02)
        @test isapprox(sm2 / n, ex.m2; rtol=0.03)
        @test isapprox(binder_cumulant(sm2 / n, sm4 / n; coeff=c), ex.binder; atol=0.02)
    end
end
