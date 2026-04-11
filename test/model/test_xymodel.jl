@testset "XY Model Tests" begin
    rng = MersenneTwister(42)
    lat = build_lattice(Square, 3, 3)
    N = lat.N

    model = XYModel(; J=1.0)

    @testset "Local Hamiltonian" begin
        grids = zeros(Float64, N)
        E_loc = local_hamiltonian(grids, lat, model, 1)
        @test E_loc ≈ -4.0

        grids[1] = π / 2
        E_loc_90 = local_hamiltonian(grids, lat, model, 1)
        @test E_loc_90 ≈ 0.0 atol=1e-10

        grids[1] = π
        E_loc_180 = local_hamiltonian(grids, lat, model, 1)
        @test E_loc_180 ≈ 4.0
    end

    @testset "Proposal: UniformShift" begin
        grids = zeros(Float64, N)
        width = 0.2
        alg = LocalUpdate(; proposal=UniformShift(; width=width))

        for _ in 1:20
            changes = propose(rng, alg.proposal, grids, lat, model, 1)
            c = changes[1]

            @test c.index == 1
            @test c.old_val == 0.0

            diff = abs(c.new_val - c.old_val)
            @test diff <= width / 2.0 + 1e-9
            @test diff > 0.0
        end
    end

    @testset "Energy Difference Consistency" begin
        grids = rand(rng, Float64, N) .* 2π
        E_total_old = total_energy(grids, lat, model)

        site = 1
        current = grids[site]
        new_val = current + 0.5
        changes = (LocalChange(site, new_val, current),)

        dE_diff = calculate_diff_energy(grids, lat, model, changes)

        grids_new = copy(grids)
        grids_new[site] = new_val
        E_total_new = total_energy(grids_new, lat, model)

        @test dE_diff ≈ (E_total_new - E_total_old)
    end
end
