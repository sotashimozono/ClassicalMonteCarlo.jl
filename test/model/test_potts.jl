@testset "Potts Model Tests" begin
    rng = MersenneTwister(42)
    lat = build_lattice(Square, 3, 3)
    N = num_sites(lat)

    q = 3
    model = PottsModel(; q=q, J=1.0)

    @testset "Local Hamiltonian" begin
        grids = ones(Int, N)
        E_loc = local_hamiltonian(grids, lat, model, 1)
        @test E_loc ≈ -4.0

        grids[1] = 2
        E_loc_changed = local_hamiltonian(grids, lat, model, 1)
        @test E_loc_changed ≈ 0.0
    end

    @testset "Proposal: SpinFlip" begin
        grids = ones(Int, N)
        alg = LocalUpdate(; proposal=SpinFlip())

        for _ in 1:20
            changes = propose(rng, alg.proposal, grids, lat, model, 1)
            c = changes[1]

            @test c.index == 1
            @test c.old_val == 1
            @test c.new_val != 1
            @test 1 <= c.new_val <= q
        end
    end

    @testset "Energy Difference Consistency" begin
        grids = rand(rng, 1:q, N)
        E_total_old = total_energy(grids, lat, model)

        site = 1
        current = grids[site]
        new_val = mod1(current + 1, q)
        changes = (LocalChange(site, new_val, current),)

        dE_diff = calculate_diff_energy(grids, lat, model, changes)

        grids_new = copy(grids)
        grids_new[site] = new_val
        E_total_new = total_energy(grids_new, lat, model)

        @test dE_diff ≈ (E_total_new - E_total_old)
    end
end
