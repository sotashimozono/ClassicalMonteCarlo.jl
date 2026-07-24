using Test
using Random
using ClassicalMonteCarlo
using Lattice2D

@testset "Ising Model Tests" begin
    rng = MersenneTwister(42)
    lat = build_lattice(Square, 3, 3)
    N = num_sites(lat)

    model = IsingModel(; J=1.0, h=0.1)
    @testset "Local Hamiltonian" begin
        grids = ones(Int, N)

        E_loc = local_hamiltonian(grids, lat, model, 1)
        @test E_loc ≈ -4.1

        grids[1] = -1
        E_loc_flip = local_hamiltonian(grids, lat, model, 1)
        @test E_loc_flip ≈ 4.1
    end

    @testset "Proposal: SpinFlip" begin
        grids = ones(Int, N)
        alg = LocalUpdate(; proposal=SpinFlip())

        changes = propose(rng, alg.proposal, grids, lat, model, 1)
        @test length(changes) == 1
        c = changes[1]
        @test c.index == 1
        @test c.old_val == 1
        @test c.new_val == -1
    end

    @testset "Proposal: SpinExchange" begin
        grids = vec([((i + j) % 2 == 0 ? 1 : -1) for j in 1:3, i in 1:3])

        alg = LocalUpdate(; proposal=SpinExchange())

        success = false
        for _ in 1:100
            changes = propose(rng, alg.proposal, grids, lat, model, 1)
            if !isempty(changes)
                success = true
                @test length(changes) == 2
                c1, c2 = changes

                @test c1.new_val == c2.old_val
                @test c2.new_val == c1.old_val

                @test c2.index in lat.nearest_neighbors[1]
                break
            end
        end
        @test success
    end

    @testset "Energy Difference Consistency" begin
        grids = rand(rng, [-1, 1], N)

        E_total_old = total_energy(grids, lat, model)

        @testset "SpinFlip Consistency" begin
            site = 1
            changes = (LocalChange(site, -grids[site], grids[site]),)

            dE_diff = calculate_diff_energy(grids, lat, model, changes)

            grids_new = copy(grids)
            grids_new[site] *= -1
            E_total_new = total_energy(grids_new, lat, model)

            @test dE_diff ≈ (E_total_new - E_total_old)
        end

        @testset "SpinExchange Consistency (Neighbor Correction)" begin
            site1 = 1
            neighbors = lat.nearest_neighbors[site1]
            target_neighbor = 0
            for n in neighbors
                if grids[n] != grids[site1]
                    target_neighbor = n
                    break
                end
            end

            if target_neighbor == 0
                target_neighbor = neighbors[1]
                grids[target_neighbor] = -grids[site1]
                E_total_old = total_energy(grids, lat, model)
            end

            site2 = target_neighbor

            c1 = LocalChange(site1, grids[site2], grids[site1])
            c2 = LocalChange(site2, grids[site1], grids[site2])
            changes = (c1, c2)

            dE_diff = calculate_diff_energy(grids, lat, model, changes)

            grids_new = copy(grids)
            grids_new[site1], grids_new[site2] = grids_new[site2], grids_new[site1]
            E_total_new = total_energy(grids_new, lat, model)

            @test dE_diff ≈ (E_total_new - E_total_old)
        end
    end
end
