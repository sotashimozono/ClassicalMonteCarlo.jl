using Test
using Random, Statistics
using ClassicalMonteCarlo
using Lattice2D

# Simulated annealing as a global optimiser. The best energy it finds must reach
# the EXACT global minimum over all 2^N configurations (independent oracle =
# brute-force enumeration), for ferromagnet, antiferromagnet (staggered Néel
# ground on the bipartite 4×4), and the 3-state Potts ferromagnet, and the
# returned config must actually attain that minimum.
@testset "Simulated annealing — reaches exact ground state" begin
    rng = MersenneTwister(2024)
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)

    # exact global minimum energy over all 2^N Ising configs
    function exact_min_ising(model)
        g = ones(Int, N); Emin = Inf
        for c in 0:(2^N - 1)
            @inbounds for i in 1:N
                g[i] = ((c >> (i - 1)) & 1) == 1 ? 1 : -1
            end
            E = total_energy(g, lat, model)
            E < Emin && (Emin = E)
        end
        return Emin
    end

    for model in (IsingModel(; J=1.0, h=0.0), IsingModel(; J=-1.0, h=0.0))
        Emin = exact_min_ising(model)
        # best of a few annealing restarts must equal the exact ground energy
        best = Inf; bestcfg = Int[]
        for r in 1:6
            g = rand(rng, (-1, 1), N)
            res = simulated_anneal(rng, g, lat, model, SimulatedAnnealing(; steps=300, sweeps_per_step=15))
            if res.energy < best
                best = res.energy; bestcfg = res.config
            end
        end
        @test best ≈ Emin
        @test total_energy(bestcfg, lat, model) ≈ Emin      # returned config attains it
    end

    # 3-state Potts ferromagnet: ground = all spins equal, E = -J·#bonds
    let model = PottsModel(; q=3, J=1.0)
        nbonds = sum(length(Lattice2D.neighbors(lat, i)) for i in 1:N) ÷ 2
        Emin = -model.J * nbonds
        best = Inf
        for r in 1:6
            g = rand(rng, 1:3, N)
            res = simulated_anneal(rng, g, lat, model, SimulatedAnnealing(; steps=300, sweeps_per_step=15))
            best = min(best, res.energy)
        end
        @test best ≈ Emin
    end

    @test_throws ArgumentError simulated_anneal(rng, rand(rng, (-1, 1), N), lat, IsingModel(), SimulatedAnnealing(; steps=1))
    @test_throws ArgumentError simulated_anneal(rng, rand(rng, (-1, 1), N), lat, IsingModel(), SimulatedAnnealing(; T0=1.0, Tf=2.0))
end
