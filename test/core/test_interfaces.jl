struct MockModel <: AbstractModel{Int} end
struct MockAlgorithm <: UpdateAlgorithm end

mutable struct MockObserver <: AbstractObserver
    interval::Int
    call_count::Int
    last_step_recorded::Int
end
MockObserver(interval) = MockObserver(interval, 0, -1)

function Lattice2DMonteCarlo.update_step!(
    rng::AbstractRNG,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::MockModel,
    alg::MockAlgorithm;
    kbT::Float64=1.0,
)
    return nothing
end

function Lattice2DMonteCarlo.observe!(
    obs::MockObserver,
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::MockModel,
    step::Int;
    kbT::Float64=1.0,
) where {T}
    obs.call_count += 1
    obs.last_step_recorded = step
    return nothing
end

function Lattice2DMonteCarlo.local_hamiltonian(
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::MockModel,
    site::Int;
    val::Int=grids[site],
)
    return float(val)
end

@testset "Core Interfaces & Dispatch" begin
    rng = MersenneTwister(1234)
    Lx, Ly = 10, 10

    lat = build_lattice(rand(Lattice2D.AVAILABLE_LATTICES), Lx, Ly)

    grids = ones(Int, lat.N)
    model = MockModel()
    alg = MockAlgorithm()
    kbT = 1.0

    @testset "Energy Calculation" begin
        E = total_energy(grids, lat, model)
        @test E ≈ float(lat.N) / 2.0
    end

    @testset "Simulation Loop (run!)" begin
        nsteps = 100
        obs_interval = 10
        observer = MockObserver(obs_interval)

        run!(
            rng, grids, lat, model, alg, AbstractObserver[observer]; kbT=kbT, nsteps=nsteps
        )

        expected_calls = 1 + (nsteps ÷ obs_interval)
        @test observer.call_count == expected_calls
        @test observer.last_step_recorded == nsteps
    end

    @testset "Dispatch Error Check" begin
        struct UnimplementedAlg <: UpdateAlgorithm end
        bad_alg = UnimplementedAlg()

        @test_throws ErrorException run!(
            rng, grids, lat, model, bad_alg, AbstractObserver[]; kbT=kbT, nsteps=1
        )
    end
end
