using Test
using Random
using ClassicalMonteCarlo
using Lattice2D

# ==============================================================================
# テスト用のMock定義
# ==============================================================================
struct MockLogicModel <: AbstractModel{Int} end

# 提案メソッドのMock
struct MockProposal <: ProposalMethod end

# 【変更】引数順序: alg, grids, lat, model, site
function ClassicalMonteCarlo.propose(
    rng::AbstractRNG,
    alg::MockProposal,
    grids::AbstractVector{Int}, # <--- gridsがここに来た
    lat::AbstractLattice,
    model::MockLogicModel,
    site::Int,
)
    return (LocalChange(site, -1, grids[site]),)
end

# 【変更】引数順序: grids, lat, model, site
function ClassicalMonteCarlo.local_hamiltonian(
    grids::AbstractVector{Int},
    lat::AbstractLattice,
    model::MockLogicModel, # <--- modelがsiteの前に来た
    site::Int;
    val::Int=grids[site],
)
    return float(val)
end

# サイト選択順序を確認するためのMockモデル
struct OrderCheckModel <: AbstractModel{Int} end

# ==============================================================================
# テスト実行
# ==============================================================================

@testset "Local Update Logic" begin
    rng = MersenneTwister(1234)
    lat = build_lattice(rand([AVAILABLE_LATTICES...]), 4, 4)
    N = num_sites(lat)

    @testset "Acceptance Rules" begin
        metro = Metropolis()
        @test check_acceptance(rng, metro, -1.0, 1.0) == true
        @test check_acceptance(rng, metro, 1.0, 1e-10) == false

        glau = Glauber()
        @test check_acceptance(rng, glau, -100.0, 1.0) == true
    end

    @testset "Energy Difference Calculation" begin
        model = MockLogicModel()
        grids = fill(10, N)
        change = LocalChange(1, 20, 10)

        dE = calculate_diff_energy(grids, lat, model, (change,))
        @test dE ≈ 10.0

        multi_change = (LocalChange(1, 20, 10), LocalChange(2, 20, 10))
        @test_throws ErrorException calculate_diff_energy(grids, lat, model, multi_change)
    end

    @testset "Site Selection Logic (Sequential)" begin
        visited_sites = Int[]

        # 【変更】引数順序: grids, lat, model, alg; kbT
        function ClassicalMonteCarlo.update_single_site!(
            rng::AbstractRNG,
            site::Int,
            grids::AbstractVector{Int},
            lat::AbstractLattice,
            model::OrderCheckModel,
            alg::LocalUpdate;
            kbT::Float64=1.0,
        )
            push!(visited_sites, site)
            return nothing
        end

        model = OrderCheckModel()
        alg = LocalUpdate(; selection=SequentialSweep())
        grids = zeros(Int, N)

        process_site_selection!(rng, alg.selection, grids, lat, model, alg; kbT=1.0)

        @test length(visited_sites) == N
        @test visited_sites == 1:N
        @test issorted(visited_sites)
    end

    @testset "Update Flow (Propose -> Accept -> Update)" begin
        grids = fill(10, N)
        model = MockLogicModel()

        alg = LocalUpdate(;
            rule=Metropolis(), selection=SequentialSweep(), proposal=MockProposal()
        )

        # 【変更】update_step! の呼び出し変更
        update_step!(rng, grids, lat, model, alg; kbT=1.0)

        @test all(grids .== -1)
    end
end
