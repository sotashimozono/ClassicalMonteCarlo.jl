@testset "Advanced Observer System" begin
    lat = build_lattice(Square, 4, 4)
    N = num_sites(lat)
    kbT = 2.0
    rng = Random.default_rng()

    @testset "Physical Measurements (Physics Layer)" begin
        @testset "Ising Magnetization" begin
            model = IsingModel()
            grids = ones(Int, N)
            @test measure_magnetization(grids, lat, model) ≈ 1.0

            grids[1:8] .= -1
            @test measure_magnetization(grids, lat, model) ≈ 0.0
        end

        @testset "Potts Magnetization (Order Parameter)" begin
            model = PottsModel(; q=3)

            grids = ones(Int, N)
            @test measure_magnetization(grids, lat, model) ≈ 1.0

            grids = zeros(Int, N)
            grids[1:5] .= 1
            grids[6:10] .= 2
            grids[11:16] .= 3
            M = measure_magnetization(grids, lat, model)
            @test M ≈ 0.0625
        end

        @testset "XY Magnetization" begin
            model = XYModel()
            grids = zeros(Float64, N)
            @test measure_magnetization(grids, lat, model) ≈ 1.0 atol = 1e-10

            grids[1:8] .= π
            @test measure_magnetization(grids, lat, model) ≈ 0.0 atol = 1e-10
        end
    end

    @testset "FunctionObserver (Measurement Layer)" begin
        model = IsingModel()
        grids = ones(Int, N)

        obs = FunctionObserver("Site1", (g, l, m) -> float(g[1]))
        observe!(obs, grids, lat, model, 100; kbT=kbT)
        @test length(obs.steps) == 1
        @test obs.history[1] == 1.0

        df = to_dataframe(obs)
        @test size(df) == (1, 2)
        @test names(df) == ["Step", "Site1"]
        @test df.Site1[1] == 1.0
    end

    @testset "ThermodynamicObserver (Analysis Layer)" begin
        @testset "Accumulation & Calculation" begin
            model = IsingModel()
            obs = ThermodynamicObserver()

            grids1 = ones(Int, N)
            observe!(obs, grids1, lat, model, 1; kbT=kbT)

            grids2 = -ones(Int, N)
            observe!(obs, grids2, lat, model, 2; kbT=kbT)

            @test obs.n_samples == 2
            @test obs.sum_M == 2.0
            @test obs.sum_M2 == 2.0

            res = get_thermodynamics(obs, kbT, N, model)
            @test res["Magnetization"] ≈ 1.0
            @test res["Susceptibility"] ≈ 0.0
        end

        @testset "Binder Parameter Logic" begin
            model_ising = IsingModel()
            obs_i = ThermodynamicObserver()
            obs_i.n_samples = 1
            obs_i.sum_M2 = 1.0
            obs_i.sum_M4 = 1.0

            res_i = get_thermodynamics(obs_i, 1.0, 10, model_ising)
            @test res_i["BinderParam"] ≈ 2 / 3

            model_xy = XYModel()
            obs_xy = ThermodynamicObserver()
            obs_xy.n_samples = 1
            obs_xy.sum_M2 = 1.0
            obs_xy.sum_M4 = 1.0

            res_xy = get_thermodynamics(obs_xy, 1.0, 10, model_xy)
            @test res_xy["BinderParam"] ≈ 0.5
        end
    end
end
