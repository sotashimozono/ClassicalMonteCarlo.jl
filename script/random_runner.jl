using ClassicalMonteCarlo
using Lattice2D
using Random
using Dates
using JSON
using Plots

# ==============================================================================
# 1. パラメータ空間の定義
# ==============================================================================

# 格子の種類とサイズの候補
const LATTICE_TYPES = Lattice2D.AVAILABLE_LATTICES
const SIZES = [(10, 10), (20, 20), (32, 32)]

const MODEL_BUILDERS = [
    () -> IsingModel(; J=1.0, h=0.0),
    () -> IsingModel(; J=1.0, h=0.1),
    () -> PottsModel(; q=3, J=1.0),
    () -> PottsModel(; q=4, J=1.0),
    () -> XYModel(; J=1.0),
]

# 温度範囲
const T_RANGE = (0.1, 5.0)

# アルゴリズム
const ALGORITHMS = [
    () -> LocalUpdate(; rule=Metropolis(), selection=RandomSiteSelection()),
    () -> LocalUpdate(; rule=Metropolis(), selection=SequentialSweep()),
    # Glauberなども追加可能
]

# ==============================================================================
# 2. ランダムサンプリングと実行
# ==============================================================================

function run_random_simulation()
    rng = Random.default_rng()

    # --- Random Selection ---
    LType = rand(rng, LATTICE_TYPES)
    (Lx, Ly) = rand(rng, SIZES)
    lat = build_lattice(LType(), Lx, Ly)

    model_builder = rand(rng, MODEL_BUILDERS)
    model = model_builder()

    alg_builder = rand(rng, ALGORITHMS)
    alg = alg_builder()

    kbT = rand(rng) * (T_RANGE[2] - T_RANGE[1]) + T_RANGE[1]

    # ログ出力
    println("--- Starting Simulation ---")
    println("Lattice: $(typeof(lat.topology)) ($Lx x $Ly)")
    println("Model:   $(typeof(model))")
    println("Alg:     $(typeof(alg))")
    println("Temp:    $(round(kbT, digits=3))")

    # --- Simulation ---
    # 初期化
    grids = if model isa XYModel
        rand(rng, Float64, lat.N) .* 2π
    elseif model isa PottsModel
        rand(rng, 1:(model.q), lat.N)
    else # Ising
        rand(rng, [-1, 1], lat.N)
    end

    # Burn-in (空回し)
    run!(rng, grids, lat, model, alg, AbstractObserver[]; kbT=kbT, nsteps=2000)

    # Sampling & Observation
    # ここでは最終状態のスナップショットだけ撮る例ですが、
    # 磁化率などのObserverを入れて時系列データを取るのもアリです
    obs = AbstractObserver[]
    run!(rng, grids, lat, model, alg, obs; kbT=kbT, nsteps=1000)

    # --- Save Results ---
    # 保存パスの取得 (以前作った関数を活用)
    # ここでは docs/gallery というフォルダに入れる想定

    # ファイル名に温度などの詳細情報を含めるためのカスタム拡張
    # path/to/IsingModel/Square_Lx10_Ly10_Metropolis_T1.234.png
    T_str = "T$(round(kbT, digits=3))"
    timestamp = Dates.format(now(), "MMdd_HHmm")

    base_name = ClassicalMonteCarlo.get_file_base(lat, model, alg)
    filename = "$(base_name)_$(T_str)_$(timestamp)"

    # 保存先 (GitHub Actionsのワークスペース内)
    gallery_dir = joinpath(
        pkgdir(ClassicalMonteCarlo), "gallery", string(nameof(typeof(model)))
    )
    mkpath(gallery_dir)

    # 1. 画像保存
    p = visualize_snapshot(grids, lat, model)
    title!(p, "$base_name\nT=$kbT")
    png_path = joinpath(gallery_dir, filename * ".png")
    savefig(p, png_path)
    println("Saved plot: $png_path")

    # 2. メタデータ保存 (後で解析したくなった時用)
    json_path = joinpath(gallery_dir, filename * ".json")
    meta = Dict(
        "lattice" => string(nameof(typeof(lat.topology))),
        "Lx" => Lx,
        "Ly" => Ly,
        "model" => string(nameof(typeof(model))),
        "algorithm" => string(nameof(typeof(alg))),
        "kbT" => kbT,
        "timestamp" => timestamp,
    )
    open(json_path, "w") do f
        JSON.print(f, meta, 4)
    end
    return println("Saved meta: $json_path")
end

# 実行
run_random_simulation()
