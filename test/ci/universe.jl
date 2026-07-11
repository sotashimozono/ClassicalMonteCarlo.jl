# test/ci/universe.jl — canonical VALIDATION-CASE universe (single source of truth).
#
# Included by BOTH the validation test files (test/validation/test_*.jl) and the
# shard planner (test/ci/plan_shards.jl).  Pure stdlib, NO
# `using ClassicalMonteCarlo` / `using QAtlas`, so the planner stays fast (it
# never precompiles the package).  This file alone decides WHAT the shardable
# validation universe is; the timing plane only decides HOW to split it.
#
# The shard "universe" here is a set of validation CASES keyed by (model,
# parameter), each owned by exactly one validation test file (`group`).  A CI
# shard runs a subset of these cases; the union of all shards is exactly this
# list, so no leg is ever dropped and none runs twice.
#
# This file is idempotent: multiple validation files `include` it into the same
# process, so the whole body is guarded by a load flag (same trick as
# test/validation/mc_helpers.jl).

if !isdefined(Main, :CMC_VALIDATION_UNIVERSE_LOADED)
    const CMC_VALIDATION_UNIVERSE_LOADED = true

    # test/ci/ → test/
    const TEST_ROOT = dirname(@__DIR__)
    const VALIDATION_DIR = joinpath(TEST_ROOT, "validation")

    # ── Canonical, ordered case universe ─────────────────────────────────────
    # One entry per shardable validation leg.  Keyed by (model, param).
    #   id     : unique, stable case id (the timing key + CMC_TEST_CASES token)
    #   group  : which validation test file OWNS this case (basename, no `.jl`)
    #   model  : physical model the case validates
    #   param  : the parameter point / observable that distinguishes the leg
    #   weight : rough runtime estimate; used ONLY as the round-robin fallback
    #            ordering — real recorded timings override it via LPT packing.
    const VALIDATION_CASES = [
        (
            id="exact_ising_bruteZ",
            group="test_ising_exact_small",
            model="IsingSquare",
            param="partition_function(exact-vs-exact)",
            weight=2,
        ),
        (
            id="exact_ising_mc",
            group="test_ising_exact_small",
            model="IsingSquare",
            param="mc_vs_bruteforce_4x4",
            weight=6,
        ),
        (
            id="run_observer_e2e",
            group="test_run_observer_e2e",
            model="IsingSquare",
            param="observe!_dispatch_e2e",
            weight=1,
        ),
        (
            id="cross_algo_ising",
            group="test_cross_algorithm",
            model="IsingSquare",
            param="T=3.0_L=8_scheme_agreement",
            weight=4,
        ),
        (
            id="limits_ising_lowT",
            group="test_limits",
            model="IsingSquare",
            param="T->0_ground_state",
            weight=1,
        ),
        (
            id="limits_ising_highT",
            group="test_limits",
            model="IsingSquare",
            param="T->inf_paramagnet",
            weight=2,
        ),
        (
            id="qatlas_oracle_sanity",
            group="test_qatlas_corroboration",
            model="IsingSquare",
            param="Tc+critical_exponents(exact)",
            weight=1,
        ),
        (
            id="qatlas_yang_mag",
            group="test_qatlas_corroboration",
            model="IsingSquare",
            param="SpontaneousMagnetization(T<Tc)",
            weight=8,
        ),
        (
            id="qatlas_binder_crossing",
            group="test_qatlas_corroboration",
            model="IsingSquare",
            param="Binder_cumulant_crossing(Tc)",
            weight=12,
        ),
        (
            id="qatlas_beta_consistency",
            group="test_qatlas_corroboration",
            model="IsingSquare",
            param="beta_exponent_consistency",
            weight=10,
        ),
    ]

    const ALL_CASE_IDS = [c.id for c in VALIDATION_CASES]

    # ── Completeness / wiring guard ──────────────────────────────────────────
    # Runs wherever this file is included (every shard + the planner) and fails
    # loudly, so a case can't be added without wiring nor a test leg left
    # unsharded:
    #   (1) every case id is unique, and
    #   (2) the SET of `group` values equals the SET of validation test-file
    #       basenames actually on disk — i.e. every validation file owns at
    #       least one case, and no case names a file that does not exist.
    let
        # (1) unique ids
        if length(unique(ALL_CASE_IDS)) != length(ALL_CASE_IDS)
            seen = Dict{String,Int}()
            for id in ALL_CASE_IDS
                seen[id] = get(seen, id, 0) + 1
            end
            dups = sort([id for (id, n) in seen if n > 1])
            error("universe.jl guard: duplicate VALIDATION_CASES id(s): $(dups)")
        end

        # (2) groups == validation file basenames
        _is_test_file(f) = startswith(f, "test_") && endswith(f, ".jl")
        discovered = Set{String}()
        if isdir(VALIDATION_DIR)
            for f in readdir(VALIDATION_DIR)
                _is_test_file(f) || continue
                push!(discovered, f[1:(end - 3)])   # strip ".jl"
            end
        end
        groups = Set(c.group for c in VALIDATION_CASES)

        orphan_files = sort(collect(setdiff(discovered, groups)))   # file, no case
        isempty(orphan_files) || error(
            "universe.jl guard: these validation test files own NO case and " *
            "would run unsharded/ungated — add a case with group=<basename> in " *
            "VALIDATION_CASES: $(orphan_files)",
        )
        dangling_groups = sort(collect(setdiff(groups, discovered)))  # case, no file
        isempty(dangling_groups) || error(
            "universe.jl guard: these VALIDATION_CASES groups name a validation " *
            "file that does not exist on disk: $(dangling_groups)",
        )
    end

    # ── Selection: which case ids should THIS process run? ────────────────────
    # env CMC_TEST_CASES = comma-separated ids.
    #   unset/absent   → ALL cases (local `Pkg.test()` runs everything)
    #   set & empty "" → NONE (an empty shard runs no validation legs)
    #   set & non-empty→ exactly those ids
    function selected_case_ids()
        haskey(ENV, "CMC_TEST_CASES") || return Set(ALL_CASE_IDS)
        raw = strip(ENV["CMC_TEST_CASES"])
        isempty(raw) && return Set{String}()
        return Set(String.(strip.(split(raw, ","))))
    end
    case_selected(id) = id in selected_case_ids()

    # ── Per-case runner with optional timing emit ────────────────────────────
    # Usage in a validation file:
    #     run_case("qatlas_yang_mag") do
    #         @testset "Yang magnetization vs MC" begin
    #             ...
    #         end
    #     end
    # If the case is not selected, the block is skipped entirely.  When
    # CMC_EMIT_TIMING=1 the wall time of the block is appended to
    # `$(CMC_CIOUT_DIR)/timings-$(CMC_SHARD_ID).tsv` as "<id>\t<seconds>" so
    # the CI record-timings job can feed the LPT planner next time.  @testset
    # nests correctly even from inside the closure (the testset stack is
    # dynamically scoped).
    function _emit_timing(id, seconds)
        try
            dir = get(ENV, "CMC_CIOUT_DIR", joinpath(TEST_ROOT, ".ci-out"))
            mkpath(dir)
            sid = get(ENV, "CMC_SHARD_ID", string(getpid()))
            open(joinpath(dir, "timings-$(sid).tsv"), "a") do io
                println(io, string(id), '\t', seconds)
            end
        catch e
            @warn "CMC timing emit failed (non-fatal)" id exception = (e,)
        end
        return nothing
    end

    function run_case(f, id::AbstractString)
        case_selected(id) || return nothing
        if get(ENV, "CMC_EMIT_TIMING", "0") == "1"
            t = @elapsed f()
            _emit_timing(id, t)
        else
            f()
        end
        return nothing
    end
end
