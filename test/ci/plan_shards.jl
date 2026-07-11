# test/ci/plan_shards.jl — emit a balanced GitHub-matrix shard plan over the
# validation-CASE universe.
#
#   julia test/ci/plan_shards.jl <N> [timings.tsv]
#
# Prints (stdout, last line) a JSON array for `matrix: include:` —
#   [{"sid":"s01","cases":"id1,id2","full":"0"}, …]
#
# WHAT to run is the canonical case universe (universe.jl, the single source of
# truth + completeness guard).  Timings only decide HOW to split:
#   * timings.tsv present  → Longest-Processing-Time bin-packing so all shards
#                            finish at ≈ the same wall-clock.
#   * absent / unreadable  → deterministic round-robin (never a leak, just not
#                            yet time-optimal).
# Cases with no recorded time get a pessimistic estimate (P90 of known, or the
# case's own declared `weight` if there is no history at all) so a surprise-heavy
# new case is isolated rather than piled onto an already-heavy shard.
#
# The `full` field marks whether a shard should run with CMC_TEST_FULL=1.  The
# PR merge-gate keeps every shard fast (full="0"); the workflow overrides this
# to "1" on push:main so the persisted timings reflect the heavy computation.

include(joinpath(@__DIR__, "universe.jl"))   # VALIDATION_CASES, ALL_CASE_IDS

const N = let a = get(ARGS, 1, "")
    n = tryparse(Int, a)
    (n !== nothing && n >= 1) ||
        error("plan_shards.jl: arg 1 must be N>=1; got $(repr(a))")
    n
end
const TIMINGS_PATH = get(ARGS, 2, "")

# universe → ordered case ids + declared weights
const KEYS = ALL_CASE_IDS
const WEIGHT = Dict(c.id => Float64(c.weight) for c in VALIDATION_CASES)

# Load timings TSV ("id\tseconds"); silently ignore missing/garbage rows — the
# plane is advisory and must degrade, never abort planning.
function load_timings(path)
    t = Dict{String,Float64}()
    (isempty(path) || !isfile(path)) && return t
    for ln in eachline(path)
        parts = split(strip(ln), '\t')
        length(parts) == 2 || continue
        v = tryparse(Float64, parts[2])
        v === nothing && continue
        t[String(parts[1])] = v
    end
    return t
end
const TIMES = load_timings(TIMINGS_PATH)

# Pessimistic default for cases with no recorded time: P90 of the known
# timings, or (with no history at all) the case's own declared weight.
const DEFAULT_T = if isempty(TIMES)
    0.0   # unused when there is no history (weight is used directly)
else
    s = sort(collect(values(TIMES)))
    s[clamp(ceil(Int, 0.9 * length(s)), 1, length(s))]   # P90 of known
end

est(k) = get(TIMES, k, isempty(TIMES) ? get(WEIGHT, k, 1.0) : DEFAULT_T)

# Assignment: LPT when we have history, else round-robin over declared weight.
bins = [String[] for _ in 1:N]
loads = zeros(Float64, N)

if isempty(TIMES)
    for (i, k) in enumerate(KEYS)
        b = ((i - 1) % N) + 1
        push!(bins[b], k)
        loads[b] += est(k)
    end
    mode = "round-robin (no timing history; declared weights)"
else
    for k in sort(KEYS; by=est, rev=true)         # longest first
        b = argmin(loads)                          # least-loaded bin
        push!(bins[b], k)
        loads[b] += est(k)
    end
    mode = "LPT bin-packing (recorded timings)"
end

# Emit JSON (ASCII case ids only ⇒ no escaping needed).  `full="0"` on every
# shard: the PR gate is fast; the workflow flips CMC_TEST_FULL on push:main.
io = IOBuffer()
print(io, "[")
for b in 1:N
    b == 1 || print(io, ",")
    sid = "s" * lpad(string(b), 2, '0')
    print(io, "{\"sid\":\"", sid, "\",\"cases\":\"", join(bins[b], ","), "\",")
    print(io, "\"full\":\"0\"}")
end
print(io, "]")
plan_json = String(take!(io))

# Human-readable summary → stderr (does not pollute the JSON stdout).
println(
    stderr,
    "plan_shards: N=$N  mode=$mode  cases=$(length(KEYS))" *
    (isempty(TIMES) ? "" : "  default_t=$(round(DEFAULT_T; digits=3))s"),
)
for b in 1:N
    ids = isempty(bins[b]) ? "(empty)" : join(bins[b], ", ")
    println(
        stderr,
        "  s$(lpad(string(b),2,'0')): $(length(bins[b])) case(s)  " *
        "est=$(round(loads[b]; digits=1))  [$(ids)]",
    )
end

println(plan_json)
