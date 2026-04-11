using Lattice2D
using Random, Statistics, Plots

abstract type MonteCarloMethod end

abstract type AbstractModel <: MonteCarloMethod end
function total_hamiltonian(grids::AbstractVector{Int}, lat::Lattice, model::AbstractModel)
    total_h = 0.0
    for site in 1:(lat.N)
        total_h += local_hamiltonian(grids, lat, site, model)
    end
    return total_h / 2
end

@kwdef struct IsingModel <: AbstractModel
    J::Float64 = 1.0
    H::Float64 = 0.0
end
function local_hamiltonian(
    grids::AbstractVector{Int}, lat::Lattice, site::Int, model::IsingModel
)
    local_h = -model.H * grids[site]
    for neighbor in lat.nearest_neighbors[site]
        local_h -= model.J * grids[site] * grids[neighbor]
    end
    return local_h
end

abstract type MeasurementMethod <: MonteCarloMethod end
struct NoMeasurement <: MeasurementMethod end
struct EnergyMeasurement <: MeasurementMethod end
struct MagnetizationMeasurement <: MeasurementMethod end
struct CorrelationMeasurement <: MeasurementMethod end
struct EntropyMeasurement <: MeasurementMethod end
const no_measurement = NoMeasurement()

abstract type VisualizationMethod <: MeasurementMethod end
struct NoFigure <: VisualizationMethod end
struct LiveFigure <: VisualizationMethod end
const no_figure = NoFigure()
const live_figure = LiveFigure()

abstract type UpdateAlgorithm <: MonteCarloMethod end

abstract type AcceptanceRule <: UpdateAlgorithm end
function accept_proposal(rule::AcceptanceRule, dE::Float64, kbT::Float64)
    return error("accept_proposal not implemented for $(typeof(rule)).
           available rules are $(AVAILABLE_ACCEPTANCE_RULES...).
           ")
end
struct MetropolisRule <: AcceptanceRule end
function accept_proposal(::MetropolisRule, dE::Float64, kbT::Float64)
    if dE <= 0
        return true
    elseif kbT <= 0
        return false
    else
        return rand() < exp(-dE / kbT)
    end
end
struct GlauberRule <: AcceptanceRule end
function accept_proposal(::GlauberRule, dE::Float64, kbT::Float64)
    if kbT <= 0
        return dE < 0
    else
        return rand() < 1 / (1 + exp(dE / kbT))
    end
end
struct SuwaTodoRule <: AcceptanceRule end
function accept_proposal(::SuwaTodoRule, dE::Float64, kbT::Float64)
    return accept_proposal(MetropolisRule(), dE, kbT)
end
abstract type SiteSelectionMethod <: MonteCarloMethod end
function update_step!(
    grids::AbstractVector{Int},
    lat::Lattice,
    kbT::Float64,
    model::AbstractModel,
    alg::UpdateAlgorithm;
    method=sweep,
)
    return update_step!(grids, lat, kbT, model, alg, method)
end
struct Sweep <: MonteCarloMethod end
const sweep = Sweep()
function update_step!(
    grids::AbstractVector{Int},
    lat::Lattice,
    kbT::Float64,
    model::AbstractModel,
    alg::UpdateAlgorithm,
    ::Sweep,
)
    for site in 1:(lat.N)
        update_single_site!(grids, lat, site, kbT, model, alg)
    end
end
struct RandomChoice <: MonteCarloMethod end
const random = RandomChoice()
function update_step!(
    grids::AbstractVector{Int},
    lat::Lattice,
    kbT::Float64,
    model::AbstractModel,
    alg::UpdateAlgorithm,
    ::RandomChoice,
)
    for _ in 1:(lat.N)
        site = rand(1:(lat.N))
        update_single_site!(grids, lat, site, kbT, model, alg)
    end
end

struct Metropolis <: UpdateAlgorithm end
const metropolis = Metropolis()
function update_single_site!(
    grids::AbstractVector{Int},
    lat::Lattice,
    site::Int,
    kbT::Float64,
    model::IsingModel,
    ::Metropolis,
)
    dE = -2 * local_hamiltonian(grids, lat, site, model)
    if dE <= 0 || rand() < exp(-dE / kbT)
        grids[site] *= -1
    end
end

struct Kawasaki <: UpdateAlgorithm end
struct HeatBath <: UpdateAlgorithm end
struct WolffCluster <: UpdateAlgorithm end
const heat_bath = HeatBath()
const wolff = WolffCluster()

# visualization methods
const p_init = plot(;
    aspect_ratio=:equal, grid=false, axis=false, ticks=false, legend=:bottomright
)
function find_marker_size(lat::Lattice; ms_scale=80.0)
    min_dist = 0.0
    if isempty(lat.bonds)
        min_dist = 1.0
    else
        min_dist = minimum([
            norm(lat.positions[b.src] - lat.positions[b.dst]) for b in lat.bonds
        ])
    end
    xs = [p[1] for p in lat.positions]
    ys = [p[2] for p in lat.positions]
    area = [(maximum(xs) - minimum(xs)), (maximum(ys) - minimum(ys))]
    scaling = min_dist / norm(area)

    marker_size = ms_scale * scaling
    return marker_size
end
function visualize_bonds(p, lat::Lattice)
    threhold = 1.5 * maximum(norm.(lat.unit_cell.basis))
    seg_x, seg_y = Float64[], Float64[]
    for bond in lat.bonds
        src_pos = lat.positions[bond.src]
        dst_pos = lat.positions[bond.dst]
        if norm(dst_pos - src_pos) < threhold
            push!(seg_x, src_pos[1], dst_pos[1], NaN)
            push!(seg_y, src_pos[2], dst_pos[2], NaN)
        end
    end
    return plot!(p, seg_x, seg_y; color=:black, lw=1.0, label="")
end
function plot_spins(
    p, lat::Lattice, spins::AbstractVector; ms_scale=find_marker_size(lat), title_str=""
)
    marker_size = ms_scale * 3.0
    xs = [p[1] for p in lat.positions]
    ys = [p[2] for p in lat.positions]

    colors = [spins[i] > 0 ? :red : :blue for i in 1:(lat.N)]
    scatter!(
        p, xs, ys; ms=marker_size, mc=colors, markerstrokewidth=0, label="", title=title_str
    )
    return p
end
function visualize_realtime!(p; sleep_time=0.001)
    IJulia.clear_output(true)
    display(p)
    sleep(sleep_time)
    return nothing
end
function make_animation(p_list; fps=5, filename="ising_simulation.gif")
    anim = @animate for p in p_list
        plot(p)
    end
    return gif(anim, filename; fps=fps)
end
function make_combined_gif(p_list; filename="simulation.gif", fps=10, layout=nothing) end
