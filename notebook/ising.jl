using Lattice2D
using Random, Statistics, Plots

abstract type MonteCarloMethod end
abstract type MetropolisMethod <: MonteCarloMethod end
struct Sweep <: MetropolisMethod end
struct RandomChoice <: MetropolisMethod end

const sweep = Sweep()
const random = RandomChoice()

abstract type MeasurementMethod <: MonteCarloMethod end

abstract type VisualizationMethod <: MeasurementMethod end
struct NoFigure <: VisualizationMethod end
struct LiveFigure <: VisualizationMethod end

const no_figure = NoFigure()
const live_figure = LiveFigure()

function local_hamiltonian(
    grids::AbstractVector{Int}, lat::Lattice, site::Int; J=1.0, H=0.0
)
    local_h = H * grids[site]
    for neighbor in lat.nearest_neighbors[site]
        local_h -= J * grids[site] * grids[neighbor]
    end
    return local_h
end
function total_hamiltonian(grids::AbstractVector{Int}, lat::Lattice; J=1.0, H=0.0)
    total_h = 0.0
    for site in 1:(lat.N)
        total_h += local_hamiltonian(grids, lat, site; J=J, H=H)
    end
    return total_h / 2
end
function metropolis_step!(
    grids::AbstractVector{Int}, lat::Lattice, kbT::Float64; J=1.0, H=0.0, method=sweep
)
    return metropolis_step!(method, grids, lat, kbT; J=J, H=H)
end
function metropolis_step!(
    ::Sweep, grids::AbstractVector{Int}, lat::Lattice, kbT::Float64; J=1.0, H=0.0
)
    for site in 1:(lat.N)
        dE = -2 * local_hamiltonian(grids, lat, site; J=J, H=H)
        if dE <= 0 || rand() < exp(-dE / kbT)
            grids[site] *= -1
        end
    end
end
function metropolis_step!(
    ::RandomChoice, grids::AbstractVector{Int}, lat::Lattice, kbT::Float64; J=1.0, H=0.0
)
    for _ in 1:(lat.N)
        site = rand(1:(lat.N))
        dE = -2 * local_hamiltonian(grids, lat, site; J=J, H=H)
        if dE <= 0 || rand() < exp(-dE / kbT)
            grids[site] *= -1
        end
    end
end

function Metropolis(
    grids::AbstractVector{Int},
    lat::Lattice,
    kbT::Float64;
    visualize::VisualizationMethod=no_figure,
    kwargs...,
)
    return Metropolis(visualize, grids, lat, kbT; kwargs...)
end
function Metropolis(
    ::NoFigure,
    grids::AbstractVector{Int},
    lat::Lattice,
    kbT::Float64;
    J=1.0,
    H=0.0,
    method=sweep,
    n_thermal=1000,
    n_steps=1000,
    visualize_steps=nothing,
)
    energies = zeros(Float64, n_steps)
    magnetizations = zeros(Float64, n_steps)
    for step in 1:n_thermal
        metropolis_step!(grids, lat, kbT; J=J, H=H, method=method)
    end
    for step in 1:n_steps
        metropolis_step!(grids, lat, kbT; J=J, H=H, method=method)
        E = total_hamiltonian(grids, lat; J=J, H=H) / lat.N
        M = mean(grids)
        energies[step] = E
        magnetizations[step] = M
    end
    return grids, energies, magnetizations
end
function Metropolis(
    ::LiveFigure,
    grids::AbstractVector{Int},
    lat::Lattice,
    kbT::Float64;
    J=1.0,
    H=0.0,
    method=sweep,
    n_thermal=1000,
    n_steps=1000,
    visualize_steps=1000,
)
    energies = zeros(Float64, n_steps)
    magnetizations = zeros(Float64, n_steps)

    n_frames = floor(Int, n_steps / visualize_steps)
    frame_idx = 1
    p_base = visualize_bonds(deepcopy(p_init), lat)
    p_list = Vector{Plots.Plot}(undef, n_frames)

    for step in 1:n_thermal
        metropolis_step!(grids, lat, kbT; J=J, H=H, method=method)
    end
    for step in 1:n_steps
        metropolis_step!(grids, lat, kbT; J=J, H=H, method=method)
        E = total_hamiltonian(grids, lat; J=J, H=H) / lat.N
        M = mean(grids)
        energies[step] = E
        magnetizations[step] = M

        if step % visualize_steps != 0
            continue
        end
        p = deepcopy(p_base)
        p = plot_spins(
            p, lat, grids; title_str=string(lat.topology, " kbT=", kbT, " Step=", step)
        )
        visualize_realtime!(p)
        p_list[frame_idx] = p
        frame_idx += 1
    end
    return grids, energies, magnetizations, p_list
end

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
