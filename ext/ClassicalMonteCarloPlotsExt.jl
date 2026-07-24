module ClassicalMonteCarloPlotsExt

# Lattice / spin-configuration visualisation. Loaded automatically when both
# `Plots` and `ColorSchemes` are available; provides the methods for the
# generic functions declared in `ClassicalMonteCarlo/src/utils/visualize.jl`.

using Plots, ColorSchemes
using LinearAlgebra: norm
using LatticeCore: AbstractLattice, bonds, position, num_sites, basis_vectors
using ClassicalMonteCarlo: AbstractModel, IsingModel, PottsModel, XYModel
import ClassicalMonteCarlo:
    plot_setup,
    find_marker_size,
    visualize_bonds,
    plot_state!,
    visualize_snapshot,
    get_state_colors

function plot_setup(lat::AbstractLattice; title="")
    ms = find_marker_size(lat)
    p = plot(;
        aspect_ratio=:equal, grid=false, axis=false, ticks=false, legend=false, title=title
    )
    visualize_bonds(p, lat)
    return p, ms
end

function find_marker_size(lat::AbstractLattice; ms_scale=80.0)
    all_bonds = collect(bonds(lat))
    min_dist = if isempty(all_bonds)
        1.0
    else
        minimum([norm(position(lat, b.i) - position(lat, b.j)) for b in all_bonds])
    end
    N = num_sites(lat)
    xs = [position(lat, i)[1] for i in 1:N]
    ys = [position(lat, i)[2] for i in 1:N]
    area = [(maximum(xs) - minimum(xs)), (maximum(ys) - minimum(ys))]
    return ms_scale * (min_dist / norm(area))
end

function visualize_bonds(p, lat::AbstractLattice)
    A = basis_vectors(lat)
    threshold = 1.5 * max(norm(A[:, 1]), norm(A[:, 2]))
    seg_x, seg_y = Float64[], Float64[]
    for bond in bonds(lat)
        src_pos = position(lat, bond.i)
        dst_pos = position(lat, bond.j)
        if norm(dst_pos - src_pos) < threshold
            push!(seg_x, src_pos[1], dst_pos[1], NaN)
            push!(seg_y, src_pos[2], dst_pos[2], NaN)
        end
    end
    return plot!(p, seg_x, seg_y; color=:black, lw=1.0, label="")
end

function get_state_colors(::IsingModel, grids::AbstractVector)
    return [s > 0 ? :red : :blue for s in grids]
end
function get_state_colors(model::PottsModel, grids::AbstractVector)
    q = model.q
    palette = cgrad(:tab10, q; categorical=true)
    return [palette[(s - 1) / (q - 1)] for s in grids]
end

function plot_state!(
    p::Plots.Plot,
    lat::AbstractLattice,
    grids::AbstractVector,
    model::AbstractModel;
    marker_size=10.0,
)
    xs = [position(lat, i)[1] for i in 1:num_sites(lat)]
    ys = [position(lat, i)[2] for i in 1:num_sites(lat)]
    colors = get_state_colors(model, grids)
    return scatter!(p, xs, ys; ms=marker_size, mc=colors, markerstrokewidth=0, label="")
end

function plot_state!(
    p::Plots.Plot,
    lat::AbstractLattice,
    grids::AbstractVector,
    model::XYModel;
    marker_size=10.0,
)
    xs = [position(lat, i)[1] for i in 1:num_sites(lat)]
    ys = [position(lat, i)[2] for i in 1:num_sites(lat)]
    u = cos.(grids) .* (marker_size * 0.01)
    v = sin.(grids) .* (marker_size * 0.01)
    return quiver!(p, xs, ys; quiver=(u, v), color=:black)
end

function visualize_snapshot(grids, lat, model)
    p, ms = plot_setup(lat; title="$(typeof(model)) Step")
    plot_state!(p, lat, grids, model; marker_size=ms)
    return p
end

end # module
