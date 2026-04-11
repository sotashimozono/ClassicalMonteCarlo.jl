"""
    plot_setup(lat::AbstractLattice; title="") -> (Plots.Plot, Float64)

Initializes the plotting environment for a lattice simulation.

# Returns
- A `Plots.Plot` object with the lattice bonds already drawn.
- A calculated `marker_size` optimized for the lattice scale.
"""
const p_init = plot(;
    aspect_ratio=:equal, grid=false, axis=false, ticks=false, legend=:bottomright
)

function plot_setup(lat::AbstractLattice; title="")
    ms = find_marker_size(lat)
    p = plot(;
        aspect_ratio=:equal, grid=false, axis=false, ticks=false, legend=false, title=title
    )
    visualize_bonds(p, lat)
    return p, ms
end
"""
    find_marker_size(lat::AbstractLattice; ms_scale=80.0) -> Float64

Heuristically determines an appropriate marker size for plotting sites.

# Logic
It calculates the minimum distance between connected sites (bond length) and scales it 
relative to the total area of the lattice. This ensures markers don't overlap too much
on dense lattices or appear too small on sparse ones.
"""
function find_marker_size(lat::AbstractLattice; ms_scale=80.0)
    all_bonds = collect(bonds(lat))
    min_dist = 0.0
    if isempty(all_bonds)
        min_dist = 1.0
    else
        min_dist = minimum([
            norm(position(lat, b.i) - position(lat, b.j)) for b in all_bonds
        ])
    end
    N = num_sites(lat)
    xs = [position(lat, i)[1] for i in 1:N]
    ys = [position(lat, i)[2] for i in 1:N]
    area = [(maximum(xs) - minimum(xs)), (maximum(ys) - minimum(ys))]
    scaling = min_dist / norm(area)
    marker_size = ms_scale * scaling
    return marker_size
end
"""
    visualize_bonds(p, lat::AbstractLattice)

Draws the lattice connections (edges) on the plot `p`.

# Periodic Boundary Conditions
It includes a threshold check (`1.5 * max_basis_norm`) to prevent drawing lines across the entire plot
when sites are connected via periodic boundaries. Bonds longer than this threshold are skipped.
"""
function visualize_bonds(p, lat::AbstractLattice)
    A = basis_vectors(lat)
    threhold = 1.5 * max(norm(A[:, 1]), norm(A[:, 2]))
    seg_x, seg_y = Float64[], Float64[]
    for bond in bonds(lat)
        src_pos = position(lat, bond.i)
        dst_pos = position(lat, bond.j)
        if norm(dst_pos - src_pos) < threhold
            push!(seg_x, src_pos[1], dst_pos[1], NaN)
            push!(seg_y, src_pos[2], dst_pos[2], NaN)
        end
    end
    return plot!(p, seg_x, seg_y; color=:black, lw=1.0, label="")
end
"""
    plot_state!(p, lat, grids, model; marker_size=10.0)

Plots the current state of the spins onto the existing plot `p`.

# Dispatch Behavior
- **Discrete Models (Ising, Potts)**: Uses `scatter!` to draw colored dots at lattice sites.
- **Continuous Models (XY)**: Uses `quiver!` to draw arrows representing the spin direction.
"""
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
"""
    get_state_colors(model, grids) -> Vector{Symbol} | Vector{Color}

Determines the color of each site based on its spin state.

# Implementations
- `IsingModel`: Maps up (+1) to `:red` and down (-1) to `:blue`.
- `PottsModel`: Maps states `1..q` to a categorical color gradient (`:tab10`).
"""
function get_state_colors(model::IsingModel, grids::AbstractVector)
    return [s > 0 ? :red : :blue for s in grids]
end
function get_state_colors(model::PottsModel, grids::AbstractVector)
    q = model.q
    palette = cgrad(:tab10, q; categorical=true)

    return [palette[(s - 1) / (q - 1)] for s in grids]
end
# XY model : visualize as arrows
"""
    plot_state!(p, lat, grids, model::XYModel; marker_size=10.0)

Specialized visualization for the XY Model.
Instead of colored dots, it draws arrows (quivers) at each site.
The direction of the arrow corresponds to the angle \\theta of the spin.
"""
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
"""
    visualize_snapshot(grids, lat, model) -> Plots.Plot

A high-level function to generate a complete visual snapshot of the current system state.
It sets up the plot, draws bonds, and overlays the current spin configuration.
"""
function visualize_snapshot(grids, lat, model)
    p, ms = plot_setup(lat; title="(typeof(model)) Step")
    plot_state!(p, lat, grids, model; marker_size=ms)
    return p
end

export plot_setup,
    find_marker_size, visualize_bonds, plot_state!, visualize_snapshot, get_state_colors
