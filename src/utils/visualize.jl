# Visualisation stubs. The methods live in `ClassicalMonteCarloPlotsExt`, loaded
# automatically when `Plots` and `ColorSchemes` are available. Keeping the
# generic functions (and their docstrings) here means the names are always
# defined and exported, so calling them without `Plots` loaded gives a clear
# MethodError rather than an UndefVar.

"""
    plot_setup(lat::AbstractLattice; title="") -> (Plots.Plot, Float64)

Initialise the plotting environment for a lattice simulation: a `Plots.Plot`
with the lattice bonds drawn, and a `marker_size` scaled to the lattice.

Requires `Plots` (and `ColorSchemes`) — provided by a package extension.
"""
function plot_setup end

"""
    find_marker_size(lat::AbstractLattice; ms_scale=80.0) -> Float64

Heuristic marker size for plotting sites: the minimum bond length scaled by the
lattice extent. Provided by the `Plots` extension.
"""
function find_marker_size end

"""
    visualize_bonds(p, lat::AbstractLattice)

Draw the lattice bonds on plot `p`, skipping periodic wrap-around bonds longer
than `1.5 × max_basis_norm`. Provided by the `Plots` extension.
"""
function visualize_bonds end

"""
    plot_state!(p, lat, grids, model; marker_size=10.0)

Overlay the current spin configuration on plot `p` — coloured dots for
Ising/Potts, arrows for XY. Provided by the `Plots` extension.
"""
function plot_state! end

"""
    get_state_colors(model, grids)

Per-site colours from the spin state (Ising: red/blue; Potts: a categorical
gradient). Provided by the `Plots` extension.
"""
function get_state_colors end

"""
    visualize_snapshot(grids, lat, model) -> Plots.Plot

A complete snapshot of the current system state (bonds + configuration).
Provided by the `Plots` extension.
"""
function visualize_snapshot end

export plot_setup,
    find_marker_size, visualize_bonds, plot_state!, visualize_snapshot, get_state_colors
