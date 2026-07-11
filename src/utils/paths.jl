const PATH_BASE = pkgdir(ClassicalMonteCarlo)
const PATHS_DATA = joinpath(PATH_BASE, "data")
const PATHS_FIGURES = joinpath(PATH_BASE, "docs", "src", "assets")

# function get_path(lat::AbstractLattice, model::AbstractModel, alg::AbstractAlgorithm ; base_dir=PATHS_BASE)
#     return nothing
# end
