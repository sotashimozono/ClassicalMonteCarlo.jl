module ClassicalMonteCarlo

using Lattice2D, Pkg
using Random, Statistics, LinearAlgebra
using Plots, ColorSchemes
using CSV, DataFrames

include("core/abstracttypes.jl")
include("core/observers.jl")
include("core/interfaces.jl")
include("core/alias.jl")

include("model/ising.jl")
include("model/pottsmodel.jl")
include("model/xymodel.jl")

include("algorithm/localupdates.jl")
include("algorithm/wolff.jl")
include("algorithm/swendsen-wang.jl")
include("algorithm/exchange-montecarlo.jl")

include("utils/paths.jl")
include("utils/visualize.jl")

end # module ClassicalMonteCarlo
