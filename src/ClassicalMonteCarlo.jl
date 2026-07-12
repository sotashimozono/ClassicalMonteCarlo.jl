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
include("algorithm/wang-landau.jl")
include("algorithm/heatbath.jl")
include("algorithm/worm.jl")
include("algorithm/nfoldway.jl")
include("algorithm/exchange-montecarlo.jl")
include("algorithm/reweighting.jl")
include("algorithm/wham.jl")
include("algorithm/multicanonical.jl")
include("algorithm/annealing.jl")
include("measurement/correlation_length.jl")
include("measurement/thermodynamics.jl")
include("measurement/autocorrelation.jl")
include("measurement/correlation_function.jl")
include("measurement/resampling.jl")
include("measurement/blocking.jl")

include("utils/paths.jl")
include("utils/visualize.jl")

end # module ClassicalMonteCarlo
