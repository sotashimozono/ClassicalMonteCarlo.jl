module ClassicalMonteCarloDataFramesExt

# `to_dataframe` output. Loaded automatically when `DataFrames` is available;
# provides the method for the generic function declared in
# `ClassicalMonteCarlo/src/core/observers.jl`.

using DataFrames: DataFrame
using ClassicalMonteCarlo: FunctionObserver
import ClassicalMonteCarlo: to_dataframe

function to_dataframe(obs::FunctionObserver)
    return DataFrame(:Step => obs.steps, Symbol(obs.name) => obs.history)
end

end # module
