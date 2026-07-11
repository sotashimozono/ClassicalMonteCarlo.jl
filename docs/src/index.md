# ClassicalMonteCarlo.jl

Overview of the 2D Lattice Monte Carlo simulation package.

**ClassicalMonteCarlo.jl** is an example application of the [Lattice2D](https://github.com/sotashimozono/Lattice2D.jl) package. It provides implementations of various classical lattice systems. By adhering to specific interfaces, you can easily implement custom models or update algorithms.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/sotashimozono/ClassicalMonteCarlo.jl")
```

## Quick Start

```julia
using Lattice2D
using ClassicalMonteCarlo
using Random

# 1. Setup Lattice and Model
lat = build_lattice(Honeycomb, 10, 10)
model = IsingModel(J=1.0, h=0.0)

# 2. Initialize State
rng = Random.default_rng()
# Random initial configuration (+1 or -1)
grids = rand(rng, [-1, 1], lat.N)

# 3. Define Algorithm and Observer
alg = LocalUpdate(rule=Metropolis(), proposal=SpinFlip())
obs = FunctionObserver("Magnetization", measure_magnetization)

# 4. Simulation Loop
kbT = 2.269 # Critical temperature
n_steps = 1000

for step in 1:n_steps
    update_step!(rng, grids, lat, model, alg; kbT=kbT)
    observe!(obs, grids, lat, kbT, model, step)
end

df = to_dataframe(obs)
display(first(df, 5))
```
