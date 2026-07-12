# Thermodynamic observables from fluctuations. From a Monte-Carlo stream of the
# order parameter m = |Σ_i s_i|/N and the total energy E:
#
#   Binder cumulant   U₄ = 1 − ⟨m⁴⟩ / (c ⟨m²⟩²)     (c = get_binder_coeff),
#   susceptibility    χ  = N (⟨m²⟩ − ⟨m⟩²) / kbT,
#   specific heat     C  = (⟨E²⟩ − ⟨E⟩²) / (kbT² N).
#
# U₄ is dimensionless and size-independent at criticality, so ξ-free Tc location
# comes from U₄(T) curves for different L crossing at Tc; χ and C are the standard
# fluctuation–dissipation responses.

"""
    binder_cumulant(m2, m4; coeff=3.0) -> Float64

Binder fourth-order cumulant U₄ = 1 − ⟨m⁴⟩/(coeff·⟨m²⟩²). `coeff` is the
order-parameter factor ([`get_binder_coeff`](@ref): 3 for a scalar/Ising/Potts
order parameter, 2 for the planar XY vector).
"""
function binder_cumulant(m2::Real, m4::Real; coeff::Real=3.0)
    m2 > 0 || throw(ArgumentError("⟨m²⟩ must be positive"))
    return 1.0 - m4 / (coeff * m2^2)
end
export binder_cumulant

"""
    susceptibility(m_mean, m2_mean, kbT, N) -> Float64

Finite-size magnetic susceptibility χ = N(⟨m²⟩ − ⟨m⟩²)/kbT from per-spin
order-parameter moments (`m_mean` = ⟨m⟩, `m2_mean` = ⟨m²⟩, `N` = number of spins).
"""
susceptibility(m_mean::Real, m2_mean::Real, kbT::Real, N::Int) =
    N * (m2_mean - m_mean^2) / kbT
export susceptibility

"""
    specific_heat(e_mean, e2_mean, kbT, N) -> Float64

Per-spin specific heat C = (⟨E²⟩ − ⟨E⟩²)/(kbT²·N) from total-energy moments.
"""
specific_heat(e_mean::Real, e2_mean::Real, kbT::Real, N::Int) =
    (e2_mean - e_mean^2) / (kbT^2 * N)
export specific_heat

"""
    measure_thermodynamics(rng, grids, lat, model, updater; kbT, sweeps,
                           therm=sweeps÷10, interval=1)
        -> (; energy, mag, mag2, binder, susceptibility, specific_heat)

Sample the order parameter m = [`measure_magnetization`](@ref) and total energy
and return the canonical averages plus the Binder cumulant, susceptibility and
specific heat. `grids` is mutated.
"""
function measure_thermodynamics(
    rng::AbstractRNG,
    grids::AbstractVector,
    lat::AbstractLattice,
    model::AbstractModel,
    updater::UpdateAlgorithm;
    kbT::Float64,
    sweeps::Int,
    therm::Int=sweeps ÷ 10,
    interval::Int=1,
)
    N = num_sites(lat)
    sm = 0.0
    sm2 = 0.0
    sm4 = 0.0
    sE = 0.0
    sE2 = 0.0
    n = 0
    for s in 1:(therm + sweeps)
        update_step!(rng, grids, lat, model, updater; kbT=kbT)
        if s > therm && (s - therm) % interval == 0
            m = measure_magnetization(grids, lat, model)
            E = total_energy(grids, lat, model)
            sm += m
            sm2 += m^2
            sm4 += m^4
            sE += E
            sE2 += E^2
            n += 1
        end
    end
    m1 = sm / n
    m2 = sm2 / n
    m4 = sm4 / n
    e1 = sE / n
    e2 = sE2 / n
    return (;
        energy=e1,
        mag=m1,
        mag2=m2,
        binder=binder_cumulant(m2, m4; coeff=get_binder_coeff(model)),
        susceptibility=susceptibility(m1, m2, kbT, N),
        specific_heat=specific_heat(e1, e2, kbT, N),
    )
end
export measure_thermodynamics
