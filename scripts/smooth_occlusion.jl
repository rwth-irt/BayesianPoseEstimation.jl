# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using BenchmarkTools
using MCMCDepth
using Plots
using SpecialFunctions
using StatsFuns

plotly()
MCMCDepth.diss_defaults()
r = -0.1:0.01:1.5

# Error function part
# NOTE for both bounds, erf has a special implementation which makes it numerically stable
accurate_erf(θ::T, σ::T, μ::T, z::T) where {T} = inv(2) * erf(
    (σ^2 / θ - z) / (sqrt2 * σ),
    (μ + σ^2 / θ - z) / (sqrt2 * σ))
# NOTE for upper bound only, when σ ≪ min_depth, StatsFuns.jl has some extra numerical stability implementations 
performance_erf(θ::T, σ::T, μ::T, z::T) where {T} = normccdf(μ + σ^2 / θ, σ, z)
plot(r, [accurate_erf(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [performance_erf(1.0, 0.1, 1.0, z) for z in r])

# Exponential part
# TODO (σ / θ)^2 << z / θ
occ_exp(θ::T, σ::T, μ::T, z::T) where {T} = exp(-z / θ + (σ / θ)^2 / 2) / (θ * (1 - exp(-μ / θ)))
plot(r, [occ_exp(1.0, 0.02, 1.0, z) for z in r]);
plot!(r, [occ_exp(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [occ_exp(1.0, 0.5, 1.0, z) for z in r])

# Full function
full_erf(θ::T, σ::T, μ::T, z::T) where {T} = occ_exp(θ, σ, μ, z) * accurate_erf(θ, σ, μ, z)
# TODO this mean or the other one?
full_partial(θ::T, σ::T, μ::T, z::T) where {T} = occ_exp(θ, σ, μ, z) * performance_erf(θ, σ, μ, z)
plot(r, [full_erf(1.0, 0.01, 1.0, z) for z in r]);
plot!(r, [full_partial(1.0, 0.01, 1.0, z) for z in r])

# Logarithmic
occ_logerf(θ::T, σ::T, μ::T, z::T) where {T} = loghalf + logerf(
    (σ^2 / θ - z) / (sqrt2 * σ),
    (μ + σ^2 / θ - z) / (sqrt2 * σ))
occ_logpartial(θ::T, σ::T, μ::T, z::T) where {T} = normlogccdf(μ + σ^2 / θ, σ, z)

plot(r, [occ_logerf(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [log(accurate_erf(1.0, 0.1, 1.0, z)) for z in r]);
plot!(r, [occ_logpartial(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [log(performance_erf(1.0, 0.1, 1.0, z)) for z in r])

# Logarithmic Exponential part
# NOTE μ/θ won't be that small that I need the accuracy of log1mexp
occ_logexp(θ::T, σ::T, μ::T, z::T) where {T} = (-z / θ + (σ / θ)^2 / 2) - log(θ) - log(1 - exp(-μ / θ))
plot(r, [occ_logexp(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [log(occ_exp(1.0, 0.02, 1.0, z)) for z in r])

# Logarithmic full
full_logerf(θ::T, σ::T, μ::T, z::T) where {T} = occ_logexp(θ, σ, μ, z) + occ_logerf(θ, σ, μ, z)
full_logpartial(θ::T, σ::T, μ::T, z::T) where {T} = occ_logexp(θ, σ, μ, z) + occ_logpartial(θ, σ, μ, z) # normccdf(μ - λ * σ^2, σ, z)
plot(r, [full_logerf(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [full_logpartial(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [log(full_erf(1.0, 0.1, 1.0, z)) for z in r])


# NOTE neglecting the lower bound can result in 5x speedup on CPU, GPU almost no difference. Moreover, for small σ=0.01 there is no speedup.
@benchmark accurate_erf(1.0, 0.2, 1.0, 1.1)
@benchmark performance_erf(1.0, 0.2, 1.0, 1.1)
# NOTE About half the speed of the non-logarithmic version
@benchmark occ_logerf(1.0, 0.2, 1.0, 1.1)
@benchmark occ_logpartial(1.0, 0.2, 1.0, 1.1)
# NOTE The exponential term is not that expensive
@benchmark full_erf(1.0, 0.2, 1.0, 1.1)
@benchmark full_partial(1.0, 0.2, 1.0, 1.1)
@benchmark full_logerf(1.0, 0.01, 1.0, 1.1)
@benchmark full_logpartial(1.0, 0.01, 1.0, 1.1)
