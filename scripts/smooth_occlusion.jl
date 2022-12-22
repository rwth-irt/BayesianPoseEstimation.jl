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

# NOTE for upper bound only, when σ ≪ min_depth, StatsFuns.jl has some extra numerical stability implementations 
performant_erf(θ::T, σ::T, μ::T, z::T) where {T} = normccdf(μ + σ^2 / θ, σ, z)
plot!(r, [accurate_erf(1.0, 0.1, 1.0, z) for z in r])

# Exponential part
# TODO (σ / θ)^2 << z / θ
occ_exp(θ::T, σ::T, μ::T, z::T) where {T} = exp(-z / θ + (σ / θ)^2 / 2) / (θ * (1 - exp(-μ / θ)))
plot(r, [occ_exp(1.0, 0.02, 1.0, z) for z in r]);
plot!(r, [occ_exp(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [occ_exp(1.0, 0.5, 1.0, z) for z in r])

# Full function
full_erf(θ::T, σ::T, μ::T, z::T) where {T} = occ_exp(θ, σ, μ, z) * accurate_erf(θ, σ, μ, z)
# TODO this mean or the other one?
full_partial(θ::T, σ::T, μ::T, z::T) where {T} = occ_exp(θ, σ, μ, z) * performant_erf(θ, σ, μ, z)
plot(r, [full_erf(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [full_partial(1.0, 0.1, 1.0, z) for z in r])

# Logarithmic
function accurate_logerf(θ::T, σ::T, μ::T, z::T) where {T}
    sqrt2σ = sqrt2 * σ
    common = σ / (sqrt2 * θ) - z / sqrt2σ
    lower = zero(T) / sqrt2σ
    upper = μ / sqrt2σ
    loghalf + logerf(common + lower, common + upper)
end
occ_logpartial(θ::T, σ::T, μ::T, z::T) where {T} = normlogccdf(μ + σ^2 / θ, σ, z)

plot(r, [accurate_logerf(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [log(accurate_erf(1.0, 0.1, 1.0, z)) for z in r]);
plot!(r, [occ_logpartial(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [log(performant_erf(1.0, 0.1, 1.0, z)) for z in r])

# Logarithmic Exponential part
# NOTE μ/θ won't be that small that I need the accuracy of log1mexp
occ_logexp(θ::T, σ::T, μ::T, z::T) where {T} = (-z / θ + (σ / θ)^2 / 2) - log(θ) - log(1 - exp(-μ / θ))
plot(r, [occ_logexp(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [log(occ_exp(1.0, 0.02, 1.0, z)) for z in r])

# Logarithmic full
full_logerf(θ::T, σ::T, μ::T, z::T) where {T} = occ_logexp(θ, σ, μ, z) + accurate_logerf(θ, σ, μ, z)
full_logpartial(θ::T, σ::T, μ::T, z::T) where {T} = occ_logexp(θ, σ, μ, z) + occ_logpartial(θ, σ, μ, z) # normccdf(μ - λ * σ^2, σ, z)
plot(r, [full_logerf(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [full_logpartial(1.0, 0.1, 1.0, z) for z in r]);
plot!(r, [log(full_erf(1.0, 0.1, 1.0, z)) for z in r])


# NOTE neglecting the lower bound can result in 5x speedup on CPU, GPU almost no difference. Moreover, for small σ=0.01 there is no speedup.
@benchmark accurate_erf(1.0, 0.2, 1.0, 1.1)
@benchmark performant_erf(1.0, 0.2, 1.0, 1.1)
using CUDA
A = 2 .* CUDA.rand(100)
@benchmark CUDA.@sync accurate_erf.(1.0f0, 0.2f0, 1.0f0, A)
@benchmark CUDA.@sync performant_erf.(1.0f0, 0.2f0, 1.0f0, A)
# NOTE About half the speed of the non-logarithmic version
@benchmark accurate_logerf(1.0, 0.2, 1.0, 1.1)
@benchmark occ_logpartial(1.0, 0.2, 1.0, 1.1)
# NOTE The exponential term is not that expensive
@benchmark full_erf(1.0, 0.2, 1.0, 1.1)
@benchmark full_partial(1.0, 0.2, 1.0, 1.1)
@benchmark full_logerf(1.0, 0.01, 1.0, 1.1)
@benchmark full_logpartial(1.0, 0.01, 1.0, 1.1)

using Random
function rand_smooth(rng::AbstractRNG, θ::T, σ::T, μ::T) where {T}
    x = rand(rng, truncated(KernelExponential(θ), zero(T), μ))
    rand(rng, KernelNormal(x, σ))
end
N = 100_000
res = [rand_smooth(Random.default_rng(), 1.0, 0.1, 1.0) for _ in 1:N]
histogram(res; normalize=true)