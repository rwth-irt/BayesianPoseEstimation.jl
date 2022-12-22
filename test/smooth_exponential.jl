# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# WARN Do not run this if you want Revise to work
include("../src/MCMCDepth.jl")
using .MCMCDepth

using BenchmarkTools
using MCMCDepth
using Plots
using SpecialFunctions
using StatsFuns

plotly()
MCMCDepth.diss_defaults()
r = -0.1:0.01:1.5

accurate_exp(θ, σ, min, max, z) = exp(-z / θ + (σ / θ)^2 / 2) / (θ * (exp(-min / θ) - exp(-max / θ)))
function accurate_erf(θ, σ, min, max, z)
    sqrt2σ = sqrt2 * σ
    common = σ / (sqrt2 * θ) - z / sqrt2σ
    lower = min / sqrt2σ
    upper = max / sqrt2σ
    erf(common + lower, common + upper) / 2
end
full_pdf(θ, σ, min, max, z) = accurate_exp(θ, σ, min, max, z) * accurate_erf(θ, σ, min, max, z)

accurate_logexp(θ, σ, min, max, z) = (-z / θ + (σ / θ)^2 / 2) - log(θ) - log(exp(-min / θ) - exp(-max / θ))
function accurate_logerf(θ, σ, min, max, z)
    sqrt2σ = sqrt2 * σ
    common = σ / (sqrt2 * θ) - z / sqrt2σ
    lower = min / sqrt2σ
    upper = max / sqrt2σ
    loghalf + logerf(common + lower, common + upper)
end
full_logpdf(θ, σ, min, max, z) = accurate_logexp(θ, σ, min, max, z) + accurate_logerf(θ, σ, min, max, z)

θ = 0.8f0
σ = 0.1f0
mini = 0.0f0
maxi = 1.1f0

dist = SmoothExponential(θ, σ, mini, maxi)

# Check whether the generative model matches the likelihood
N = 10_000
histogram(rand(dist, N); normalize=true);
plot!(r, [full_pdf(θ, σ, mini, maxi, z) for z in r], linewidth=2.5)

# Check whether the logdensity is correct
plot(r, [full_logpdf(θ, σ, mini, maxi, z) for z in r]);
plot!(r, [log(full_pdf(θ, σ, mini, maxi, z)) for z in r]);
plot!(r, [logdensityof(dist, z) for z in r])

@benchmark logdensityof(dist, 0.9f0)
@benchmark full_logpdf(θ, σ, mini, maxi, 0.9f0)
@benchmark full_pdf(θ, σ, mini, maxi, 0.9f0)
