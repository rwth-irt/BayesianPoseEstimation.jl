# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using StatsFuns
using SpecialFunctions

"""
    SmoothExponential
Smooth truncated exponential distribution.
"""
struct SmoothExponential{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    θ::T
    σ::T
    # TODO revisit min_depth and ValidPixel. With the normal in the mixture, min_depth will be ignored
    min::T
    max::T
end
# TODO SmoothExponential(::Type{T}=Float32) where {T} = SmoothExponential{T}(1.0)

Base.show(io::IO, dist::SmoothExponential{T}) where {T} = print(io, "SmoothExponential{$(T)}, θ: $(dist.θ),  σ: $(dist.σ), min: $(dist.min), max: $(dist.max)")

# Accurate uses lower and upper bound
acc_logerf(d::SmoothExponential{T}, x) where {T} = loghalf + logerf(
    (d.min + d.σ^2 / d.θ - x) / (sqrt2 * d.σ),
    (d.max + d.σ^2 / d.θ - x) / (sqrt2 * d.σ))
# NOTE for upper bound only, when σ ≪ min_depth, StatsFuns.jl has some extra numerical stability implementations. On CPU, it can be 5x faster, on GPU almost no difference
perf_logerf(d::SmoothExponential, x) = normlogccdf(d.max + d.σ^2 / d.θ, d.σ, x)

acc_normalization(d::SmoothExponential) = -log(exp(-d.min / d.θ) - exp(-d.max / d.θ))
perf_normalization(d::SmoothExponential) = -log1p(-exp(-d.max / d.θ))

acc_factor(d::SmoothExponential, x) = (-x / d.θ + (d.σ / d.θ)^2 / 2) - log(d.θ) + acc_normalization(d)
perf_factor(d::SmoothExponential, x) = (-x / d.θ + (d.σ / d.θ)^2 / 2) - log(d.θ) + perf_normalization(d)

# Distributions.logpdf(dist::SmoothExponential{T}, x) where {T} = insupport(dist, x) ? perf_factor(dist, x) + perf_logerf(dist, x) : typemin(T)

Distributions.logpdf(dist::SmoothExponential{T}, x) where {T} = insupport(dist, x) ? acc_factor(dist, x) + acc_logerf(dist, x) : typemin(T)

# Exponential convoluted with normal: Sample from exponential and then add noise of normal
# TODO test if the plots match
function Base.rand(rng::AbstractRNG, dist::SmoothExponential{T}) where {T}
    μ = rand(rng, truncated(KernelExponential(dist.θ), dist.min, dist.max))
    rand(rng, KernelNormal(μ, dist.σ))
end

# Compared to a regular exponential distribution, this one is defined on ℝ 😃
Base.maximum(::SmoothExponential{T}) where {T} = typemax(T)
Base.minimum(::SmoothExponential{T}) where {T} = typemin(T)
Bijectors.bijector(::SmoothExponential) = ZeroIdentity()
Distributions.insupport(dist::SmoothExponential, x::Real) = true
# TODO do I want support truncated via normlogcdf and invlogcdf ?