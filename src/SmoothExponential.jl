# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using StatsFuns
using SpecialFunctions

"""
    SmoothExponential
Smooth truncated exponential distribution by convolving the exponential with a normal distribution: Smooth = Exp ⋆ Normal
This results in smooth min and max limits and a definition on ℝ instead of ℝ⁺
"""
struct SmoothExponential{T<:Real} <: AbstractKernelDistribution{T,Continuous}
    min::T
    max::T
    θ::T
    σ::T
end

Base.show(io::IO, dist::SmoothExponential{T}) where {T} = print(io, "SmoothExponential{$(T)}, min: $(dist.min), max: $(dist.max), θ: $(dist.θ), σ: $(dist.σ)")

# Accurate version uses lower and upper bound
accurate_normalization(d::SmoothExponential) = -logsubexp(-d.min / d.θ, -d.max / d.θ)
accurate_factor(d::SmoothExponential, x) = (-x / d.θ + (d.σ / d.θ)^2 / 2) - log(d.θ) + accurate_normalization(d)
function accurate_logerf(d::SmoothExponential{T}, x) where {T}
    invsqrt2σ = inv(sqrt2 * d.σ)
    common = d.σ / (sqrt2 * d.θ) - x * invsqrt2σ
    lower = d.min * invsqrt2σ
    upper = d.max * invsqrt2σ
    loghalf + logerf(common + lower, common + upper)
end

# See my (Tim Redick) dissertation for the derivation.
Distributions.logpdf(dist::SmoothExponential{T}, x) where {T} = insupport(dist, x) ? accurate_factor(dist, x) + accurate_logerf(dist, x) : typemin(T)


# Exponential convoluted with normal: Sample from exponential and then add noise of normal
function Base.rand(rng::AbstractRNG, dist::SmoothExponential{T}) where {T}
    μ = rand(rng, truncated(KernelExponential(dist.θ), dist.min, dist.max))
    rand(rng, KernelNormal(μ, dist.σ))
end

# Compared to a regular exponential distribution, this one is defined on ℝ 😃
Base.maximum(::SmoothExponential{T}) where {T} = typemax(T)
Base.minimum(::SmoothExponential{T}) where {T} = typemin(T)
Bijectors.bijector(::SmoothExponential) = ZeroIdentity()
Distributions.insupport(dist::SmoothExponential, x::Real) = true
