# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# Quaternion from Bivectors
# https://probablydance.com/2017/08/05/intuitive-quaternions/

# Sphere sampling
# http://corysimon.github.io/articles/uniformdistn-on-sphere/
# https://mathworld.wolfram.com/SpherePointPicking.html

# Perturbations
# [1] J. Sola, „Quaternion kinematics for the error-state KF“, Laboratoire dAnalyse et dArchitecture des Systemes-Centre national de la recherche scientifique (LAAS-CNRS), Toulouse, France, Tech. Rep, 2012.

# Implementations Rotations.jl uses Quaternion.jl and I think I use Rotations.jl in SciGL
# https://github.com/JuliaGeometry/Quaternions.jl/blob/master/src/Quaternion.jl

using Bijectors
using DensityInterface
using Distributions
using Random

struct QuaternionDistribution{T} <: Distribution{ArrayLikeVariate{1},Continuous} end

QuaternionDistribution(::Type{T}=Float32) where {T} = QuaternionDistribution{T}()

const quat_logp = -log(π^2)

function Distributions.logpdf(::QuaternionDistribution{T}, x::AbstractArray{<:Real}) where {T}
    _, s... = size(x)
    res = similar(x, T, s)
    # Quaternions lie on the surface of a 4D hypersphere. Due to to the duality of quaternions, it is only the surface of the half sphere.
    # https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html 
    fill!(res, T(quat_logp))
end

Distributions.logpdf(::QuaternionDistribution{T}, x::AbstractVector{<:Real}) where {T} = T(quat_logp)

# By default, Distributions.jl disallows logdensityof with multiple samples (Arrays and Matrices). BroadcastedDistribution is inherently designed for multiple samples so allow them explicitly.
DensityInterface.logdensityof(dist::QuaternionDistribution, x) = logpdf(dist, x)
DensityInterface.logdensityof(dist::QuaternionDistribution, x::AbstractMatrix) = logpdf(dist, x)

# Random Interface

"""
    rand(rng, dist, [dims...])
Sample an array from `dist` of size `(size(marginals)..., dims...)`.
The array type is based on the `rng` and the parameter type of the distribution.
"""
function Base.rand(rng::AbstractRNG, dist::QuaternionDistribution{T}, dims::Int...) where {T}
    # could probably be generalized by implementing Base.eltype(AbstractVectorizedDistribution)
    A = array_for_rng(rng, T, 4, dims...)
    rand!(rng, dist, A)
end

"""
    rand!(rng, dist, [dims...])
Mutate the array `A` by sampling from `dist`.
"""
function Random.rand!(rng::AbstractRNG, ::QuaternionDistribution{T}, A::AbstractArray{<:Real}) where {T}
    # Draw from standard normal distribution and normalize components https://imois.in/posts/random-vectors-and-rotations-in-3d/
    rand!(rng, KernelNormal(T), A)
    normalize_dims!(A)
end

# Bijectors
Bijectors.bijector(dist::QuaternionDistribution) = Bijectors.Identity{0}()
