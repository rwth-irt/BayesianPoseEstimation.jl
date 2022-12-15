# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

# Resources:
# Quaternion from Bivectors
# https://probablydance.com/2017/08/05/intuitive-quaternions/
# Sphere sampling
# http://corysimon.github.io/articles/uniformdistn-on-sphere/
# https://mathworld.wolfram.com/SpherePointPicking.html

using Bijectors
using DensityInterface
using Distributions
using LinearAlgebra
using Quaternions
using Random

"""
    robust_normalize(q)
Compared to the implementation in Quaternions.jl, this implementation takes care of the re-normalization and avoiding divisions by zero.
Eq. (44) in J. Sola, „Quaternion kinematics for the error-state KF“.
"""
function robust_normalize(q::Quaternion{T}) where {T}
    a = abs(q)
    if iszero(a)
        Quaternion(one(T), zero(T), zero(T), zero(T), true)
    else
        # Rotations.jl likes normalized quaternions → do not ignore small deviations from 1
        q = q / a
        Quaternion(q.s, q.v1, q.v2, q.v3, true)
    end
end

"""
    approx_qrotation(x, y, z)
Approximate conversion of small rotation vectors ϕ = (x, y, z) to a quaternion.
≈ double the speed of Quaternions.qrotation
Eq. (193) in J. Sola, „Quaternion kinematics for the error-state KF“
"""
approx_qrotation(x, y, z) = Quaternion(1, x / 2, y / 2, z / 2) |> robust_normalize

"""
    QuaternionDistribution
Allows true uniform sampling of 3D rotations.
Normalization requires scalar indexing, thus CUDA is not supported.
"""
struct QuaternionDistribution{T} <: AbstractKernelDistribution{Quaternion{T},Continuous} end

QuaternionDistribution(::Type{T}=Float32) where {T} = QuaternionDistribution{T}()

# Quaternions lie on the surface of a 4D hypersphere with radius 1. Due to to the duality of quaternions, it is only the surface of the half sphere.
# https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html 
const quat_logp = -log(π^2)

Distributions.logpdf(::QuaternionDistribution{T}, x::Quaternion) where {T} = T(quat_logp)

# Random Interface

Base.rand(rng::AbstractRNG, ::QuaternionDistribution{T}) where {T} = Quaternion(randn(rng, T), randn(rng, T), randn(rng, T), randn(rng, T)) |> robust_normalize

# Bijectors
Bijectors.bijector(::QuaternionDistribution) = ZeroIdentity()
Bijectors.logabsdetjac(::ZeroIdentity, ::Union{Quaternion{T},AbstractArray{<:Quaternion{T}}}) where {T} = zero(T)

"""
    AdditiveQuaternion(q)
Wraps a quaternion and remaps `+` to `*` and `-` to `/` to support additive proposal interfaces.
"""
struct AdditiveQuaternion{T}
    q::Quaternion{T}
end

Broadcast.broadcastable(q::AdditiveQuaternion) = Ref(q)
Base.:+(a::Quaternion, b::AdditiveQuaternion) = robust_normalize(a * b.q)
Base.:-(a::Quaternion, b::AdditiveQuaternion) = robust_normalize(a / b.q)

"""
    QuaternionPerturbation
Taylor approximation for small perturbation as described in:
J. Sola, „Quaternion kinematics for the error-state KF“, Laboratoire dAnalyse et dArchitecture des Systemes-Centre national de la recherche scientifique (LAAS-CNRS), Toulouse, France, Tech. Rep, 2012.
"""
struct QuaternionPerturbation{T} <: AbstractKernelDistribution{AdditiveQuaternion{T},Continuous}
    σ_x::T
    σ_y::T
    σ_z::T
end

QuaternionPerturbation(σ=0.01f0::Real) = QuaternionPerturbation(σ, σ, σ)

Distributions.logpdf(dist::QuaternionPerturbation{T}, x::Quaternion) where {T} = logpdf(KernelNormal(zero(T), dist.σ_x), 2 * x.v1) + logpdf(KernelNormal(zero(T), dist.σ_y), 2 * x.v2) + logpdf(KernelNormal(zero(T), dist.σ_z), 2 * x.v3)
Distributions.logpdf(dist::QuaternionPerturbation, x::AdditiveQuaternion) = logpdf(dist, x.q)

Base.rand(rng::AbstractRNG, dist::QuaternionPerturbation{T}) where {T} = approx_qrotation(rand(rng, KernelNormal(0, dist.σ_x)), rand(rng, KernelNormal(0, dist.σ_y)), rand(rng, KernelNormal(0, dist.σ_z))) |> AdditiveQuaternion

# Bijectors
Bijectors.bijector(::QuaternionPerturbation) = ZeroIdentity()
