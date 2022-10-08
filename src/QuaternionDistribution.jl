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
Compared to the implementation in Quaternions.jl, this implementation takes care of the re-normalization and avoiding divisions by zero as described by J. Sola in „Quaternion kinematics for the error-state KF“.
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
Bijectors.bijector(::QuaternionDistribution) = Identity{0}()
Bijectors.logabsdetjac(::Identity, ::Quaternion{T}) where {T} = zero(T)
function Bijectors.logabsdetjac(::Identity, x::AbstractArray{<:Quaternion{T}}) where {T}
    res = similar(x, T)
    fill!(res, zero(T))
end

#TEST
"""
    QuaternionPerturbation
Taylor approximation for small perturbation as described in:
J. Sola, „Quaternion kinematics for the error-state KF“, Laboratoire dAnalyse et dArchitecture des Systemes-Centre national de la recherche scientifique (LAAS-CNRS), Toulouse, France, Tech. Rep, 2012.
"""
struct QuaternionPerturbation{T} <: AbstractKernelDistribution{Quaternion{T},Continuous}
    σ_x::T
    σ_y::T
    σ_z::T
end

QuaternionPerturbation(σ=0.01f0::Real) = QuaternionPerturbation(σ, σ, σ)

Distributions.logpdf(dist::QuaternionPerturbation{T}, x::Quaternion) where {T} = logpdf(KernelNormal(zero(T), dist.σ_x), 2 * x.v1) + logpdf(KernelNormal(zero(T), dist.σ_y), 2 * x.v2) + logpdf(KernelNormal(zero(T), dist.σ_z), 2 * x.v2)

Base.rand(rng::AbstractRNG, dist::QuaternionPerturbation{T}) where {T} = Quaternion(1, rand(rng, KernelNormal(0, dist.σ_x)) / 2, rand(rng, KernelNormal(0, dist.σ_y)) / 2, rand(rng, KernelNormal(0, dist.σ_z)) / 2) |> robust_normalize

# Bijectors
Bijectors.bijector(::QuaternionPerturbation) = Bijectors.Identity{0}()


# TEST
"""
    QuaternionProposal
SymmtericPropsal for quaternions: Uses broadcasted (Hamiltonian) product `.*` operator instead of sum `+` and normalizes the result.
"""
struct QuaternionProposal{var_names,T} <: AbstractProposal
    models::IndependentModel{var_names,T}
end

"""
    propose(rng, proposal::QuaternionProposal, sample, dims...)
Inspired by EKFs for IMUs, the perturbation is assumed to be defined in the local frame.
Together with the Hamiltonian convention, the perturbation is a right side multiplication.
"""
propose(rng::AbstractRNG, proposal::QuaternionProposal, sample::Sample, dims...) = map_merge((a, b) -> robust_normalize.(a .* b), sample, rand(rng, proposal.models, dims...))

# symmetric proposal, can be neglected
transition_probability(proposal::QuaternionProposal, new_sample::Sample, ::Sample) = 0.0
