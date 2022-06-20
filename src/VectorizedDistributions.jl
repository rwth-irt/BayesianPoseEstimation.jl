# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base.Broadcast: broadcasted, Broadcasted, materialize
using Bijectors
using CUDA
using DensityInterface
using SciGL

"""
Vectorization support by storing multiple distributions in an array for broadcasting.

Implement:
- marginals(): Return the internal array of distributions
- DensityInterface.logdensityof(dist::AbstractVectorizedDistribution, y) - individual reduction strategy for the array of KernelDistributions

You can use:
- Bijectors.bictor(dist)
- rand & rand!
- to_cpu(dist)
- to_gpu(dist)
"""
abstract type AbstractVectorizedDistribution end

"""
    to_cpu(dist)
Transfer the internal distributions to the CPU.
"""
to_cpu(dist::T) where {T<:AbstractVectorizedDistribution} = T.name.wrapper(Array(marginals(dist)))

"""
    to_gpu(dist)
Transfer the internal distributions to the GPU.
"""
SciGL.to_gpu(dist::T) where {T<:AbstractVectorizedDistribution} = T.name.wrapper(CuArray(marginals(dist)))

"""
    rand(rng, dist, [dims...])
Mutate the array `A` by sampling from `dist`.
"""
Random.rand!(rng::AbstractRNG, dist::AbstractVectorizedDistribution, A::AbstractArray) = rand!(rng, marginals(dist), A)

"""
    rand(rng, dist, [dims...])
Sample an array from `dist` of size `dims`.
"""
Base.rand(rng::AbstractRNG, dist::AbstractVectorizedDistribution, dims::Integer...) = rand(rng, marginals(dist), dims...)

"""
    rand(rng, dist, [dims...])
Sample an array from `dist` of size `dims`.
"""
Base.rand(rng::AbstractRNG, dist::AbstractVectorizedDistribution) = rand(rng, marginals(dist))

"""
    bijector(dist)
Scalar bijector for the marginals
"""
Bijectors.bijector(dist::AbstractVectorizedDistribution) = marginals(dist) |> first |> bijector

# VectorizedDistribution

"""
    VectorizedDistribution
Behaves similar to the ProductKernel but assumes a vectorization of the data over last dimension.
"""
struct VectorizedDistribution{T<:AbstractArray{<:AbstractKernelDistribution}} <: AbstractVectorizedDistribution
    marginals::T
end

marginals(dist::VectorizedDistribution) = dist.marginals

function DensityInterface.logdensityof(dist::VectorizedDistribution, x)
    n_red = ndims(marginals(dist))
    # TODO is the intended design to be automatically broadcasted or should / can I leave it up to the user?
    R = sum(logdensityof.(marginals(dist), x); dims=1:n_red)
    # Returns an array of size () instead of a scalar. Conditional conversion to scalar would defeat type stability. 
    # Does only support Tuple for dims
    dropdims(R; dims=(1:n_red...,))
end

"""
    VectorizedDistribution(dist)
Convert an AbstractVectorizedDistribution to a VectorizedDistribution.
"""
VectorizedDistribution(dist::AbstractVectorizedDistribution) = VectorizedDistribution(marginals(dist))

Base.show(io::IO, dist::VectorizedDistribution{T}) where {T} = print(io, "VectorizedDistribution{$(T)}\n  marginals: $(eltype(dist.marginals)) \n  size: $(size(dist.marginals))")


# ProductDistribution

"""
    ProductDistribution
Assumes independent marginals, whose logdensity is the sum of each individual logdensity (like the MeasureTheory Product measure).
"""
struct ProductDistribution{T<:AbstractArray{<:AbstractKernelDistribution}} <: AbstractVectorizedDistribution
    # WARN CUDA kernels only work for the same distribution with different parametrization
    marginals::T
end

"""
    ProductDistribution(dist)
Convert an AbstractVectorizedDistribution to a VectorizedDistribution.
"""
ProductDistribution(dist::AbstractVectorizedDistribution) = ProductDistribution(marginals(dist))

marginals(dist::ProductDistribution) = dist.marginals

Base.show(io::IO, dist::ProductDistribution{T}) where {T} = print(io, "ProductDistribution{$(T)}\n  marginals: $(typeof(dist.marginals))\n  size: $(size(dist.marginals))")

DensityInterface.logdensityof(dist::ProductDistribution, x) = sum(logdensityof.(marginals(dist), x))

# TODO is this what the Vectorized distribution should have been? 
# TEST: to_gpu should not be required anymore since it should automatically operate on the correct device?
struct BroadcastedDistribution{T<:Broadcasted,N} <: AbstractVectorizedDistribution
    marginals::T
    size::Dims{N}
end

# TODO is dims determined by the max size of params?
BroadcastedDistribution(::Type{T}, dims::Dims, params...) where {T<:AbstractKernelDistribution} = BroadcastedDistribution(broadcasted(T, params...), dims)

BroadcastedDistribution(T::Type, params...) = BroadcastedDistribution(T, (), params...)

# TODO simple but maybe not always efficient
# TODO should the rng be based on the location of the params?
marginals(dist::BroadcastedDistribution) = materialize(dist.marginals)

# TEST benchmark vs VectorizedDistribution
function DensityInterface.logdensityof(dist::BroadcastedDistribution, x)
    # WARN Efficient reduction of Broadcasted only works without dims. 
    # logdensities = broadcasted(logdensityof, dist.marginals, x)
    # TODO is this the same implementation as VectorizedDistribution?
    R = sum(logdensityof.(dist.marginals, x), dims=1:length(dist.size))
    # Does not support range, only tuple for dims
    dropdims(R; dims=(1:length(dist.size)...,))
end

# TODO might enable more efficient random number generation
# struct BroadcastedTypedDistribution{T<:AbstractKernelDistribution,U,N} <: AbstractVectorizedDistribution
#     dist_type::Type{T}
#     params::U
#     dims::Dims{N}
# end

# # function Base.rand(rng::AbstractRNG, dist::BroadcastedDistribution, dims::Integer...)
# # Would need to know the type of the internal distribution
# #     array_for_rng(rng, Dist_T, dist.dims..., dims...))
# #     rand(rng, marginals(dist), size(marginals(dist))..., dims...)
# # end

# # TEST benachmark vs VectorizedDistribution
# function DensityInterface.logdensityof(dist::BroadcastedTypedDistribution, x)
#     # TODO single line is broadcasted lazily right?
#     logdensities = broadcasted(logdensityof, dist.marginals, x)
#     # TODO dims required as tuple or is range okay?
#     R = reduce(+, logdensities, dims=1:length(dist.size))
#     dropdims(R; dims=(1:length(dist.size)...,))
# end
