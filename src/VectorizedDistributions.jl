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
- Distributions.logpdf(dist::AbstractVectorizedDistribution, y) - individual reduction strategy for the array of KernelDistributions

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

# DensityInterface

@inline DensityInterface.DensityKind(::AbstractVectorizedDistribution) = IsDensity()

# Avoid method ambiguities for AbstractVectorizedDistribution with a specialized x
DensityInterface.logdensityof(dist::AbstractVectorizedDistribution, x) = logpdf(dist, x)

# Random interface

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
Broadcasts the marginals over the data for the evaluation of the logdensity.
Specify the dimensions `dims` of a single data point to sum over these for the reduction.

Special cases:
* `dims = size(x)`: behaves like ProductDistribution
* `dims = ()`: behaves like logdensity.(marginals, x)
"""
struct VectorizedDistribution{T<:AbstractArray{<:AbstractKernelDistribution},N} <: AbstractVectorizedDistribution
    marginals::T
    dims::Dims{N}
end

"""
    VectorizedDistribution(dists)
Specify custom reduction dimensions which differ from `dists` dimensions.
"""
VectorizedDistribution(dists::AbstractArray{<:AbstractKernelDistribution}, dims) = VectorizedDistribution(dists, Dims(dims))

VectorizedDistribution(dists::AbstractArray{<:AbstractKernelDistribution}, dim::Integer) = VectorizedDistribution(dists, (dim,))

"""
    VectorizedDistribution(dists)
Defaults the reduction dimensions to the first `ndims(dists)` dimensions.
"""
VectorizedDistribution(dists::T) where {N,T<:AbstractArray{<:AbstractKernelDistribution,N}} = VectorizedDistribution{T,N}(dists, Dims(1:N))

"""
    VectorizedDistribution(dist)
Convert an AbstractVectorizedDistribution to a VectorizedDistribution.
"""
VectorizedDistribution(dist::AbstractVectorizedDistribution) = VectorizedDistribution(marginals(dist), size(marginals(dist)))

Base.show(io::IO, dist::VectorizedDistribution{T}) where {T} = print(io, "VectorizedDistribution{$(T)}\n  marginals: $(eltype(dist.marginals)) of size: $(size(dist.marginals)) \n  dims: $(dist.dims)")

Base.ndims(dist::VectorizedDistribution{<:Any,N}) where {N} = N

marginals(dist::VectorizedDistribution) = dist.marginals

# Custom implementation since the abstract type does not consider the reduction dims
to_cpu(dist::VectorizedDistribution) = VectorizedDistribution(Array(marginals(dist)), dist.dims)
SciGL.to_gpu(dist::VectorizedDistribution) = VectorizedDistribution(CuArray(marginals(dist)), dist.dims)

"""
    sum_and_dropdims(A; dims)
Sum the matrix A over the given dimensions and drop the very same dimensions afterwards.
Returns an array of size () instead of a scalar. Conditional conversion to scalar would defeat type stability. 
"""
sum_and_dropdims(A; dims) = dropdims(sum(A; dims=dims), dims=Tuple(dims))

Distributions.logpdf(dist::VectorizedDistribution, x) = sum_and_dropdims(logdensityof.(marginals(dist), x); dims=dist.dims)

# ProductDistribution

"""
    ProductDistribution
Assumes independent marginals, whose logdensity is the sum of each individual logdensity (like the MeasureTheory Product measure).
"""
struct ProductDistribution{T,N} <: AbstractVectorizedDistribution
    # WARN CUDA kernels only work for the same distribution with different parametrization
    marginals::T

    ProductDistribution(dists::T) where {N,T<:AbstractArray{<:AbstractKernelDistribution,N}} = new{T,N}(dists)
end

"""
    ProductDistribution(dist)
Convert an AbstractVectorizedDistribution to a ProductDistribution.
"""
ProductDistribution(dist::AbstractVectorizedDistribution) = ProductDistribution(marginals(dist))

Base.show(io::IO, dist::ProductDistribution{T}) where {T} = print(io, "ProductDistribution{$(T)}\n  marginals: $(typeof(dist.marginals))\n  size: $(size(dist.marginals))")

marginals(dist::ProductDistribution) = dist.marginals

Distributions.logpdf(dist::ProductDistribution, x) = sum(logdensityof.(marginals(dist), x))


# TODO is this what the Vectorized distribution should have been? 
# TODO remove?

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
function Distributions.logpdf(dist::BroadcastedDistribution, x)
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
