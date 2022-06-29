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
    rand!(rng, dist, [dims...])
Mutate the array `A` by sampling from `dist`.
"""
Random.rand!(rng::AbstractRNG, dist::AbstractVectorizedDistribution, A::AbstractArray) = rand!(rng, marginals(dist), A)

"""
    rand(rng, dist, [dims...])
Sample an array from `dist` of size `dims`.
"""
Base.rand(rng::AbstractRNG, dist::AbstractVectorizedDistribution, dims::Integer...) = rand(rng, marginals(dist), dims...)

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
Defaults the reduction dimensions of the first `ndims(dists)` dimensions.
"""
VectorizedDistribution(dists::T) where {N,T<:AbstractArray{<:AbstractKernelDistribution,N}} = VectorizedDistribution{T,N}(dists, Dims(1:N))

"""
    VectorizedDistribution(dist)
Convert an AbstractVectorizedDistribution to a VectorizedDistribution.
"""
# TODO is size correct?
VectorizedDistribution(dist::AbstractVectorizedDistribution) = VectorizedDistribution(marginals(dist), size(marginals(dist)))

Base.show(io::IO, dist::VectorizedDistribution{T}) where {T} = print(io, "VectorizedDistribution{$(T)}\n  marginals: $(eltype(dist.marginals)) of size: $(size(dist.marginals)) \n  dims: $(dist.dims)")

Base.ndims(::VectorizedDistribution{<:Any,N}) where {N} = N

marginals(dist::VectorizedDistribution) = dist.marginals

# Custom implementation since the abstract type does not consider the reduction dims
to_cpu(dist::VectorizedDistribution) = VectorizedDistribution(Array(marginals(dist)), dist.dims)
SciGL.to_gpu(dist::VectorizedDistribution) = VectorizedDistribution(CuArray(marginals(dist)), dist.dims)

Distributions.logpdf(dist::VectorizedDistribution, x) = sum_and_dropdims(logdensityof.(marginals(dist), x); dims=dist.dims)

# TransformedVectorizedDistribution

struct TransformedVectorizedDistribution{T,N} <: AbstractVectorizedDistribution
    internal::VectorizedDistribution{T,N}
end

Bijectors.transformed(dist::VectorizedDistribution) = TransformedVectorizedDistribution(dist)

Base.show(io::IO, dist::TransformedVectorizedDistribution{T,N}) where {T,N} = print(io, "TransformedVectorizedDistribution{$(T),$(N)}\n  internal: $(dist.internal)")

Base.ndims(::TransformedVectorizedDistribution{<:Any,N}) where {N} = N

marginals(dist::TransformedVectorizedDistribution) = dist.internal |> marginals .|> transformed

# Custom implementation since the abstract type does not consider the reduction dims
to_cpu(dist::TransformedVectorizedDistribution) = dist.internal |> to_cpu |> TransformedVectorizedDistribution
SciGL.to_gpu(dist::TransformedVectorizedDistribution) = dist.internal |> to_gpu |> TransformedVectorizedDistribution

# TODO maybe introduce AbstractTransformedVectorizedDistribution

# Broadcastable implementations to avoid allocations for the transformations

Random.rand!(rng::AbstractRNG, dist::TransformedVectorizedDistribution, A::AbstractArray) = _rand!(rng, broadcasted(transformed, marginals(dist.internal)), A)

function Base.rand(rng::AbstractRNG, dist::TransformedVectorizedDistribution, dims::Integer...)
    internal_marginals = marginals(dist.internal)
    A = array_for_rng(rng, eltype(first(internal_marginals)), size(internal_marginals)..., dims...)
    rand!(rng, dist, A)
end

Distributions.logpdf(dist::TransformedVectorizedDistribution, x) = sum_and_dropdims(logdensityof.(transformed.(marginals(dist.internal)), x); dims=ndims(dist.internal))


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
