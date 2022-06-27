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

# TODO remove
# """
#     rand(rng, dist, [dims...])
# Sample an array from `dist` of size `dims`.
# """
# Base.rand(rng::AbstractRNG, dist::AbstractVectorizedDistribution) = rand(rng, marginals(dist))

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

"""
    sum_and_dropdims(A; dims)
Sum the matrix A over the given dimensions and drop the very same dimensions afterwards.
Returns an array of size () instead of a scalar. Conditional conversion to scalar would defeat type stability. 
"""
sum_and_dropdims(A; dims) = dropdims(sum(A; dims=dims), dims=Tuple(dims))

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

# TODO is this what the Vectorized distribution should have been? 
# TODO remove?

# TEST: to_gpu should not be required anymore since it should automatically operate on the correct device?
struct BroadcastedDistribution{T,N,M} <: AbstractVectorizedDistribution where {T,N,M<:Broadcasted}
    partype::Type{T}
    # See VectorizedDistribution
    dims::Dims{N}
    marginals::M
end

# TODO Design decision: either explicitly type the distribution
# BroadcastedDistribution(dist_type::Type{<:AbstractKernelDistribution{T}}, dims::Dims, params...) where {T} = BroadcastedDistribution(T, dims, broadcasted(dist_type, params...))
# BroadcastedDistribution(dist_type::Type{<:AbstractKernelDistribution}, params...) = BroadcastedDistribution(dist_type, (), params...)

# TODO or infer from parameters, which I think is the idiomatic way
params_eltype(params...) = promote_type(eltype.(params)...)

BroadcastedDistribution(dist_type::Type, dims::Dims, params...) = BroadcastedDistribution(params_eltype(params...), dims, broadcasted(dist_type, params))

"""
    BroadcastedDistribution(dists)
Defaults the reduction dimensions of the first `ndims(dists)` dimensions.
"""
function BroadcastedDistribution(dist_type::Type, params...)
    # TODO broadcasted for multiple args? Partial application?
    marginals = broadcasted(dist_type, params...)
    dims = Dims(1:maximum(ndims.(params)))
    BroadcastedDistribution(params_eltype(params...), dims, marginals)
end

marginals(dist::BroadcastedDistribution) = dist.marginals

function Base.rand(rng::AbstractRNG, dist::BroadcastedDistribution{T}, dims::Integer...) where {T}
    # TODO could probably be generalized by implementing Base.eltype(AbstractVectorizedDistribution) or array_for_rng(rng, ::AbstractVectorizedDistribution)
    A = array_for_rng(rng, T, size(marginals(dist))..., dims...)
    rand!(rng, dist, A)
end

Random.rand!(rng::AbstractRNG, dist::BroadcastedDistribution, A::AbstractArray) = _rand!(rng, marginals(dist), A)

# TODO same as VectorizedDistribution but marginals are lazy 😄
Distributions.logpdf(dist::BroadcastedDistribution, x) = sum_and_dropdims(logdensityof.(marginals(dist), x); dims=dist.dims)

# This makes it very natural without having to define a separate type
Bijectors.transformed(dist::BroadcastedDistribution) = BroadcastedDistribution(dist.partype, dist.dims, broadcasted(transformed, dist.marginals))


# TEST benchmark BroadcastedDistribution vs VectorizedDistribution
# Result: Exactly the same number of allocations as VectorizedDistribution but without the need to materialize the distribution first. So we save one hidden allocation of the size of the parameters. The flexibility and less allocations of the broadcasted variant make it a winner. I do not even have to care about transferring the distributions to the gpu, because they are initialized based on the params type 😄


# TODO might enable more efficient random number generation
struct BroadcastedTypedDistribution{T,U<:AbstractKernelDistribution{T},V,N} <: AbstractVectorizedDistribution
    dist_type::Type{U}
    params::V
    dism::Dims{N}
end

function Base.rand(rng::AbstractRNG, dist::BroadcastedTypedDistribution{T}, dims::Integer...) where {T}
    # Would need to know the type of the internal distribution
    A = array_for_rng(rng, T, dist.size..., dims...)
    # TODO this is different, the rest is the same for broadcasted distributions
    dists = broadcasted(dist.type, dist.params)
    _rand!(rng, dists, A)
end

function Distributions.logpdf(dist::BroadcastedTypedDistribution, x)
    dists = broadcasted(dist.type, dist.params)
    # TODO single line is broadcasted lazily right?
    logdensities = logdensityof.(dists, x)
    sum_and_dropdims(logdensities; dims=dist.dims)
end

struct BroadcastedTransformedDistribution{T,U<:AbstractKernelDistribution{T},V,N} <: AbstractVectorizedDistribution
    type::Type{U}
    params::V
    dims::Dims{N}
end

BroadcastedTransformedDistribution(dist::BroadcastedTypedDistribution) = BroadcastedTransformedDistribution(dist.dist_type, dist.params, dist.size)

Bijectors.transformed(dist::BroadcastedTransformedDistribution) = BroadcastedTransformedDistribution(dist)

# TODO does it work?
# Bijectors.transformed(dist::BroadcastedTransformedDistribution) = BroadcastedTypedDistribution(dist.type)

# TODO reuse marginals for an AbstractBroadcastedDistribution
# marginals(dist::BroadcastedTransformedDistribution) = broadcasted(transformed, broadcasted(dist.type, dist.params))

function Base.rand(rng::AbstractRNG, dist::BroadcastedTransformedDistribution{T}, dims::Integer...) where {T}
    # Would need to know the type of the internal distribution
    A = array_for_rng(rng, T, dist.size..., dims...)
    # TODO this is different, the rest is the same for broadcasted distributions
    dists = broadcasted(transformed, broadcasted(dist.type, dist.params))
    _rand!(rng, dists, A)
end

function Distributions.logpdf(dist::BroadcastedTransformedDistribution, x)
    dists = broadcasted(transformed, broadcasted(dist.type, dist.params))
    # TODO single line is broadcasted lazily right?
    logdensities = logdensityof.(dists, x)
    sum_and_dropdims(logdensities; dims=dist.dims)
end

