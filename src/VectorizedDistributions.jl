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
# TODO only used in logdensityof -> drop it and directly implement logdensityof
- reduce_vectorized(operator, dist::AbstractVectorizedDistribution, A::AbstractArray) - individual reduction strategy, used in logdensityof

You can use:
- DensityInterface.logdensityof(dist::AbstractVectorizedDistribution, x) - use logdensityof for arrays of KernelDistributions
- Bijectors.bictor(dist)
- rand & rand!
- to_cpu(dist)
- to_gpu(dist)
"""
abstract type AbstractVectorizedDistribution end

DensityInterface.logdensityof(dist::AbstractVectorizedDistribution, x) = reduce_vectorized(+, dist, logdensityof(marginals(dist), x))

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
Base.rand(rng::AbstractRNG, dist::AbstractVectorizedDistribution, dims::Integer...) = rand(rng, marginals(dist), size(marginals(dist))..., dims...)

"""
    rand(rng, dist, [dims...])
Sample an array from `dist` of size `dims`.
"""
Base.rand(rng::AbstractRNG, dist::AbstractVectorizedDistribution) = rand(rng, marginals(dist), size(marginals(dist))...)

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

"""
    VectorizedDistribution(dist)
Convert an AbstractVectorizedDistribution to a VectorizedDistribution.
"""
VectorizedDistribution(dist::AbstractVectorizedDistribution) = VectorizedDistribution(marginals(dist))

Base.show(io::IO, dist::VectorizedDistribution{T}) where {T} = print(io, "VectorizedDistribution{$(T)}\n  marginals: $(eltype(dist.marginals)) \n  size: $(size(dist.marginals))")


"""
    reduce_vectorized(operator, dist, A)
Reduces the first `ndims(dist)` dimensions of the Matrix `A` using the `operator`. 
"""
function reduce_vectorized(operator, dist::VectorizedDistribution, A::AbstractArray)
    n_red = ndims(marginals(dist))
    R = reduce(operator, A; dims=(1:n_red...,))
    # Returns an array of size () instead of a scalar. Conditional conversion to scalar would defeat type stability. 
    dropdims(R; dims=(1:n_red...,))
end

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

"""
    reduce_vectorized(operatpr, dist, A)
Reduces all dimensions of the Matrix `A` using the `operator`. 
"""
reduce_vectorized(operator, ::ProductDistribution, A::AbstractArray) = reduce(operator, A)


# TODO is this what the Vectorized distribution should have been? 
# TEST: to_gpu should not be required anymore since it should automatigacally operate on the correct device?
struct BroadcastedDistribution{T<:Broadcasted,N} <: AbstractVectorizedDistribution
    marginals::T
    # TODO does it make sense to store the params as tuple?
    size::Dims{N}
end

# TODO is dims determined by the max size of params?
BroadcastedDistribution(::Type{T}, dims::Dims, params...) where {T<:AbstractKernelDistribution} = BroadcastedDistribution(broadcasted(T, params...), dims)

BroadcastedDistribution(T::Type, params...) = BroadcastedDistribution(T, (), params...)

# TODO simple but maybe not always efficient
# TODO should the rng be based on the location of the params?
marginals(dist::BroadcastedDistribution) = materialize(dist.marginals)

# function Base.rand(rng::AbstractRNG, dist::BroadcastedDistribution, dims::Integer...)
# Would need to know the type of the internal distribution
#     array_for_rng(rng, Dist_T, dist.dims..., dims...))
#     rand(rng, marginals(dist), size(marginals(dist))..., dims...)
# end

# TEST benachmark vs VectorizedDistribution
function DensityInterface.logdensityof(dist::BroadcastedDistribution, x)
    logdensities = broadcasted(logdensityof, dist.marginals, x)
    # TODO dims required as tuple or is range okay?
    R = reduce(+, logdensities, dims=1:length(dist.size))
    dropdims(R; dims=(1:length(dist.size)...,))
end
