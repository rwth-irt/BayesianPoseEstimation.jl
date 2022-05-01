# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CUDA
using DensityInterface
using TransformVariables

"""
Vectorization support by storing multiple distributions in an array for broadcasting.

Implement:
- marginals(): Return the internal array of distributions
- reduce_vectorized(op, d::AbstractVectorizedDistribution, A::AbstractArray) - individual reduction strategy, used in logdensityof

You can use:
- DensityInterface.logdensityof(d::AbstractVectorizedDistribution, x) - use logdensityof for arrays of KernelDistributions
- TransformVariables.as
- rand & rand!
- to_cpu(d)
- to_gpu(d)
"""
abstract type AbstractVectorizedDistribution{T} end

DensityInterface.logdensityof(d::AbstractVectorizedDistribution, x) = reduce_vectorized(+, d, logdensityof(marginals(d), x))

"""
    to_cpu(d)
Transfer the internal distributions to the CPU.
"""
to_cpu(d::T) where {T<:AbstractVectorizedDistribution} = T.name.wrapper(Array(marginals(d)))

"""
    to_gpu(d)
Transfer the internal distributions to the GPU.
"""
to_gpu(d::T) where {T<:AbstractVectorizedDistribution} = T.name.wrapper(CuArray(marginals(d)))

"""
    rand(rng, m, dims)
Mutate the array `A` by sampling from the distribution `d`.
"""
Random.rand!(rng::AbstractRNG, d::AbstractVectorizedDistribution, A::AbstractArray) = rand!(rng, marginals(d), A)

"""
    rand(rng, m, dims)
Sample an array from the distribution `d` of size `dims`.
"""
Base.rand(rng::AbstractRNG, d::AbstractVectorizedDistribution, dims::Integer...) = rand(rng, marginals(d), size(marginals(d))..., dims...)

"""
    rand(rng, m, dims)
Sample an array from the distribution `d` of size 1.
"""
Base.rand(rng::AbstractRNG, d::AbstractVectorizedDistribution) = rand(rng, marginals(d), size(marginals(d))..., 1)

"""
    as(d)
Scalar transform variable for the marginals
"""
TransformVariables.as(d::AbstractVectorizedDistribution) = marginals(d) |> first |> as

# VectorizedDistribution

"""
    VectorizedDistribution
Behaves similar to the ProductKernel but assumes a vectorization of the data over last dimension.
"""
struct VectorizedDistribution{T<:Real,U<:AbstractKernelDistribution{T},V<:AbstractArray{U}} <: AbstractVectorizedDistribution{T}
    marginals::V
end

marginals(d::VectorizedDistribution) = d.marginals

"""
    VectorizedDistribution(d)
Convert an AbstractVectorizedDistribution to a VectorizedDistribution.
"""
VectorizedDistribution(d::AbstractVectorizedDistribution) = VectorizedDistribution(marginals(d))

Base.show(io::IO, d::VectorizedDistribution{T}) where {T} = print(io, "VectorizedDistribution{$(T)}\n  marginals: $(eltype(d.marginals)) \n  size: $(size(d.marginals))")

Base.size(d::VectorizedDistribution) = d.size

"""
    reduce_vectorized(op, d, A)
Reduces the first `ndims(d)` dimensions of the Matrix `A` using the operator `op`. 
"""
function reduce_vectorized(op, d::VectorizedDistribution, A::AbstractArray)
    n_red = ndims(marginals(d))
    R = reduce(op, A; dims=(1:n_red...,))
    dropdims(R; dims=(1:n_red...,))
end

# ProductDistribution

"""
    ProductDistribution
Assumes independent marginals, whose logdensity is the sum of each individual logdensity (like the MeasureTheory Product measure).
"""
struct ProductDistribution{T<:Real,U<:AbstractKernelDistribution{T},V<:AbstractArray{U}} <: AbstractVectorizedDistribution{T}
    # WARN CUDA kernels only work for the same distribution with different parametrization
    marginals::V
end

"""
    ProductDistribution(d)
Convert an AbstractVectorizedDistribution to a VectorizedDistribution.
"""
ProductDistribution(d::AbstractVectorizedDistribution) = ProductDistribution(marginals(d))

marginals(d::ProductDistribution) = d.marginals

Base.show(io::IO, d::ProductDistribution{T}) where {T} = print(io, "ProductDistribution{$(T)}\n  marginals: $(typeof(d.marginals))\n  size: $(size(d.marginals))")

"""
    reduce_vectorized(op, d, A)
Reduces all dimensions of the Matrix `A` using the operator `op`. 
"""
reduce_vectorized(op, ::ProductDistribution, A::AbstractArray) = reduce(op, A)