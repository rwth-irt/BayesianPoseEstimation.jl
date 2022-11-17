# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base.Broadcast: broadcasted, instantiate, materialize, Broadcasted
using Bijectors
using DensityInterface
using Distributions

"""
    BroadcastedDistribution{T,N,M}
A **lazy** implementation for multi-dimensional distributions which makes it natural to use it on different devices and applying transformations afterwards.

At the core, `marginals` is a broadcasted distribution for a set of parameters.
Generating random numbers is based on the promoted type `T` of the parameters and stored in.
Logdensities are evaluated using broadcasted and reduced by summing up `dims`, similar to a product distribution.
The reduction dimensions `N` might differ from the dimensions of the parameters, in case that the parameters represent multiple samples.
"""
struct BroadcastedDistribution{T,N,M<:Broadcasted,S<:ValueSupport} <: Distribution{ArrayLikeVariate{N},S}
    dims::Dims{N}
    marginals::M
end

# WARN Inferring the support via marginals |> first |> typeof cannot be executed on the GPU. What works is marginals |> materialize |> eltype but I want to avoid materializations which cause allocations.
BroadcastedDistribution(::Type{T}, dims::Dims{N}, marginals::M, ::Type{S}) where {T,N,M,S<:ValueSupport} = BroadcastedDistribution{T,N,M,S}(dims, marginals)

"""
    BroadcastedDistribution(dist, dims, params...)
Construct a BroadcastedDistribution for a distribution generating function, conditioned on params.
The `dims` of the distribution which are reduced are set manually so they can differ from the dims of the parameters.
"""
BroadcastedDistribution(dist, dims::Dims, params...) =
    BroadcastedDistribution(promote_params_eltype(params...), dims, broadcasted(dist, params...), Continuous)

"""
    DiscreteBroadcastedDistribution(dist, dims, params...)
Construct a BroadcastedDistribution for a discrete distribution generating function, conditioned on params.
The `dims` of the distribution which are reduced are set manually so they can differ from the dims of the parameters.
"""
DiscreteBroadcastedDistribution(dist, dims::Dims, params...) = BroadcastedDistribution(promote_params_eltype(params...), dims, broadcasted(dist, params...), Discrete)

# WARN needs a different name because the params... cause problems with inferring the correct method when compiling
"""
    ProductBroadcastedDistribution(dist, params...)
Construct a BroadcastedDistribution for a distribution generating function, conditioned on params.
Automatically reduces all dimensions of the parameters, like a product distribution.
"""
ProductBroadcastedDistribution(dist, params...) = BroadcastedDistribution(promote_params_eltype(params...), param_dims(params...), broadcasted(dist, params...), Continuous)

"""
    promote_params_eltype(params...)
Promote the types of the elements in params to get the minimal common type.
"""
promote_params_eltype(params...) = promote_type(eltype.(params)...)

"""
    n_param_dims(params...)
Finds the maximum ndims of the parameters.
"""
n_param_dims(params...) = maximum(ndims.(params))

"""
    param_dims(params...)
Finds the maximum possible Dims of the parameters.
"""
param_dims(params...) = (1:n_param_dims(params...)...,)

"""
    marginals(dist)
Lazy broadcasted array of distributions → use dot syntax, Broadcast.broadcasted([...], marginals) or Broadcast.materialize(marginals).
"""
marginals(dist::BroadcastedDistribution) = dist.marginals

Base.show(io::IO, dist::BroadcastedDistribution{T}) where {T} = print(io, "BroadcastedDistribution{$(T)}\n  dist function: $(recursive_marginals_string(marginals(dist)))\n  size: $(size(dist))\n  dims: $(dist.dims)\n  support: $(Distributions.value_support(dist))")

"""
    recursive_marginals_string
Recursively generates a string of the distribution type (function) of the broadcasted marginals.
"""
function recursive_marginals_string(marginals)
    res = "$(marginals.f) "
    if isempty(marginals.args)
        res
    elseif marginals.args[1] isa Broadcasted
        res *= recursive_marginals_string(marginals.args[1])
    end
    res
end

Base.axes(dist::BroadcastedDistribution) = axes(dist.marginals)
Base.Dims(dist::BroadcastedDistribution) = dist.dims
# Might differ from the dims of the marginals
Base.ndims(::BroadcastedDistribution{<:Any,N}) where {N} = N
Base.size(dist::BroadcastedDistribution) = size(dist.marginals)
Distributions.value_support(::BroadcastedDistribution{<:Any,<:Any,<:Any,S}) where {S} = S

"""
    logpdf(dist, x)
Evaluate the logdensity of multi-dimensional distributions and data using broadcasting.
"""
Distributions.logpdf(dist::BroadcastedDistribution, x) = sum_and_dropdims(logdensityof.(marginals(dist), x), dist.dims)

# TODO Maybe rethink basing this on Distributions.Distribution because this causes a lot of ambiguities
# Avoid ambiguities with Distributions.jl
Distributions.logpdf(dist::BroadcastedDistribution, x::AbstractArray) = sum_and_dropdims(logdensityof.(marginals(dist), x), dist.dims)
# <:Real Required to avoid ambiguities with Distributions.jl
Distributions.logpdf(dist::BroadcastedDistribution, x::AbstractArray{<:Real}) = sum_and_dropdims(logdensityof.(marginals(dist), x), dist.dims)

Distributions.logpdf(dist::BroadcastedDistribution{<:Any,N}, x::AbstractArray{<:AbstractArray{<:Real,N}}) where {N} = sum_and_dropdims(logdensityof.(marginals(dist), x), dist.dims)

# Scalar case for CUDA
Distributions.logpdf(dist::BroadcastedDistribution{<:Any,<:Any,<:Broadcasted{<:Broadcast.DefaultArrayStyle{0}}}, x::AbstractArray{<:Real}) = sum_and_dropdims(logdensityof.(materialize(marginals(dist)), x), dist.dims)

# By default, Distributions.jl disallows logdensityof with multiple samples (Arrays and Matrices). BroadcastedDistribution is inherently designed for multiple samples so allow them explicitly.
DensityInterface.logdensityof(dist::BroadcastedDistribution, x::AbstractArray) = logpdf(dist, x)
DensityInterface.logdensityof(dist::BroadcastedDistribution, x::AbstractMatrix) = logpdf(dist, x)

# Random Interface

"""
    rand(rng, dist, [dims...])
Sample an array from `dist` of size `(size(marginals)..., dims...)`.
The array type is based on the `rng` and the parameter type of the distribution.
"""
function Base.rand(rng::AbstractRNG, dist::BroadcastedDistribution{T}, dims::Int...) where {T}
    # could probably be generalized by implementing Base.eltype(AbstractVectorizedDistribution)
    A = array_for_rng(rng, T, size(marginals(dist))..., dims...)
    rand!(rng, dist, A)
end

# Scalar case
Base.rand(rng::AbstractRNG, dist::BroadcastedDistribution{<:Any,<:Any,<:Broadcasted{<:Broadcast.DefaultArrayStyle{0}}}, dims::Int...) = rand(rng, materialize(marginals(dist)), dims...)

# TODO this fix only works for BroadcastedDistribution of size 0, each KernelDistribution would require a specific implementation to avoid method ambiguities with the AbstractRNG
# Scalars can not be sampled on the GPU as they result in scalar indexing. Generate on the CPU instead
function Base.rand(rng::CUDA.RNG, dist::BroadcastedDistribution{<:Any,<:Any,<:Broadcasted{<:Broadcast.DefaultArrayStyle{0}}})
    # Init CPU rng from the CUDA rng
    cpu_rng = Random.default_rng()
    Random.seed!(cpu_rng, rng.seed)
    # Increment the CUDA rng
    # TODO copy-pasted from KernelDistributions.jl → move into its own function
    new_counter = Int64(rng.counter) + 1
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    # Return random value from CPU rng
    rand(cpu_rng, materialize(marginals(dist)))
end

"""
    rand!(rng, dist, [dims...])
Mutate the array `A` by sampling from `dist`.
"""
Random.rand!(rng::AbstractRNG, dist::BroadcastedDistribution, A::AbstractArray{<:Real}) = _rand!(rng, marginals(dist), A)

# Bijectors

# Each entry might have an individual parameterization of the bijector, also helps with correct device
Bijectors.bijector(dist::BroadcastedDistribution) = BroadcastedBijector(dist.dims, broadcasted(bijector, dist.marginals))

"""
    transformed(dist)
Lazily transforms the distribution type to the unconstrained domain.
"""
Bijectors.transformed(dist::BroadcastedDistribution{T,<:Any,<:Any,S}) where {T,S} = BroadcastedDistribution(T, dist.dims, broadcasted(transformed, dist.marginals), S)

Bijectors.link(dist::BroadcastedDistribution, x) = link.(dist |> marginals, x)
Bijectors.invlink(dist::BroadcastedDistribution, y) = invlink.(dist |> marginals, y)
