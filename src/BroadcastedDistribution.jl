# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Base.Broadcast: broadcasted, Broadcasted, materialize
using Bijectors
using DensityInterface
using Distributions

"""
    BroadcastedDistribution{T,N,M}
A **lazy** implementation for multi-dimensional distributions which makes it natural to use it on different devices and applying transformations afterwards.

At the core, `marginals` is a broadcasted distribution for a set of parameters.
Generating random numbers is based on the promoted type of the parameters and stored in `partype`.
Logdensities are evaluated using broadcasted and reduced by summing up `dims`, similar to a product distribution.
The reduction dimensions might differ from the dimensions of the parameters, in case that the parameters represent multiple samples.

`T` is the parameter type, `N` the number of 
"""
struct BroadcastedDistribution{T,N,M<:Broadcasted,S<:ValueSupport} <: Distribution{ArrayLikeVariate{N},S}
    partype::Type{T}
    dims::Dims{N}
    marginals::M

    # WARN Inferring the support via marginals |> first |> typeof cannot be executed on the GPU. What works is marginals |> materialize |> eltype but I want to avoid materializations which cause allocations.
    BroadcastedDistribution(partype::Type{T}, dims::Dims{N}, marginals::M, ::Type{S}) where {T,N,M,S<:ValueSupport} = new{T,N,M,S}(partype, dims, marginals)
end

# TODO reuse code in constructors?

"""
    BroadcastedDistribution(dist_fn, dims, params...)
Construct a BroadcastedDistribution for a distribution generating function, conditioned on params.
The `dims` of the distribution which are reduced are set manually so they can differ from the dims of the parameters.
"""
BroadcastedDistribution(dist_fn, dims::Dims, params...) = BroadcastedDistribution(promote_params_eltype(params...), dims, broadcasted(dist_fn, params...), Continuous)

"""
    BroadcastedDistribution(dist_fn, dims, params...)
Construct a BroadcastedDistribution for a distribution generating function, conditioned on params.
Defaults the reduction dimensions of the first `ndims(dists)` dimensions.
"""
function BroadcastedDistribution(dist_fn, params...)
    marginals = broadcasted(dist_fn, params...)
    # Dims(1:length(...)) not type stable?
    dims = (1:length(axes(marginals))...,)
    BroadcastedDistribution(promote_params_eltype(params...), dims, marginals, Continuous)
end

"""
    DiscreteBroadcastedDistribution(dist_fn, dims, params...)
Construct a BroadcastedDistribution for a discrete distribution generating function, conditioned on params.
The `dims` of the distribution which are reduced are set manually so they can differ from the dims of the parameters.
"""
DiscreteBroadcastedDistribution(dist_fn, dims::Dims, params...) = BroadcastedDistribution(promote_params_eltype(params...), dims, broadcasted(dist_fn, params...), Discrete)

"""
    promote_params_eltype(params...)
Promote the types of the elements in params to get the minimal common type.
"""
promote_params_eltype(params...) = promote_type(eltype.(params)...)

"""
    marginals(dist)
Lazy broadcasted array of distributions → use dot syntax, Broadcast.broadcasted(..., marginals) or Broadcast.materialize(marginals).
"""
marginals(dist::BroadcastedDistribution) = dist.marginals

Base.show(io::IO, dist::BroadcastedDistribution{T}) where {T} = print(io, "BroadcastedDistribution{$(T)}\n  dist function: $(recursive_marginals_string(marginals(dist)))\n  size: $(size(dist))\n  dims: $(dist.dims)\n  support: $(Distributions.value_support(dist))")

"""
    recursive_marginals_string
Recursively generates a string of the distribution type (function) of the broadcasted marginals.
"""
function recursive_marginals_string(marginals)
    res = "$(marginals.f) "
    if marginals.args[1] isa Broadcasted
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
The 
"""
Distributions.logpdf(dist::BroadcastedDistribution, x) = sum_and_dropdims(logdensityof.(marginals(dist), x); dims=dist.dims)

"""
    logpdf(dist, x)
Evaluate the logdensity of multi-dimensional distributions and data using broadcasting.
Special case for matching dimensions behaves like a `Product` distribution and returns a scalar.
"""
Distributions.logpdf(dist::BroadcastedDistribution{<:Any,N}, x::AbstractArray{<:Any,N}) where {N} = sum(logdensityof.(marginals(dist), x))

# <:Real Required to avoid ambiguities with Distributions.jl
Distributions.logpdf(dist::BroadcastedDistribution, x::AbstractArray{<:Real}) = sum_and_dropdims(logdensityof.(marginals(dist), x); dims=dist.dims)
Distributions.logpdf(dist::BroadcastedDistribution{<:Any,N}, x::AbstractArray{<:Real,N}) where {N} = sum(logdensityof.(marginals(dist), x))

# By default, Distributions.jl disallows logdensityof with multiple samples (Arrays and Matrices). BroadcastedDistribution should be inherently allowing multiple samples.
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

"""
    rand!(rng, dist, [dims...])
Mutate the array `A` by sampling from `dist`.
"""
Random.rand!(rng::AbstractRNG, dist::BroadcastedDistribution, A::AbstractArray{<:Real}) = _rand!(rng, marginals(dist), A)

# Bijectors

Bijectors.bijector(dist::BroadcastedDistribution) = dist |> marginals |> first |> bijector

"""
    transformed(dist)
Lazily transforms the distribution type to the unconstrained domain.
"""
Bijectors.transformed(dist::BroadcastedDistribution) = BroadcastedDistribution(dist.partype, dist.dims, broadcasted(transformed, dist.marginals))
