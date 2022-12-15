# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors
using Base.Broadcast: broadcasted, materialize

"""
    Circular
Transform ℝ → [0,2π)
"""
struct Circular{N} <: Bijector{N} end

"""
    (::Circular)(x)
Transform from [0,2π] to ℝ.
In theory inverse of mod does not exist, in practice the same value is returned, since `[0,2π] ∈ ℝ`
"""
(::Circular)(x) = x

"""
    (::Circular)(y)
Uses `mod2pi` to transform ℝ to [0,2π].
"""
(::Inverse{<:Circular})(y) = mod2pi.(y)

"""
    logabsdetjac(::Circular, x)
mod2pi will not be zero for n*2*π, thus the discontinuity will not be reached.
Thus, the log Jacobian is always 0. 
"""
Bijectors.logabsdetjac(::Circular, x) = zero(x)
Bijectors.logabsdetjac(::Inverse{<:Circular}, y) = zero(y)

"""
    ZeroIdentity
Identity bijector without any allocations.
"""
struct ZeroIdentity <: Bijector{0} end
(::ZeroIdentity)(x) = x
Bijectors.inverse(b::ZeroIdentity) = b

Bijectors.logabsdetjac(::ZeroIdentity, x) = zero(eltype(x))
Bijectors.logabsdetjac(::Inverse{<:ZeroIdentity}, x) = zero(eltype(x))

# Custom reduction like BroadcastedDistribution
"""
    BroadcastedBijector
Uses broadcasting to enable bijectors over multiple dimensions.
Moreover reduction dims can be specified which are applied when calculating the logabsdetjac correction.
"""
struct BroadcastedBijector{N,B} <: Bijector{N}
    dims::Dims{N}
    bijectors::B
end

"""
    (::BroadcastedBijector)(x)
Applies the internal bijectors via broadcasting.
"""
(b::BroadcastedBijector)(x) = x .|> b.bijectors

"""
    inverse(b::BroadcastedBijector)
Lazily applies inverse to the internal bijectors.
"""
Bijectors.inverse(b::BroadcastedBijector) = BroadcastedBijector(b.dims, broadcasted(inverse, b.bijectors))

"""
    materialize(b::BroadcastedBijector)
Materialize the possibly broadcasted internal bijectors.
Bijectors are usually required to transform the priors domain, which does not change.
"""
Broadcast.materialize(b::BroadcastedBijector) = BroadcastedBijector(b.dims, materialize(b.bijectors))

"""
    logabsdetjac(b::BroadcastedBijector, x)
Calculate the logabsdetjac correction an reduce the `b.dims` by summing them up.
"""
Bijectors.logabsdetjac(b::BroadcastedBijector, x) = sum_and_dropdims(logabsdetjac.(b.bijectors, x), b.dims)

"""
    with_logabsdet_jacobian(b::BroadcastedBijector, x)
Calculate the transformed variables with the logabsdetjac correction in an optimized fashion.
The logabsdetjac correction is reduced by summing up `b.dims`.
"""
Bijectors.with_logabsdet_jacobian(b::BroadcastedBijector, x) = with_logabsdet_jacobian_array(b, x)
Bijectors.with_logabsdet_jacobian(b::BroadcastedBijector{0}, x::AbstractArray) = with_logabsdet_jacobian_array(b, x)

function with_logabsdet_jacobian_array(b, x)
    with_logjac = with_logabsdet_jacobian.(b.bijectors, x)
    y, logjacs = first.(with_logjac), last.(with_logjac)
    y, sum_and_dropdims(logjacs, b.dims)
end

# Scalar case results in a tuple for with_logjac instead of an array of tuples
Bijectors.with_logabsdet_jacobian(b::BroadcastedBijector{0}, x) = with_logabsdet_jacobian.(b.bijectors, x)
Bijectors.with_logabsdet_jacobian(b::BroadcastedBijector{0}, x::AbstractArray{<:Any,0}) = with_logabsdet_jacobian.(b.bijectors, x)


"""
    is_identity(::Bijector)
Returns true if the bijector is the identity, i.e. maps ℝ → ℝ
"""
is_identity(::Bijector) = false
is_identity(::Bijectors.Identity) = true
is_identity(::ZeroIdentity) = true
is_identity(bijector::AbstractArray{<:Bijector}) = mapreduce(is_identity, &, bijector)
is_identity(bijector::BroadcastedBijector) = is_identity(materialize(bijector.bijectors))
is_identity(bijectors::Union{Bijector,AbstractArray{<:Bijector}}...) = mapreduce(is_identity, &, bijectors)


# Wrapper
is_identity(dist::Distribution) = is_identity(bijector(dist))

