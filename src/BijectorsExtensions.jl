# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors

# TODO Quaternion Bijector? Distribution? Proposal?
# Quaternion from Bivectors
# https://probablydance.com/2017/08/05/intuitive-quaternions/

# Integration of PDF
# https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html

# Sphere sampling
# https://imois.in/posts/random-vectors-and-rotations-in-3d/
# http://corysimon.github.io/articles/uniformdistn-on-sphere/
# https://mathworld.wolfram.com/SpherePointPicking.html

# Perturbations
# https://public.am.files.1drv.com/y4mUWxmKEVqohznZ-iuTqNUjep9-nJkJps-GkKE9hhep1FFDzA6AuAsFfTw75s89ILNVlXFRN2hxyrRpe-vLSM7YjfLBKWzEGI2e2NyiFE7cfv9xTos2DvLToyTsohi1tG23IGWQV4V9cjwPBtTwlhhnSV1VhVj0W--ZtfltQwF3DqqChpHg8fuGzv6GjptWuk1SWN9mLIzgsVd_DSiSNapd0QlYHrkWds4J5Olbp_oQVs

# Implementations Rotations.jl uses Quaternion.jl and I think I use Rotations.jl in SciGL
# https://github.com/JuliaGeometry/Quaternions.jl/blob/master/src/Quaternion.jl

# TODO would have to write custom proposal, since the composition is the quaternion product

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
    is_identity(::Bijector)
Returns true if the bijector is the identity, i.e. maps ℝ → ℝ
"""
is_identity(::Bijector) = false
is_identity(::Bijectors.Identity) = true

"""
    is_identity(bijector)
Returns true if the `bijector` is the identity, i.e. maps ℝ → ℝ
"""
is_identity(bijector::Bijector...) = mapreduce(is_identity, &, bijector)
is_identity(bijector::AbstractArray{<:Bijector}...) = reduce(&, mapreduce.(is_identity, &, bijector))

# Wrapper
Bijectors.bijector(rng_model::RngModel) = bijector(model(rng_model))

is_identity(dist::Distribution) = is_identity(bijector(dist))
is_identity(model::IndependentModel) = is_identity(bijector.(values(models(model)))...)
is_identity(model::ComposedModel) = is_identity(models(model))
is_identity(rng_model::RngModel) = is_identity(model(rng_model))
