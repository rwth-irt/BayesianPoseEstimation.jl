# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Bijectors

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
    is_identity(b)
Returns true if the bijector is the identity, i.e. maps ℝ → ℝ
"""
is_identity(b::Bijector...) = mapreduce(is_identity, &, b)

"""
    is_identity(v)
Returns true if the bijector the variable is the identity, i.e. map ℝ → ℝ
"""
is_identity(v::AbstractVariable) = is_identity(bijector(v))

"""
    is_identity(v)
Returns true if the bijectors the variables of `s` are the identity, i.e. map ℝ → ℝ
"""
is_identity(s::Sample) = is_identity(values(vars(s))...)
