# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using TransformVariables

"""
    Circular
Transform ℝ → [0,2π)
"""
struct Circular <: TransformVariables.ScalarTransform end

"""
    transform(::Circular, x::Number)
Uses `mod2pi` to transform all the values to one circle interval.
"""
TransformVariables.transform(::Circular, x::Number) = mod2pi(x)

"""
    transform_and_logjac(::Circular, x::Number)
mod2pi will not be zero for n*2*π, thus the discontinuity will not be reached.
Thus, the log Jacobian is always 0. 
Returns: transformed value, 0
"""
TransformVariables.transform_and_logjac(t::Circular, x::Number) = transform(t, x), zero(x)

"""
    inverse(::Circular, x::Number)
In theory it does not exist, in practice the same value is returned, since `[0,2π) ∈ ℝ`
"""
TransformVariables.inverse(::Circular, x::Number) = x

"""
    as○
Transform ℝ → [0,2π)
"""
const as○ = Circular()

"""
    as_circular
Transform ℝ → [0,2π)
"""
const as_circular = as○

"""
    is_identity(tr)
Returns true if the transform is the identity, i.e. maps ℝ → ℝ
"""
is_identity(tr::T) where {T<:TransformVariables.AbstractTransform} = T <: TransformVariables.Identity

"""
    is_identity(tr)
Returns true if the transform is the identity, i.e. maps ℝ → ℝ
"""
is_identity(tr::TransformVariables.ArrayTransform) = is_identity(tr.transformation)

"""
    is_identity(tr)
Returns true if all inner transforms are identity, i.e. map ℝ → ℝ
"""
is_identity(tr::TransformVariables.TransformTuple) = mapreduce(is_identity, &, collect(tr.transformations))

"""
    is_identity(tr)
Returns true if the transforms of the sample are identity, i.e. map ℝ → ℝ
"""
is_identity(s::Sample) = is_identity(s.t)