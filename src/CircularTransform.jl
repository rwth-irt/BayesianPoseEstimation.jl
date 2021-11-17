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
Always returns zero(x)
"""
TransformVariables.transform_and_logjac(::Circular, x::Number) = zero(x)

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