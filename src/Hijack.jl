# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using CUDA
# using Strided

# Hijacking functions of other libraries

# Does not work for BitArrays https://github.com/Jutho/Strided.jl/issues/13
# Allow to use StridedView as an in-place replacement for Array
# Strided.StridedView{T}(initializer, dims::Integer...) where {T} = Array{T}(initializer, dims...) |> StridedView
# Strided.StridedView(parent::CuArray) = parent |> Array |> StridedView

# Promotoe everything to CPU, ideally as StridedView
Base.promote_rule(::Type{<:CuArray}, ::Type{<:Array}) = Array
# Base.promote_rule(::Type{<:StridedView}, ::Type{<:Array}) = StridedView
# Base.promote_rule(::Type{<:StridedView}, ::Type{<:CuArray}) = StridedView
