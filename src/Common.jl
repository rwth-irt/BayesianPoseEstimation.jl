# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CoordinateTransformations
using Rotations

"""
    flatten(x)
Flattens x to return a 1D array.
"""
flatten(x) = collect(Iterators.flatten(x))

"""
    map_intersect(f, a, b, default)
Maps the function `f` over the intersection of the keys of `a` and `b`.
Uses the value of `default`, which may be a function of `value(a[i])`, if no matching key is found in `b`.
Returns a NamedTuple with the same keys as `a` which makes it type-stable.
"""
map_intersect(f, a::NamedTuple{A}, b::NamedTuple, default) where {A} = NamedTuple{A}(map_intersect_(f, a, b, default))

# Barrier for type stability of getindex?
map_intersect_(f, a::NamedTuple{A}, b::NamedTuple{B}, default) where {A,B} =
    map(A) do k
        if k in B
            f(a[k], b[k])
        else
            default
        end
    end

map_intersect_(f, a::NamedTuple{A}, b::NamedTuple{B}, default_fn::Function) where {A,B} =
    map(A) do k
        if k in B
            f(a[k], b[k])
        else
            default_fn(value([k]))
        end
    end

"""
    map_intersect(f, a, b)
Maps the function `f` over the intersection of the keys of `a` and `b`.
Uses the value of `a` if no matching key is found in `b`.
Returns a NamedTuple with the same keys as `a` which makes it type-stable.
"""
function map_intersect(f, a::NamedTuple{A}, b::NamedTuple{B}) where {A,B}
    # Type stability is delicate
    filtered_keys = filter(in(A), B)
    filtered_values = map(f, a[filtered_keys], b[filtered_keys])
    NamedTuple{filtered_keys}(filtered_values)
end

"""
    sum_and_dropdims(A,[;] dims)
Sum the matrix A over the given dimensions and drop the very same dimensions afterwards.
In case of a matching number of dimensions, a scalar is returned
"""
sum_and_dropdims(A; dims) = sum_and_dropdims(A, dims)
# Cannot dispatch on named parameter so implement helper methods below
sum_and_dropdims(A, dims::Dims) = dropdims(sum(A; dims=dims), dims=dims)
# Case of matching dimensions → return scalar
sum_and_dropdims(A::AbstractArray{<:Any,N}, ::Dims{N}) where {N} = sum(A)

# WARN Do not try to implement reduction of Broadcasted via Base.mapreducedim!
# LinearIndices(::Broadcasted{<:Any,<:Tuple{Any}}) only works for 1D case: https://github.com/JuliaLang/julia/blob/v1.8.0/base/broadcast.jl#L245
# Type hijacking does not work, since Broadcasted handles indexing differently which results to different results
# Base.LinearIndices(bc::Broadcast.Broadcasted{<:Any,<:Tuple}) = LinearIndices(axes(bc))
# Base.has_fast_linear_indexing(bc::Broadcast.Broadcasted{<:Broadcast.BroadcastStyle,<:Tuple}) = false

"""
    pose_vector(t, r, [rot_type=RotXYZ])
Convert and broadcast positions and orientations to a vector of `Pose`.
"""
to_pose(t, r, rot_type=RotXYZ) = Pose.(to_translation(t), to_rotation(r, rot_type))
to_pose(t::AbstractVector, r::AbstractVector, rot_type=RotXYZ) = Pose(to_translation(t), to_rotation(r, rot_type))

"""
    to_translation(A)
Convert an array to a vector of `Translation` column wise.
"""
to_translation(A::AbstractArray{<:Number}) = [Translation(t) for t in eachcol(A)]
to_translation(A::AbstractArray{<:Translation}) = A
to_translation(v::AbstractVector{<:Number}) = Translation(v)

"""
    to_rotation(A, [T=RotXYZ])
Convert an array to a vector of `Rotation` column wise, optionally specifying the orientation representation `T`.
"""
to_rotation(A::AbstractArray{<:Number}, ::Type{T}=RotXYZ) where {T<:Rotation} = [T(r...) for r in eachcol(A)]
# SciGL will take care of conversion to affine transformation matrix
to_rotation(A::AbstractArray{<:Rotation}, ::Rotation) = A
to_rotation(v::AbstractVector{<:Number}, ::Type{T}=RotXYZ) where {T<:Rotation} = T(v...)
