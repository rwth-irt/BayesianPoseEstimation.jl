# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

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
    map_materialize(b)
Maps Broadcast.materialize over a collection.
Falls back to materialize without map for non-collections.
"""
map_materialize(b) = Broadcast.materialize(b)
map_materialize(b::Union{NamedTuple,Tuple,AbstractArray}) = map(Broadcast.materialize, b)

"""
    pose_vector(t, r)
Convert and broadcast positions and orientations to a vector of `Pose`.
"""
to_pose(t, r) = to_pose(to_translation(t), to_rotation(r))
to_pose(t::Translation, r::Rotation) = Pose(t, r)
to_pose(t::AbstractArray{<:Translation}, r::AbstractArray{<:Rotation}) = Pose.(t, r)
to_pose(t::AbstractArray{<:Translation}, r::Rotation) = Pose.(t, (r,))
to_pose(t::Translation, r::AbstractArray{<:Rotation}) = Pose.((t,), r)

"""
    to_translation(A)
Convert the input to a `Vector{Translation}`.
"""
to_translation(A::AbstractArray{<:Number}) = [to_translation(t) for t in eachcol(A)]
to_translation(A::AbstractArray{<:Translation}) = A

# Wrap in SVector for type stability
to_translation(v::AbstractVector{<:Number}) = Translation(SVector{3}(v))
to_translation(t::Translation) = t

"""
    to_rotation(A, [T=RotXYZ])
Convert an array to a `Vector{Rotation}` column wise, optionally specifying the orientation representation `T`.
"""
to_rotation(A::AbstractArray{T}) where {T<:Number} = [RotXYZ(r...) for r in eachcol(A)]
# SciGL will take care of conversion to affine transformation matrix
to_rotation(v::AbstractVector{T}) where {T<:Number,} = RotXYZ(v...)

# Identity if already rotation
to_rotation(A::AbstractArray{<:Rotation}) = A

"""
    to_rotation(Q)
Convert an array of Quaternions to an array of Rotations.
"""
to_rotation(Q::Array{<:Quaternion}) = QuatRotation.(Q)
to_rotation(q::Quaternion) = QuatRotation(q)
to_rotation(r::Rotation) = QuatRotation(r)


"""
    norm_dims(A, [p=2; dims=(1,)])
For any iterable container A (including arrays of any dimension) of numbers (or any element type for which norm is defined), compute the p-norm (defaulting to p=2) as if A were a vector of the corresponding length.

The p-norm is defined as

``\\|A\\|_p = \\left( \\sum_{i=1}^n | a_i | ^p \\right)^{1/p}``

Compared to norm, you can specify the dims to sum over.
"""
norm_dims(A::AbstractArray, p=2; dims=(1,)) = sum(x -> abs(x)^p, A; dims=dims) .^ inv(p)

"""
    normalize_dims!(A, [p=2; dims=(1,)])
Normalize the array A inplace so that its p-norm equals unity, i.e. norm(a, p) == 1.
Compared to normalize, you can specify the dims to sum over.
"""
function normalize_dims!(A::AbstractArray, p=2; dims=(1,))
    A .= A ./ norm_dims(A, p; dims=dims)
end

"""
    normalize_dims(A, [p=2; dims=(1,)])
Normalize the array A so that its p-norm equals unity, i.e. norm(a, p) == 1.
Compared to normalize, you can specify the dims to sum over.
"""
normalize_dims(A::AbstractArray, p=2; dims=(1,)) = normalize_dims!(similar(A), p; dims=dims)

"""
    to_cpu(x)
If required, x is converted to an Array.
"""
to_cpu(x::AbstractArray) = Array(x)
to_cpu(x::Array) = x
to_cpu(x::Real) = x
