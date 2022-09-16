# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using CoordinateTransformations
using CUDA
using Random
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
# Scalar case
sum_and_dropdims(A::Number, ::Dims{N}) where {N} = A

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

"""
    array_for_rng(rng, T, dims...)
Generate the correct array to be used in rand! based on the random number generator provided.
CuArray for CUDA.RNG and Array for all other RNGs.
"""
array_for_rng(rng::AbstractRNG, ::Type{T}, dims::Integer...) where {T} = array_for_rng(rng){T}(undef, dims...)
array_for_rng(::AbstractRNG) = Array
array_for_rng(::CUDA.RNG) = CuArray

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
