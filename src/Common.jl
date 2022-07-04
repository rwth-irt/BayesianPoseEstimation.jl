# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

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
    sum_and_dropdims(A; dims)
Sum the matrix A over the given dimensions and drop the very same dimensions afterwards.
Returns an array of size () instead of a scalar. Conditional conversion to scalar would defeat type stability. 
"""
sum_and_dropdims(A; dims) = dropdims(sum(A; dims=dims), dims=Tuple(dims))
