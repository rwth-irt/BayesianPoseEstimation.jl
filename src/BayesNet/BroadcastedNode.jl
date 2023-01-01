# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

"""
    BroadcastedNode
Broadcasts the parameters of the children using a BroadcastedDistribution.
Also takes care of matching the dimensions for broadcasting multiple samples.
"""
struct BroadcastedNode{name,child_names,C<:NamedTuple{child_names},R<:AbstractRNG,M,N,D<:Tuple{Vararg{Dims}}} <: AbstractNode{name,child_names}
    children::C
    rng::R
    # Must be function to avoid UnionAll type instabilities
    model::M
    model_dims::Dims{N}
    child_sizes::D
end

# Convenience constructor for moving name to the parametric type.
BroadcastedNode(name::Symbol, rng::R, model::M, model_dims::Dims{N}, children::C, child_sizes::D) where {child_names,C<:NamedTuple{child_names},R<:AbstractRNG,M,N,D<:Tuple{Vararg{Dims}}} = BroadcastedNode{name,child_names,C,R,M,N,D}(children, rng, model, model_dims, child_sizes)

"""
    BroadcastedNode(name, rng, distribution, children)
Construct a node which automatically broadcasts the `distribution` over the parameters given by the `children`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims for the minimal realization of the distribution given the `children`.
"""
function BroadcastedNode(name::Symbol, rng::AbstractRNG, distribution::Base.Callable, children::NamedTuple)
    # Generate one sample to calculate dimensions of the node and children. Empty Dims because they are unknown and don't make a difference for a single sample generation.
    sacrifice_model = broadcast_model(distribution, ())
    sacrifice_child_sizes = ntuple(_ -> (), length(children))
    sacrifice_node = BroadcastedNode(name, rng, sacrifice_model, (), children, sacrifice_child_sizes)
    sacrifice_nt = rand(sacrifice_node)

    model_dims = param_dims(varvalue(sacrifice_node, sacrifice_nt))
    child_sizes = size.(childvalues(sacrifice_node, sacrifice_nt))
    node_model = broadcast_model(distribution, model_dims)
    BroadcastedNode(name, rng, node_model, model_dims, children, child_sizes)
end

# Construct as leaf
"""
    BroadcastedNode(name, rng, distribution, params...)
Construct the node as leaf (no children) by broadcasting the `distribution` over the `params`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims of the `params`.
"""
BroadcastedNode(name::Symbol, rng::AbstractRNG, distribution, params...) = BroadcastedNode(name, rng, BroadcastedDistribution(distribution, params...), param_dims(params...), (;), ())


# WARN Manipulated function not type stable for Type as arg
"""
    broadcast_model(fn, dims)
Wraps the distribution generating function with a BroadcastedDistribution given the reduction `dims`.
"""
broadcast_model(::Type{T}, dims) where {T} = (x...) -> BroadcastedDistribution(T, dims, x...)
broadcast_model(fn::Function, dims) = (x...) -> BroadcastedDistribution(fn, dims, x...)

# Override generic BayesNet.jl behavior by inserting additional dims as required for broadcasting
function childvalues(node::BroadcastedNode{<:Any,childnames}, nt::NamedTuple) where {childnames}
    child_values = values(nt[childnames])
    # WARN Broadcasting not type stable?
    map(child_values, node.child_sizes) do cv, cs
        insertdims(cv, cs, node.model_dims)
    end
end

"""
    insertdims(A, child_sizes, dist_dims)
Reshape `A` by inserting dims of length one to make it compatible for broadcasting multiple random samples of differently shaped children.
`child_sizes` are the dims of a single sample from the child node and `dist_dims` the dims of the broadcasted distribution which results from a single sample of all child nodes.

# Rationale
When proposing multiple samples, originally matching dims of the BroadcastedDistribution do not work anymore.
E.g. if one child has (3,) and the other (3,2) sized samples, proposing multiple samples result in incompatible dimensions (3,5) and (3,2,5).
Julia expands dimensions of length one when broadcasting, so reshaping the array with dimensions of length one enables proposing multiple samples, for the above: (3,5) insertdims → (3,1,5) broadcast → (3,2,5)
https://freecontent.manning.com/vectorizing-your-code-using-broadcasting/
https://discourse.julialang.org/t/designating-the-axes-for-broadcasting/29203/2
"""
insertdims(A::Real, ::Dims, ::Dims) = A
# For initialization
insertdims(A, ::Dims, ::Dims{0}) = A
# Avoid ambiguities
insertdims(A::Real, ::Dims, ::Dims{0}) = A

function insertdims(A, original::Dims{O}, ::Dims{B}) where {O,B}
    # WARN fill array is not type stable
    fill_ones = ntuple(_ -> 1, B - O)
    # Dimension of multiple samples
    remaining = size(A)[O+1:end]
    reshape(A, original..., fill_ones..., remaining...)
end