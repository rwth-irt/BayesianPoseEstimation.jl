# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.


"""
    BroadcastedNode
Wraps a distribution or a child node by broadcasting it over the parameters via a BroadcastedDistribution.
You will need to take care of matching dimensionalities in the graph.
"""
struct BroadcastedNode{name,child_names,N<:NamedTuple{child_names},R<:AbstractRNG,M<:Union{Distribution,Function}} <: AbstractNode{name,child_names}
    children::N
    rng::R
    # Must be function to avoid UnionAll type instabilities
    model::M
end

# Convenience constructor for moving name to the parametric type
BroadcastedNode_(name::Symbol, children::N, rng::R, model::M) where {child_names,N<:NamedTuple{child_names},R<:AbstractRNG,M<:Union{Distribution,Function}} = BroadcastedNode{name,child_names,N,R,M}(children, rng, model)

# WARN Manipulated function not type stable for Type as arg
broadcast_model(::Type{T}, dims) where {T} = (x...) -> BroadcastedDistribution(T, dims, x...)
broadcast_model(fn::Function, dims) = (x...) -> BroadcastedDistribution(fn, dims, x...)

# Construct as parent
"""
    BroadcastedNode(name, children, rng, distribution)
Construct a node which automatically broadcasts the `distribution` over the parameters given by the `children`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims for the minimal realization of the distribution given the `children`.
"""
function BroadcastedNode(name::Symbol, children::NamedTuple, rng::AbstractRNG, distribution::Callable)
    # Workaround so D is not UnionAll but interpreted as constructor
    # No reduction by default
    sacrifice_model = broadcast_model(distribution, ())
    sacrifice_node = BroadcastedNode_(name, children, rng, sacrifice_model)
    sacrifice_values = rand(sacrifice_node)
    dims = param_dims(varvalue(sacrifice_node, sacrifice_values))
    BroadcastedNode_(name, children, rng, broadcast_model(distribution, dims))
end

# Construct as leaf
# TODO Do I like the parameter order, i.e. rng between params and distribution? Should model / distribution be moved to the end?
"""
    BroadcastedNode(name, rng, distribution, params...)
Construct the node as leaf (no children) by broadcasting the `distribution` over the `params`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims of the `params`.
"""
BroadcastedNode(name::Symbol, rng::AbstractRNG, distribution, params...) = BroadcastedNode_(name, (;), rng, ProductBroadcastedDistribution(distribution, params...))
