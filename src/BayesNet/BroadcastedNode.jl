# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.


"""
    BroadcastedNode
Wraps a distribution or a child node by broadcasting it over the parameters via a BroadcastedDistribution.
You will need to take care of matching dimensionalities in the graph.
"""
struct BroadcastedNode{name,child_names,M,N<:NamedTuple{child_names}} <: AbstractNode{name,child_names}
    # Must be function to avoid UnionAll type instabilities
    model::M
    children::N
end

# Convenience constructor for moving name to the parametric type
BroadcastedNode(name::Symbol, model::M, children::N) where {child_names,M,N<:NamedTuple{child_names}} = BroadcastedNode{name,child_names,M,N}(model, children)

# Construct as parent
"""
    BroadcastedNode(name, dist, children)
Construct a node which automatically broadcasts the `distribution` over the parameters given by the `children`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims for the minimal realization of the distribution given the `children`.
"""
function BroadcastedNode(name::Symbol, distribution::Type, children::NamedTuple)
    # Workaround so D is not UnionAll but interpreted as constructor
    # No reduction by default
    sacrifice_model = BroadcastedDistribution | (distribution, ())
    sacrifice_node = BroadcastedNode(name, sacrifice_model, children)
    sacrifice_values = rand(sacrifice_node)
    dims = param_dims(varvalue(sacrifice_node, sacrifice_values))
    wrapped = BroadcastedDistribution | (distribution, dims)
    BroadcastedNode(name, wrapped, children)
end

# Construct as leaf
"""
    BroadcastedNode(name, dist, params...)
Construct the node as leaf (no children) by broadcasting the `distribution` over the `params`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims of the `params`.
"""
BroadcastedNode(name::Symbol, distribution, params...) = BroadcastedNode(name, ProductBroadcastedDistribution(distribution, params...), (;))
