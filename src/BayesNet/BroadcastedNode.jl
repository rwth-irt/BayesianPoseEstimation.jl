# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.


# TODO Should only the returned variables of the children be broadcasted or the whole subgraph? Modifiers could be unpredictable

struct BroadcastedNode{name,child_names,M,N<:NamedTuple{child_names}} <: AbstractNode{name,child_names}
    # Must be function to avoid UnionAll type instabilities
    model::M
    children::N
end

BroadcastedNode(name, model::M, children::N) where {child_names,M,N<:NamedTuple{child_names}} = BroadcastedNode{name,child_names,M,N}(model, children)

# construct as parent
"""
    BroadcastedNode(node, dims)
Wraps the model of the `node` with a BroadcastedDistribution of `dims` reduction dimensions.
"""
function BroadcastedNode(node::AbstractNode{name}, dims::Dims) where {name}
    broadcasted_model(x...) = BroadcastedDistribution(model(node), dims, x...)
    BroadcastedNode(name, broadcasted_model, children(node))
end

# construct as leaf
function BroadcastedNode(name::Symbol, model, dims::Dims, args...)
    broadcasted_model = BroadcastedDistribution(model, dims, args...)
    BroadcastedNode(name, broadcasted_model, (;))
end

"""
    ProductNode(node)
Wraps the model of the `node` with a ProductBroadcastedDistribution which reduces the `dims` of the arguments.
"""
function ProductNode(node::AbstractNode{name}) where {name}
    product_model(x...) = ProductBroadcastedDistribution(model(node), x...)
    BroadcastedNode(name, product_model, children(node))
end

# construct as leaf
function ProductNode(name::Symbol, model, args...)
    product_model = ProductBroadcastedDistribution(model, args...)
    BroadcastedNode(name, product_model, (;))
end


# Do not broadcast since this is done internally
(node::BroadcastedNode)(x...) = model(node)(x...)
(node::BroadcastedNode{<:Any,()})(x...) = model(node)
# Avoid method ambiguities, same as AbstractNode(::NamedTuple)
(node::BroadcastedNode)(nt::NamedTuple) = node(childvalues(node, nt)...)
(node::BroadcastedNode{<:Any,()})(::NamedTuple) = model(node)

