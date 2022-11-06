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
# TODO doc that it is constructed as product distribution
"""
    BroadcastedNode(name, dist, params...)
Construct the node as leaf (no children) by broadcasting the `distribution` over the `params`.
The resulting `BroadcastedDistribution` acts like a product distribution, reducing the ndims of the `params`.
"""
BroadcastedNode(name::Symbol, distribution, params...) = BroadcastedNode(name, ProductBroadcastedDistribution(distribution, params...), (;))


# TODO Do I want to enable conversion or should a BroadcastedNode be the default?
# function broadcast_node(node::AbstractNode{name}) where {name}
#     variables = rand(node)
#     traverse(node, (;), variables) do child, _...
#         broadcast_barrier(child, variables)
#     end
# end

# function broadcast_barrier(node::AbstractNode{name}, variables::NamedTuple) where {name}
#     dims = param_dims(variables[name])
#     # TODO bake in the dims using partial application
#     wrapped = BroadcastedDistribution(model(node), dims, x...)
#     BroadcastedNode(name, broadcasted_model, children(node))
# end

# BroadcastedNode(name, model::M, children::N) where {child_names,M,N<:NamedTuple{child_names}} = BroadcastedNode{name,child_names,M,N}(model, children)

# # construct as parent
# """
#     BroadcastedNode(node, dims)
# Wraps the model of the `node` with a BroadcastedDistribution of `dims` reduction dimensions.
# """
# function BroadcastedNode(node::AbstractNode{name}, dims::Dims) where {name}
#     broadcasted_model(x...) = BroadcastedDistribution(model(node), dims, x...)
#     BroadcastedNode(name, broadcasted_model, children(node))
# end

# # construct as leaf
# function BroadcastedNode(name::Symbol, model, dims::Dims, args...)
#     broadcasted_model = BroadcastedDistribution(model, dims, args...)
#     BroadcastedNode(name, broadcasted_model, (;))
# end

# """
#     ProductNode(node)
# Wraps the model of the `node` with a ProductBroadcastedDistribution which reduces the `dims` of the arguments.
# """
# function ProductNode(node::AbstractNode{name}) where {name}
#     product_model(x...) = ProductBroadcastedDistribution(model(node), x...)
#     BroadcastedNode(name, product_model, children(node))
# end

# # construct as leaf
# function ProductNode(name::Symbol, model, args...)
#     product_model = ProductBroadcastedDistribution(model, args...)
#     BroadcastedNode(name, product_model, (;))
# end


# # Do not broadcast, BroadcastedDistribution does this internally
# (node::BroadcastedNode)(x...) = model(node)(x...)
# (node::BroadcastedNode{<:Any,()})(x...) = model(node)
# # Avoid method ambiguities, same as AbstractNode(::NamedTuple)
# (node::BroadcastedNode)(nt::NamedTuple) = node(childvalues(node, nt)...)
# (node::BroadcastedNode{<:Any,()})(::NamedTuple) = model(node)

