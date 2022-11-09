# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Random

"""
    ModifierNode
Wraps another node and represents the same variable as the `wrapped` node.
`rand(model, wrapped_value)` and `logdensityof(model, wrapped_ℓ)` allow to modify the value returned by the wrapped node by passing the returned value to the model.
When traversing the graph, only the wrapped node is returned by `nodes`. 
"""
struct ModifierNode{name,child_names,N<:AbstractNode{name,child_names},R<:AbstractRNG,M} <: AbstractNode{name,child_names}
    wrapped_node::N
    rng::R
    model::M
end

children(node::ModifierNode) = (node.wrapped_node,)

function rand_barrier(node::ModifierNode, variables::NamedTuple, dims...)
    wrapped_value = rand_barrier(node.wrapped_node, variables, dims...)
    rand(rng(node), node(variables), wrapped_value)
end

function logdensityof_barrier(node::ModifierNode, variables::NamedTuple)
    wrapped_ℓ = logdensityof_barrier(node.wrapped_node, variables)
    logdensityof(node(variables), varvalue(node, variables), wrapped_ℓ)
end

bijector_barrier(node::ModifierNode, variables::NamedTuple) = bijector_barrier(node.wrapped_node, variables)
