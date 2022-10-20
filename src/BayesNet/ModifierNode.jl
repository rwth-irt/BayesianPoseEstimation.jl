# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Random

"""
    ModifierNode
Wraps another node and represents the same variable as the `wrapped` node.
`rand(rng, model, wrapped_value)` and `logdensityof(model, wrapped_ℓ)` allow to modify the value returned by the wrapped node by passing the returned value to the model.
When traversing the graph, only the wrapped node is returned by `nodes`. 
"""
struct ModifierNode{name,child_names,M,N<:AbstractNode{name,child_names}} <: AbstractNode{name,child_names}
    model::M
    wrapped::N
end

children(node::ModifierNode) = (node.wrapped,)

function rand_barrier(node::ModifierNode, variables::NamedTuple, rng::AbstractRNG, dims...)
    wrapped_value = rand_barrier(node.wrapped, variables, rng, dims...)
    rand(rng, node(variables), wrapped_value)
end

function logdensityof_barrier(node::ModifierNode, variables::NamedTuple)
    wrapped_ℓ = logdensityof_barrier(node.wrapped, variables)
    logdensityof(node(variables), varvalue(node, variables), wrapped_ℓ)
end

bijector_barrier(node::ModifierNode, variables::NamedTuple) = bijector_barrier(node.wrapped, variables)