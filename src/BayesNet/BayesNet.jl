# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# TODO Split into separate project / library
using Bijectors
using DensityInterface
using Random

"""
    AbstractNode{name,child_names}
Construct a directed acyclic graph (DAG), i.e. a Bayesian network, where each variable conditionally depends on a set of parent variables ([Wikpedia](https://en.wikipedia.org/wiki/Bayesian_network)).

By convention, each node represents a variable and has a unique name associated to it.
Exceptions can be made for specific node implementations, i.e. a modifier which post-processes the result of its child node.
`name` is typically a symbol and `child_names` a tuple of symbols

If a node is a leaf (has nod children), it does not depend on any other variables and should have a fully specified model.

# Naming Convention
Naming of parent-child relationship is reversed in a Bayesian network compared to DAGs.
The probability of a child variable y given a parent variable x is p(y|x).
However, node x is the parent of node y in the resulting graph x→y.

Programming is done more intuitively using the graph & node notation, thus we use parent x → child y.
"""
abstract type AbstractNode{name,child_names} end

# These fields are expected to be available in <:AbstractNode for the default implementations of rand_barrier and logdensityof_barrier
children(node::AbstractNode) = node.children
model(node::AbstractNode) = node.model
name(::AbstractNode{name}) where {name} = name
rng(node::AbstractNode) = node.rng

# Interface: define custom behavior by dispatching on a specialized node type
# Also help with type stability

rand_barrier(node::AbstractNode{<:Any,()}, variables::NamedTuple, dims...) = rand(rng(node), node(variables), dims...)
# do not use dims.. in parent nodes which would lead to dimsᴺ where N=depth of the graph
rand_barrier(node::AbstractNode, variables::NamedTuple, dims...) = rand(rng(node), node(variables))

logdensityof_barrier(node::AbstractNode, variables::NamedTuple) = logdensityof(node(variables), varvalue(node, variables))

bijector_barrier(node::AbstractNode, variables::NamedTuple) = bijector(node(variables))

"""
    traverse(fn, node, variables, [args...])
Effectively implements a depth first search to all nodes of the graph.
It is crucial that each node is executed only once for random sampling:
If a node is sampled multiple times for different paths, the variables are not consistent to each other.
"""
function traverse(fn, node::AbstractNode{name}, variables::NamedTuple{varnames}, args...) where {name,varnames}
    # Termination: Value already available (conditioned on or calculate via another path)
    if name in varnames
        return variables
    end
    # Conditional = values from other nodes required, compute depth first
    for child in children(node)
        variables = traverse(fn, child, variables, args...)
    end
    # Finally the internal dist can be realized and the value for this node can be merged
    value = fn(node, variables, args...)
    merge_value(variables, node, value)
end

"""
    merge_value(variables, node, value)
Right to left merges the value for the node with the correct name into the previously sampled variables.
Allows to override / modify previous values.
"""
merge_value(variables, ::AbstractNode{name}, value) where {name} = (; variables..., name => value)

# Model interface

"""
    rand(node, dims...)
Generate the random variables from the model by traversing the child nodes.
Each node is evaluated only once and the dims are only applied to leafs.
"""
Base.rand(node::AbstractNode{varname}, dims::Integer...) where {varname} = traverse(rand_barrier, node, (;), dims...)

"""
    logdensityof(node, variables)
Calculate the logdensity of the model given the variables by traversing the child nodes.
Each node is evaluated only once.
"""
# TODO promote before reduce
function DensityInterface.logdensityof(node::AbstractNode, variables::NamedTuple)
    ℓ = traverse(node, (;)) do child, _
        logdensityof_barrier(child, variables)
    end
    reduce(add_logdensity, ℓ)
end

"""
    bijector(node)
Infer the bijectors of the model by traversing the child nodes.
Internally a random is used to instantiate the models.
"""
function Bijectors.bijector(node::AbstractNode)
    variables = rand(node)
    traverse(node, (;), variables) do child, _...
        bijector_barrier(child, variables)
    end
end

# Help to extract values from samples (NamedTuples)
childvalues(::AbstractNode{<:Any,child_names}, nt::NamedTuple) where {child_names} = values(nt[child_names])
varvalue(::AbstractNode{name}, nt::NamedTuple) where {name} = nt[name]

# Helpers for the concrete realization of the internal model by extracting the matching variables
(node::AbstractNode)(x...) = model(node)(x...)
(node::AbstractNode)(nt::NamedTuple) = node(childvalues(node, nt)...)
# leaf does not depend on any other variables and should have a fully specified model
(node::AbstractNode{<:Any,()})(x...) = model(node)
(node::AbstractNode{<:Any,()})(::NamedTuple) = model(node)

# Base implementations
Base.Broadcast.broadcastable(x::AbstractNode) = Ref(x)
Base.show(io::IO, node::T) where {varname,child_names,T<:AbstractNode{varname,child_names}} = print(io, "$(Base.typename(T).wrapper){:$varname, $child_names}")
