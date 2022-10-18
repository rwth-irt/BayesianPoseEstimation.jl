# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# TODO Split into separate project / library
using Bijectors
using DensityInterface
using Random
using Unrolled

"""
    AbstractNode{name,children}
Construct a directed acyclic graph (DAG), i.e. a Bayesian network, where each variable conditionally depends on a set of parent variables ([Wikpedia](https://en.wikipedia.org/wiki/Bayesian_network)).
By convention, each node represents a variable and has a unique name associated to it.
Exceptions can be made for specific node implementations, i.e. a modifier which post-processes the result of its child node.
`name` is typically a symbol and `children` a tuple of symbols

# Naming Convention
Naming of parent-child relationship is reversed in a Bayesian network compared to DAGs.
The probability of a child variable y given a parent variable x is p(y|x).
However, node x is the parent of node y in the resulting graph x→y.

Programming is done more intuitively using the graph & node notation, thus we use parent x → child y.
"""
abstract type AbstractNode{name,children} end

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
    for n in nodes(node)
        variables = traverse(fn, n, variables, args...)
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

Base.rand(rng::AbstractRNG, node::AbstractNode{varname}, dims::Integer...) where {varname} = traverse(rand_barrier, node, (;), rng, dims...)

DensityInterface.logdensityof(node::AbstractNode, nt::NamedTuple) =
    reduce(.+, traverse(node, (;)) do n, _
        logdensityof_barrier(n, nt)
    end)

function Bijectors.bijector(node::AbstractNode)
    variables = rand(node)
    traverse(node, (;), variables) do n, _...
        bijector_barrier(n, variables)
    end
end

# The following functions help with type stability of internal codes and makes it possible to define custom behavior by dispatching on a specialized node type

rand_barrier(node::AbstractNode{<:Any,()}, variables::NamedTuple, rng::AbstractRNG, dims...) = rand(rng, node(variables), dims...)
# dims... only for leafs, otherwise the dimensioms potentiate
rand_barrier(node::AbstractNode{<:Any}, variables::NamedTuple, rng::AbstractRNG, dims...) = rand(rng, node(variables))

logdensityof_barrier(node::AbstractNode, variables::NamedTuple) = logdensityof(node(variables), varvalue(node, variables))

bijector_barrier(node::AbstractNode, variables::NamedTuple) = bijector(node(variables))

# Helpers for the concrete realization of the internal model by extracting the matching variables

(node::AbstractNode)(x...) = model(node).(x...)
(node::AbstractNode)(nt::NamedTuple) = node(argvalues(node, nt)...)

# Override if required for specific implementations
argvalues(::AbstractNode{<:Any,childnames}, nt) where {childnames} = values(nt[childnames])
varvalue(::AbstractNode{varname}, nt) where {varname} = nt[varname]

nodes(node::AbstractNode) = node.nodes
model(node::AbstractNode) = node.model

Base.Broadcast.broadcastable(x::AbstractNode) = Ref(x)



