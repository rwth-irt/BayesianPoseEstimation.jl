# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

# TODO Split into separate project / library
using Bijectors
using DensityInterface
using Random
using Unrolled

"""
    AbstractNode{name,child_names}
Construct a directed acyclic graph (DAG), i.e. a Bayesian network, where each variable conditionally depends on a set of parent variables ([Wikpedia](https://en.wikipedia.org/wiki/Bayesian_network)).
By convention, each node represents a variable and has a unique name associated to it.
Exceptions can be made for specific node implementations, i.e. a modifier which post-processes the result of its child node.
`name` is typically a symbol and `child_names` a tuple of symbols

# Naming Convention
Naming of parent-child relationship is reversed in a Bayesian network compared to DAGs.
The probability of a child variable y given a parent variable x is p(y|x).
However, node x is the parent of node y in the resulting graph x→y.

Programming is done more intuitively using the graph & node notation, thus we use parent x → child y.
"""
abstract type AbstractNode{name,child_names} end

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
    rand(rng, node, dims...)
Generate the random variables from the model by traversing the child nodes.
Each node is evaluated only once and the dims are only applied to leafs.
"""
Base.rand(rng::AbstractRNG, node::AbstractNode{varname}, dims::Integer...) where {varname} = traverse(rand_barrier, node, (;), rng, dims...)

"""
    logdensityof(node, variables)
Calculate the logdensity of the model given the variables by traversing the child nodes.
Each node is evaluated only once.
"""
DensityInterface.logdensityof(node::AbstractNode, variables::NamedTuple) =
    reduce(.+, traverse(node, (;)) do child, _
        logdensityof_barrier(child, variables)
    end)

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
argvalues(::AbstractNode{<:Any,child_names}, nt) where {child_names} = values(nt[child_names])
varvalue(::AbstractNode{varname}, nt) where {varname} = nt[varname]

children(node::AbstractNode) = node.children
model(node::AbstractNode) = node.model

Base.Broadcast.broadcastable(x::AbstractNode) = Ref(x)


"""
    VariableNode
Basic implementation of an AbstractNode:
Represents a named variable and depends on child nodes.
"""
struct VariableNode{name,child_names,M<:Function,N<:NamedTuple{child_names}} <: AbstractNode{name,child_names}
    # Must be function to avoid UnionAll type instabilities
    model::M
    children::N
end

VariableNode(name::Symbol, model::M, children::N) where {names,M<:Function,N<:NamedTuple{names}} = VariableNode{name,names,M,N}(model, children)

function VariableNode(name::Symbol, ::Type{M}, children::N) where {names,M,N<:NamedTuple{names}}
    # Workaround so D is not UnionAll but interpreted as constructor
    wrapped = (x...) -> M(x...)
    VariableNode{name,names,typeof(wrapped),N}(wrapped, children)
end


# TODO could be a hijacked VariableNode, replacing the model with a broadcasted one and bypassing it by returning its children
function BroadcastedNode(node::AbstractNode{name}, dims...) where {name}
    broadcasted_model(x...) = BroadcastedDistribution(model(node), dims, x...)
    VariableNode(name, broadcasted_model, children(node))
end


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

nodes(node::ModifierNode) = (node.wrapped,)

function rand_barrier(node::ModifierNode, variables::NamedTuple, rng::AbstractRNG, dims...)
    wrapped_value = rand_barrier(node.wrapped, variables, rng, dims...)
    rand(rng, node(variables), wrapped_value)
end

function logdensityof_barrier(node::ModifierNode, variables::NamedTuple)
    wrapped_ℓ = logdensityof_barrier(node.wrapped, variables)
    logdensityof(node(variables), varvalue(node, variables), wrapped_ℓ)
end

bijector_barrier(node::ModifierNode, variables::NamedTuple) = bijector_barrier(node.wrapped, variables)


"""
    sequentialize(node)
Since the BayesNet is a directed acyclic graph, there is exactly one shortest path to sequentially call the nodes, so each node has the children values available upon execution.
This allows to implement type stable functions by unrolling the loop over the sequence.
Finds that path by depth search and returns it as an ordered NamedTuple. 
"""
sequentialize(node::AbstractNode) =
    traverse(node, (;)) do node, _
        node
    end

"""
    rand(rng, graph, dims...)
Type stable implementation to generate random values from the variables of the sequentialized graph.
"""
Base.rand(rng::AbstractRNG, graph::NamedTuple{<:Any,<:Tuple{Vararg{AbstractNode}}}, dims::Integer...) = rand_unroll(rng, values(graph), (;), dims...)
# unroll required for type stability
@unroll function rand_unroll(rng::AbstractRNG, graph, variables, dims::Integer...)
    @unroll for node in graph
        value = rand_barrier(node, variables, rng, dims...)
        variables = merge_value(variables, node, value)
    end
    variables
end

"""
    rand(rng, graph, dims...)
Type stable implementation to calculate the logdensity for a set of variables for the sequentialized graph.
"""
DensityInterface.logdensityof(graph::NamedTuple{names,<:Tuple{Vararg{AbstractNode}}}, nt::NamedTuple) where {names} =
    reduce(.+, map(values(graph), values(nt[names])) do node, value
        logdensityof(node(nt), value)
    end)
# still don't get why reduce(.+, map...) is type stable but mapreduce(.+,...) not
