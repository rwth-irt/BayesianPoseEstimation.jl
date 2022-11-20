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

"""
    SequentializedGraph
A graph that can be executed in sequence compared to traversing the graph.
Since Bayseian networks are DAGs, they can always be sequentialized.
"""
const SequentializedGraph = NamedTuple{<:Any,<:Tuple{Vararg{AbstractNode}}}

# These fields are expected to be available in <:AbstractNode for the default implementations of rand_barrier and logdensityof_barrier
children(node::AbstractNode) = node.children
model(node::AbstractNode) = node.model
name(::AbstractNode{NAME}) where {NAME} = NAME
rng(node::AbstractNode) = node.rng

# Interface: define custom behavior by dispatching on a specialized node type
# Also help with type stability

rand_barrier(node::AbstractNode{<:Any,()}, variables::NamedTuple, dims...) = rand(rng(node), node(variables), dims...)
# do not use dims.. in parent nodes which would lead to dimsᴺ where N=depth of the graph
rand_barrier(node::AbstractNode, variables::NamedTuple, dims...) = rand(rng(node), node(variables))

# Do only evaluate DeterministicNodes
evaluate_barrier(node::AbstractNode, variables::NamedTuple) = varvalue(node, variables)
# evaluate_barrier(::AbstractNode, ::NamedTuple) = nothing

logdensityof_barrier(node::AbstractNode, variables::NamedTuple) = logdensityof(node(variables), varvalue(node, variables))

bijector_barrier(node::AbstractNode, variables::NamedTuple) = bijector(node(variables))

"""
    traverse(fn, node, variables, [args...])
Effectively implements a depth first search to all nodes of the graph.

`fn(node, variables, args...)` is a function of the current `node`, the `variables` gathered from the recursions and `args` of the traverse function.
The return values of `fn` are accumulated in a NamedTuple indexed by the node name.
Only the first value of a node is considered, repeated calls for the same node name are ignored.
If `nothing` is returned, the value is ignored. p…
"""
function traverse(fn, node::AbstractNode{name}, variables::NamedTuple{varnames}, args...) where {name,varnames}
    #It is crucial that each node is executed only once for random sampling:
    # If a node is sampled multiple times for different paths, the variables are not consistent to each other.
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
If the value is nothing, the variable does not get merged
"""
merge_value(variables, ::AbstractNode{name}, value) where {name} = (; variables..., name => value)
merge_value(variables, ::AbstractNode, ::Nothing) = variables

# Model interface

"""
    rand(node, [variables, dims...])
Generate the random variables from the model by traversing the child nodes.
Each node is evaluated only once and the dims are only applied to leafs.
The `variables` parameter allows to condition the model and will not be re-sampled.
"""
Base.rand(node::AbstractNode{varname}, variables::NamedTuple, dims::Integer...) where {varname} = traverse(rand_barrier, node, variables, dims...)
Base.rand(node::AbstractNode, dims::Integer...) = rand(node, (;), dims...)

"""
    evaluate(node, variables)
Evaluate only the deterministic nodes in the graph given the random `variables`.
All required random variables are assumed to be available.
"""
function evaluate(node::AbstractNode, variables::NamedTuple)
    # pass empty `variables` to traverse to evaluate all nodes
    nt = traverse(node, (;)) do current, _
        evaluate_barrier(current, variables)
    end
    merge(variables, nt)
end

"""
    logdensityof(node, variables)
Calculate the logdensity of the model given the variables by traversing the child nodes.
Each node is evaluated only once.
"""
# TODO promote before reduce
DensityInterface.logdensityof(node::AbstractNode, variables::NamedTuple) = reduce(add_logdensity,
    traverse(node, (;)) do current, _
        logdensityof_barrier(current, variables)
    end)

"""
    bijector(node)
Infer the bijectors of the model by traversing the child nodes.
Internally a random is used to instantiate the models.
"""
function Bijectors.bijector(node::AbstractNode)
    variables = rand(node)
    traverse(node, (;), variables) do current, _...
        bijector_barrier(current, variables)
    end
end

"""
    prior(node)
The prior of a node are the leaf children.
Returns a SequentializedGraph for the prior 
"""
prior(node::AbstractNode) =
    traverse(node, (;)) do current, _
        if is_leaf(current)
            return current
        else
            return nothing
        end
    end

"""
    parents(root, node_name)
Returns a SequentializedGraph for the parents for the given `node_name` up until the `root` node.
"""
parents(root::AbstractNode, node_name) =
    traverse(root, (;)) do current, variables
        # current node is parent
        if node_name in keys(children(current))
            return current
        end
        # one of the child nodes is parent
        if isempty(variables)
            return nothing
        end
        is_parent = mapreduce(|, keys(variables)) do var_name
            var_name in keys(children(current))
        end
        if is_parent
            return current
        end
        return nothing
    end

parents(root::AbstractNode, nodes::AbstractNode...) =
    reduce(nodes; init=(;)) do accumulated, node
        nt = parents(root, name(node))
        # Merge only nodes which are not present in the evaluation model yet
        diff_nt = Base.structdiff(nt, accumulated)
        merge(accumulated, diff_nt)
    end

# Help to extract values from samples (NamedTuples)
childvalues(::AbstractNode{<:Any,child_names}, nt::NamedTuple) where {child_names} = values(nt[child_names])
varvalue(::AbstractNode{name}, nt::NamedTuple) where {name} = nt[name]
is_leaf(node::AbstractNode) = children(node) == (;)

# Helpers for the concrete realization of the internal model by extracting the matching variables
(node::AbstractNode)(x...) = model(node)(x...)
(node::AbstractNode)(nt::NamedTuple) = node(childvalues(node, nt)...)
# leaf does not depend on any other variables and should have a fully specified model
(node::AbstractNode{<:Any,()})(x...) = model(node)
(node::AbstractNode{<:Any,()})(::NamedTuple) = model(node)

# Base implementations
Base.Broadcast.broadcastable(x::AbstractNode) = Ref(x)
Base.show(io::IO, node::T) where {varname,child_names,T<:AbstractNode{varname,child_names}} = print(io, "$(Base.typename(T).wrapper){:$varname, $child_names}")
