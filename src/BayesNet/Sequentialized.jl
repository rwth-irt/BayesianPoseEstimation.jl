# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using DensityInterface
using Random
using Unrolled


"""
    Sequentialized.jl
Since the BayesNet is a directed acyclic graph, there is exactly one shortest path to sequentially call the nodes, so each node has the children values available upon execution.
This allows to implement type stable functions by unrolling the loop over the sequence.
"""

"""
    sequentialize(node)
Finds the shortest path where each node is executed exactly once via depth search.
The result is an ordered NamedTuple. 
"""
sequentialize(node::AbstractNode) =
    traverse(node, (;)) do node, _
        node
    end
sequentialize(graph::SequentializedGraph) = graph

"""
    rand(graph, [variables, dims...])
Type stable implementation to generate random values from the variables of the sequentialized graph.
The `variables` parameter allows to condition the model and will not be re-sampled.
"""
Base.rand(graph::SequentializedGraph, variables::NamedTuple, dims::Integer...) = rand_unroll(values(graph), variables, dims...)
Base.rand(graph::SequentializedGraph, dims::Integer...) = rand(graph, (;), dims...)

# unroll required for type stability
@unroll function rand_unroll(graph, variables, dims::Integer...)
    @unroll for node in graph
        if !(name(node) in keys(variables))
            value = rand_barrier(node, variables, dims...)
            variables = merge_value(variables, node, value)
        end
    end
    variables
end

"""
    evaluate(graph, variables)
Type stable version to only the deterministic nodes in the `graph` given the random `variables`.
All required random variables are assumed to be available.
"""
function evaluate(graph::SequentializedGraph, variables::NamedTuple)
    nt = map(graph) do node
        evaluate_barrier(node, variables)
    end
    merge(variables, nt)
end

"""
    logdensityof(graph, nt)
Type stable implementation to calculate the logdensity for a set of variables for the sequentialized graph.
"""
DensityInterface.logdensityof(graph::SequentializedGraph{names}, nt::NamedTuple) where {names} =
    reduce(add_logdensity, map(values(graph)) do node
        logdensityof_barrier(node, nt)
    end)
# still don't get why reduce(.+, map...) is type stable but mapreduce(.+,...) not

# Support for empty models (e.g. no prior)
DensityInterface.logdensityof(graph::SequentializedGraph{()}, nt::NamedTuple) = 0

"""
    bijector(node)
Infer the bijectors of the sequentialized graph.
"""
function Bijectors.bijector(graph::SequentializedGraph)
    variables = rand(graph)
    map(x -> bijector_barrier(x, variables), graph)
end
