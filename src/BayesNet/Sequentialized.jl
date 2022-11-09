# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using DensityInterface
using Random
using Unrolled

const SequentializedGraph = NamedTuple{<:Any,<:Tuple{Vararg{AbstractNode}}}

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

"""
    rand(graph, dims...)
Type stable implementation to generate random values from the variables of the sequentialized graph.
"""
Base.rand(graph::SequentializedGraph, dims::Integer...) = rand_unroll(values(graph), (;), dims...)
# unroll required for type stability
@unroll function rand_unroll(graph, variables, dims::Integer...)
    @unroll for node in graph
        value = rand_barrier(node, variables, dims...)
        variables = merge_value(variables, node, value)
    end
    variables
end

"""
    logdensityof(graph, nt)
Type stable implementation to calculate the logdensity for a set of variables for the sequentialized graph.
"""
DensityInterface.logdensityof(graph::SequentializedGraph{names}, nt::NamedTuple) where {names} =
    reduce(.+, map(values(graph), values(nt[names])) do node, value
        logdensityof(node(nt), value)
    end)
# still don't get why reduce(.+, map...) is type stable but mapreduce(.+,...) not
