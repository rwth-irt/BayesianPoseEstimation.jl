# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved.

using Bijectors
using Random

"""
    DeterministicNode
This node only takes part in the generative `rand` process but is not random.
Instead a deterministic function `fn` is provided, e.g. rendering an image for a pose.
It does not change the joint logdensity of the graph by returning a logdensity of zero.
"""
struct DeterministicNode{name,child_names,M,N<:NamedTuple{child_names}} <: AbstractNode{name,child_names}
    fn::M
    children::N
end

rand_barrier(node::ModifierNode, variables::NamedTuple, ::AbstractRNG, dims...) = model.fn(childvalues(node, variables))

# Do not change the joint probability - log probability of 0
logdensityof_barrier(node::ModifierNode, variables::NamedTuple) = zero(varvalue(node, variables))

# TODO is the assumption of ℝ support correct?
bijector_barrier(node::ModifierNode, variables::NamedTuple) = Bijectors.Identity{0}()