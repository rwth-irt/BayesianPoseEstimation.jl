# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 


"""
    SimpleNode
Basic implementation of an AbstractNode:
Represents a named variable and depends on child nodes.
Does not support logdensityof multiple samples, since no broadcasting or reduction is implemented.
"""
struct SimpleNode{name,child_names,M,N<:NamedTuple{child_names}} <: AbstractNode{name,child_names}
    # Must be function to avoid UnionAll type instabilities
    model::M
    children::N
end

SimpleNode(name::Symbol, model::M, children::N) where {child_names,M,N<:NamedTuple{child_names}} = SimpleNode{name,child_names,M,N}(model, children)

# construct as parent
function SimpleNode(name::Symbol, ::Type{distribution}, children::NamedTuple) where {distribution}
    # Workaround so D is not UnionAll but interpreted as constructor
    wrapped(x...) = distribution(x...)
    SimpleNode(name, wrapped, children)
end

# construct as leaf
SimpleNode(name::Symbol, model::M) where {M} = SimpleNode(name, model, (;))
