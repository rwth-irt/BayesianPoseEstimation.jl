# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2022, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

"""
    SimpleNode
Basic implementation of an AbstractNode:
Represents a named variable and depends on child nodes.
Does not support logdensityof multiple samples, since no broadcasting or reduction is implemented.
"""
struct SimpleNode{name,child_names,N<:NamedTuple{child_names},R<:AbstractRNG,M<:Union{Distribution,Function}} <: AbstractNode{name,child_names}
    children::N
    rng::R
    # Must be function to avoid UnionAll type instabilities
    model::M
end

SimpleNode(name::Symbol, children::N, rng::R, model::M) where {child_names,N<:NamedTuple{child_names},R<:AbstractRNG,M<:Union{Distribution,Function}} = SimpleNode{name,child_names,N,R,M}(children, rng, model)

# construct as parent
function SimpleNode(name::Symbol, children::NamedTuple, rng::AbstractRNG, ::Type{distribution}) where {distribution}
    # Workaround so D is not UnionAll but interpreted as constructor
    wrapped(x...) = distribution(x...)
    SimpleNode(name, children, rng, wrapped)
end

# construct as leaf
SimpleNode(name::Symbol, rng::AbstractRNG, model) = SimpleNode(name, (;), rng, model)