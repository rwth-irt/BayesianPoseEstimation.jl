# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Random

+(a::NamedTuple{T,U}, b::NamedTuple{T,U}) where {T,U} = NamedTuple{T,U}(collect(a) + collect(b))

-(a::NamedTuple{T,U}, b::NamedTuple{T,U}) where {T,U} = NamedTuple{T,U}(collect(a) - collect(b))

"""
    AbstractProposal
Has a `model` which supports rand() and logdensity
"""
abstract type AbstractProposal end

Base.rand(rng::Random.AbstractRNG, q::AbstractProposal) = rand(rng, q.model)

Base.rand(q::AbstractProposal) = rand(Random.GLOBAL_RNG, q.model)

logdensity(q::AbstractProposal, θ) = logdensity(q.model | θ)

"""
    propose(q, θ)
For the general case of dependent samples.
"""
propose(q::AbstractProposal, θ) = θ + rand(q)

"""
    propose(q, θ)
For the general case of dependent samples.
"""
transition_probability(q::AbstractProposal, θ, θ_cond) = logdensity(q, θ - θ_cond)

"""
    Proposal
Propose samples from the previous one by using a general proposal distribution.
"""
struct Proposal <: AbstractProposal
    model
end

"""
    IndependentProposal
Propose samples independent from the previous one.
"""
struct IndependentProposal <: AbstractProposal
    model
end

"""
    propose(q, θ)
For the general case of dependent samples.
"""
propose(q::IndependentProposal, θ) = rand(q)

"""
    propose(q, θ)
For the general case of dependent samples.
"""
transition_probability(q::IndependentProposal, θ, θ_cond) = logdensity(q, θ)


"""
    SymmetricProposal
Propose samples from the previous one by using a symmetric proposal distribution.
"""
struct SymmetricProposal <: AbstractProposal
    model
end

"""
    propose(q, θ)
For symmetric proposals, the forward and backward transition probability cancels out
"""
transition_probability(q::IndependentProposal, θ, θ_cond) = 0
