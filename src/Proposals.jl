# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using Random

"""
    AbstractProposal
Has a `model` which supports rand() and logpdf
"""
abstract type AbstractProposal end

"""
    rand(rng, q)
Generate a new raw state `θ` using the `model` of the proposal `q`.
Whether the sample is Constrained or not is determined 
"""
Base.rand(rng::Random.AbstractRNG, q::AbstractProposal) = rand(rng, q.model)

"""
    rand(q)
Generate a new raw state `θ` using the `model` of the proposal `q`.
"""
Base.rand(q::AbstractProposal) = rand(Random.GLOBAL_RNG, q)

"""
    logpdf(q, θ)
Evaluate the logpdf of a state given the `model` form the proposal `q`.
"""
logpdf(q::AbstractProposal, θ) = logpdf(q.model | θ)

"""
    logpdf(q, θ)
Evaluate the logpdf of a sample given the `model` form the proposal `q`.
"""
logpdf(q::AbstractProposal, s::Sample) = logpdf(q | state(s))

"""
    propose(q, θ)
For the general case of dependent samples.
"""
propose(q::AbstractProposal, θ) = θ + rand(q)

"""
    propose(q, θ)
For the general case of dependent samples.
"""
function propose(q::AbstractProposal, s::Sample)
    @set s.θ = propose(q, s.θ)
end

"""
    propose(q, θ)
For the general case of dependent samples.
"""
transition_probability(q::AbstractProposal, θ, θ_cond) = logpdf(q, θ - θ_cond)


"""
    Proposal
Propose samples from the previous one by using a general proposal distribution.
"""
struct Proposal <: AbstractProposal
    model
end


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
transition_probability(q::SymmetricProposal, θ, θ_cond) = 0


"""
    IndependentProposal
Propose samples independent from the previous one.
"""
struct IndependentProposal <: AbstractProposal
    model
end

"""
    propose(q, θ)
Independent samples are just random values from the model.
"""
propose(q::IndependentProposal, θ) = rand(q)

"""
    propose(q, θ)
For independent proposals, the transition probability does not depend on the previous sample.

"""
transition_probability(q::IndependentProposal, θ, θ_cond) = logpdf(q, θ)

"""
    propose(q)
This method returns a new sample without a prior sample.
"""
function propose(q::IndependentProposal)
    θ = rand(q)
    tr = xform(q.model)
    Sample(θ, -Inf, tr)
end

#TODO Does GibbsProposal make sense? Each Gibbs variable-block belongs to a sampler like MH, ConditionalGibbs, ...

"""
    GibbsProposal
Allows different proposals for different sets of state variables.
"""
struct GibbsProposal <: AbstractProposal
    # TODO type safe data structure? Min. iterable
    # TODO should should the model have the correct form (required variables as args, ist it predictive???) or have models and a set of variables. The latter would probably require transforming the model over and over again. Better provide convenience constructor for generating the correct model.
    model
    proposals::Vector{AbstractProposal}
end


# TODO does it make sense to always sample the variables which are in the hierarchy below the previous variable?
"""
    rand(rng, q)
Generate a new sample randomly using one of the `models` given the other parameters.
"""
function Base.rand(rng::Random.AbstractRNG, q::GibbsProposal)
    # TODO Random does not make sense because sampling algorithm needs to know, which sampler is used for the Block (MH, ConditionalGibbs, etc...)
    m = rand(q.models)
    rand(rng, m)
end
