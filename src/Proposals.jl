# @license BSD-3 https://opensource.org/licenses/BSD-3-Clause
# Copyright (c) 2021, Institute of Automatic Control - RWTH Aachen University
# All rights reserved. 

using MeasureTheory, Soss
using Random

"""
    AbstractProposal
Has a `model` which supports rand() and logdensity
"""
abstract type AbstractProposal end

"""
    Proposal
Propose samples from the previous one by using a general proposal distribution.
"""
struct Proposal{T<:AbstractMeasure} <: AbstractProposal
    model::T

    function Proposal(m::T) where {T<:Soss.AbstractModel}
        # Generally only unconstrained proposal models are supported
        # E.g. an exponential proposal would result in s - c_cond < 0 and thus an invalid logdensity
        if !is_identity(xform(m | (;)))
            throw(DomainError(m, "Model is not unconstrained, i.e. requires maps to a domain which is not ℝ"))
        end
        new{T}(m)
    end
end

"""
    rand(rng, q)
Generate a new raw state `θ` using the `model` of the proposal `q`.
Whether the sample is Constrained or not is determined 
"""
Base.rand(rng::AbstractRNG, q::AbstractProposal) = rand(rng, q.model)

"""
    rand(q)
Generate a new raw state `θ` using the `model` of the proposal `q`.
"""
Base.rand(q::AbstractProposal) = rand(Random.GLOBAL_RNG, q)

"""
    propose(rng, q, s)
For the general case of dependent samples.
"""
propose(rng::AbstractRNG, q::AbstractProposal, s::Sample) = s + rand(rng, q)

"""
    propose(q, s)
For the general case of dependent samples.
"""
propose(q::AbstractProposal, s::Sample) = propose(Random.GLOBAL_RNG, q, s)

"""
    logdensity(q, s)
Evaluate the logdensity of a state given the `model` form the proposal `s`.
"""
# Generally proposals are unconstrained except IndependentProposal
# In the first case, state will be the identity map
MeasureTheory.logdensity(q::AbstractProposal, s::Sample) = logdensity(q.model | state(s))

"""
    transition_probability(q, s, s_cond)
For the general case of dependent samples.
"""
transition_probability(q::AbstractProposal, s::Sample, s_cond::Sample) = logdensity(q, s - s_cond)


"""
    SymmetricProposal
Propose samples from the previous one by using a symmetric proposal distribution.
"""
struct SymmetricProposal{T<:AbstractMeasure} <: AbstractProposal
    model::T

    function SymmetricProposal(m::T) where {T<:AbstractMeasure}
        if !is_identity(as(m))
            throw(DomainError(m, "Model is not unconstrained, i.e. requires maps to a domain which is not ℝ"))
        end
        new{T}(m)
    end

    function SymmetricProposal(m::T) where {T<:Soss.AbstractModel}
        if !is_identity(xform(m | (;)))
            throw(DomainError(m, "Model is not unconstrained, i.e. requires maps to a domain which is not ℝ"))
        end
        new{T}(m)
    end
end

"""
    transition_probability(q, s, s_cond)
For symmetric proposals, the forward and backward transition probability cancels out
"""
transition_probability(q::SymmetricProposal, ::Sample, ::Sample) = 0.0


"""
    IndependentProposal
Propose samples independent from the previous one.
"""
struct IndependentProposal{T<:AbstractMeasure} <: AbstractProposal
    model::T
end

"""
    propose(rng, q)
Independent samples are just random values from the model.
"""
propose(rng::AbstractRNG, q::IndependentProposal) = Sample(rng, q.model)

"""
    propose(q)
Independent samples are just random values from the model.
"""
propose(q::IndependentProposal) = Sample(Random.GLOBAL_RNG, q.model)

"""
    propose(rng, q, θ)
Independent samples are just random values from the model.
"""
propose(rng::AbstractRNG, q::IndependentProposal, ::Sample) = propose(rng, q)

"""
    propose(q, θ)
Independent samples are just random values from the model.
"""
propose(q::IndependentProposal, ::Sample) = propose(Random.GLOBAL_RNG, q)

"""
    transition_probability(q, θ, θ_cond)
For independent proposals, the transition probability does not depend on the previous sample.
"""
# IndependentProposal is the exception: logdensity in constrained domain
transition_probability(q::IndependentProposal, s::Sample, ::Sample) = logdensity(q | state(s))



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
function Base.rand(rng::AbstractRNG, q::GibbsProposal)
    # TODO Random does not make sense because sampling algorithm needs to know, which sampler is used for the Block (MH, ConditionalGibbs, etc...)
    m = rand(q.models)
    rand(rng, m)
end
